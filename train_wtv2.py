import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import importlib
import logging
import copy
from main_eval import preprocess_predictions_df, evaluate
import pandas as pd
from data import utils_wtv2 as wt_ut
from util import *
from model import *
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


devices = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = devices


# TODO: target tokens and y tokens seem to be the same?
def compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens)
    logits = outputs[0]

    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    #print("")
    #print("---------------------")
    #print("target_tokens.size() = ", target_tokens.size())
    #print("mask.size() = ", mask.size())
    # Perform masking
    # todo: val_step does not perform masking, this might be why it fails with batch_size > 1
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    #print("logits.size() = ", logits.size())
    #print("target_tokens.size() = ", target_tokens.size())
    #print("logits.view(-1, num_logits).size() = ", logits.view(-1, num_logits).size())
    #print("target_tokens.view(-1).size() = ", target_tokens.view(-1).size())
    #print("-" * 20)
    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta * kl_loss

    return loss, ce_loss, kl_loss


def compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn,
                    beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, y_mask=x_mask, y_tokens=x_tokens, from_mean=True,
                    from_prior=False)

    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean()

    return loss, ce_loss, kl_loss


def train_step(device, model, optimizer, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn,
               beta, model_type, with_apex):
    if with_apex:
        from apex import amp

    output = []
    if model_type == 'ae_vae_fusion':
        optimizer.zero_grad()
        loss, ce_loss, kl_loss = compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                                 target_tokens, mask, loss_fn, beta)
        if with_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # max_grad_norm=1.0

        optimizer.step()
        output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))

    optimizer.zero_grad()
    loss, ce_loss, kl_loss = compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                          target_tokens, mask, loss_fn, beta)
    if with_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # max_grad_norm=1.0
    optimizer.step()
    output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))

    return output


def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None,
                    model_type='cvae'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    with torch.no_grad():
        if model_type == 'cvae':
            try:
                prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            except:
                prior_mean = prior_logvar = torch.zeros([batch_size, model.config.n_embd], device=device)
            latent_mean, latent_logvar = prior_mean, prior_logvar
            z = model.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'
        else:
            posterior_mean, posterior_logvar = model.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]
            latent_mean, latent_logvar = posterior_mean, posterior_logvar
            z = latent_mean
            assert not torch.isnan(z).any(), 'training get nan z'

        _, mem = model.transformer(input_ids=x_tokens[:, :-1], past=None, attention_mask=x_mask[:, :-1],
                                   representations=z)
        prev = x_tokens[:, -1].view(batch_size, -1)

        output = prev
        probability = torch.tensor([], dtype=z.dtype, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

        for i in range(length):  # trange
            logits, mem = model.transformer(input_ids=prev, past=mem, representations=z)

            logits = model.lm_head(logits)
            if model.add_softmax:
                logits_rep = model.lm_head_rep(z)
                logits = logits + logits_rep.unsqueeze(dim=1)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)

    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])
    parser.add_argument('--iterations', type=int, default=300001)  # 101640 * 4)  # wp 850001  wi 300001 ax 300001 yp 800001

    parser.add_argument('--warmup', type=int, default=10000,
                        help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")
    parser.add_argument("--fix-pretrained-iters", type=int, default=40000,
                        help="Number of iterations in which to keep pretrained params fixed")
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1],
                        help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[1024],
                        help='seq length per sample. Lists the schedule.')
    parser.add_argument('--switch-time', type=float, default=0,
                        help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--load', type=str, help='path to load model from')  # , default='out/test/'
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
    parser.add_argument('--fp16_opt_level', default='O0', type=str, required=False)

    # KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
    # todo: these are hard coded in the code
    parser.add_argument('--beta_0', default=1.00, type=float)
    parser.add_argument('--iters-no-cyclic-annealing', type=int,
                        help="Number of iterations in the beginning where beta is 1")
    # cyc_vae parameters
    parser.add_argument('--no-cycles', type=int, default=4)

    parser.add_argument('--add_input', action="store_true")
    parser.add_argument('--add_attn', action="store_true")
    parser.add_argument('--add_softmax', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    parser.add_argument('--learn_prior', action="store_true")

    parser.add_argument('--no-similar-hypotheses', type=int, default=20)
    parser.add_argument('--with-retrieval', action="store_true")
    parser.add_argument('--no-facts-to-retrieve', type=int, default=6)
    parser.add_argument('--central-only', action="store_true")

    parser.add_argument('--with-apex', action="store_true")

    args = parser.parse_args('test --batch-sizes 2 --seq-lens 1024 '
                             '--iters-no-cyclic-annealing 2206 --add_input --add_attn --attn_proj_vary '
                             '--learn_prior --lr 5e-5 --fp16 --fp16_opt_level O0 --iterations 13236 --warmup -1 '
                             '--fix-pretrained-iters 2206 --with-apex '
                             '--with-retrieval --no-similar-hypotheses 20 --no-facts-to-retrieve 6'.split())  # wi.12.proj_vary_beta_cvae

    if args.model_type == 'cvae':
        args.learn_prior = True
    else:
        args.learn_prior = False

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # logging
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    importlib.reload(logging)
    logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
                        level=logging.INFO, format='%(asctime)s--- %(message)s')
    logging.info('\n*******************************************************************************\n')
    logging.info("the configuration:")
    logging.info(str(args).replace(',', '\n'))

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808
    config = GPT2Config()

    VAE = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)
    init_para_frompretrained(VAE.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(VAE.encoder, gpt2_model.transformer, share_para=False)
    if args.learn_prior:
        init_para_frompretrained(VAE.encoder_prior, VAE.encoder, share_para=True)
        VAE.encoder_prior.averageSelfAttention.attention_weights = VAE.encoder.averageSelfAttention.attention_weights
    VAE.lm_head.weight = gpt2_model.lm_head.weight
    if VAE.add_softmax:
        VAE.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
    print('VAE_params:', num_params(VAE))  # 286694400
    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(args.load, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        VAE.load_state_dict(state)
        gc.collect()
    print('Done.')

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = args.fix_pretrained_iters
    tuning_all = False
    for name, parameter in VAE.named_parameters():
        # print((name, parameter.requires_grad))
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2',
                    'lm_head_rep']
        # if param is not in the list of new_param it's a pretrained param and must be fixed initially
        if not any([True if n in name else False for n in new_pars]):
            parameter.requires_grad = False

    print('Setup data...')
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    print('Batch schedule', batch_schedule)

    print("train batch size = ", batch_schedule[cur_b_schedule][0])
    print("train seq len = ", batch_schedule[cur_b_schedule][1])

    print("val batch size = ", batch_schedule[-1][0])
    print("val seq len = ", batch_schedule[-1][1])

    train_loader, val_loader, test_loader = wt_ut.prepare_dataset(data_dir=args.data_dir,
                                                                  tokenizer=tokenizer,
                                                                  train_bsz=batch_schedule[cur_b_schedule][0],
                                                                  train_seq_len=batch_schedule[cur_b_schedule][1],
                                                                  val_bsz=1,
                                                                  val_seq_len=batch_schedule[-1][1],
                                                                  test_bsz=batch_schedule[-1][0],
                                                                  test_seq_len=1,
                                                                  num_workers=1,
                                                                  make_train=True,
                                                                  make_val=True,
                                                                  make_test=True,
                                                                  with_retrieval=args.with_retrieval,
                                                                  no_hypotheses=args.no_similar_hypotheses,
                                                                  no_facts=args.no_facts_to_retrieve,
                                                                  central_only=args.central_only
                                                                  )

    print('Done.')

    print('Wrapping models and optimizers...')
    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    VAE = VAE.to(device)
    VAE.train()

    optimizer = AdamW(VAE.parameters(), lr=args.lr, correct_bias=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    if args.with_apex:
        from apex import amp
        VAE, optimizer = amp.initialize(VAE, optimizer, opt_level=args.fp16_opt_level)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')

    print('Begin training iterations')
    logging.info("Begin training iterations")
    max_val_batches = 2000  # max num. of val batches
    logging.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def val_step(val_loader):
        VAE.eval()

        n_words_bpe = 0
        n_words = 0
        logp_sum = 0.0
        kl_loss_sum = 0.0

        logging.info("Validation loop.         Batches: %d" % len(val_loader))
        logging.info("Validation loop. max_val_batches: %d" % max_val_batches)

        with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, topic) in enumerate(
                    val_loader):
                with torch.no_grad():
                    if args.model_type == 'cvae':
                        loss, ce_loss, kl_loss = compute_loss(device, VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                              input_tokens, target_tokens, mask, loss_fn, 1.0)
                    else:
                        loss, ce_loss, kl_loss = compute_loss_ae(device, VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                                 input_tokens, target_tokens, mask, loss_fn, 1.0)

                if len(target_tokens.size()) == 1:
                    target_tokens = target_tokens.unsqueeze(0)
                n, l = target_tokens.size()

                #print("target_tokens.size() = ", target_tokens.size())
                text = target_tokens[0, :].tolist()
                logprob = ce_loss.tolist()
                #print("ce_loss.size() = ", ce_loss.size())
                assert len(text) == len(logprob)

                # only for story
                idx = text.index(endoftext)
                text = text[idx + 1:]
                logprob = logprob[idx + 1:]

                if endoftext in text:
                    idx = text.index(endoftext)
                    text = text[:idx]
                    logprob = logprob[:idx]

                logp_sum += sum(logprob)

                n_words_bpe += len(text)

                story = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                story = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in story]
                story = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                         story]
                words = sum([len(
                    [t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', s) if t != ' ' and t != '']) for
                    s in story])
                n_words += words

                kl_loss_sum += kl_loss.item()

                if i > max_val_batches:
                    break
                pbar.update(1)

        loss_bpe = logp_sum / n_words_bpe
        ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 3)
        ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 3)
        kl = kl_loss_sum / len(val_loader)

        v_writer.add_scalar('loss', loss_bpe, num_iters)
        v_writer.add_scalar('ppl_bpe', ppl_bpe, num_iters)
        v_writer.add_scalar('ppl_word', ppl_word, num_iters)
        v_writer.add_scalar('kl', kl, num_iters)
        logging.info('val loss    : %.4f' % loss_bpe)
        logging.info('val ppl_bpe : %.4f' % ppl_bpe)
        logging.info('val ppl_word: %.4f' % ppl_word)
        logging.info('val   kl    : %.4f' % kl)

        VAE.train()

    def test_plot(val_loader, epoch):
        VAE.eval()

        # get embedding
        X_emb = None
        y = None

        with tqdm(total=len(val_loader)) as pbar:
            for i, (x_mask, x_tokens, _, _, _, _, _, topic) in enumerate(
                    val_loader):

                x_mask = x_mask.to(device)
                x_tokens = x_tokens.to(device)
                with torch.no_grad():
                    if args.model_type == 'cvae':
                        latent_mean, latent_logvar = VAE.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
                    else:
                        latent_mean, latent_logvar = VAE.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]

                if i == 0:
                    X_emb = latent_mean.data
                    y = topic
                else:
                    X_emb = torch.cat((X_emb, latent_mean.data), dim=0)
                    y.extend(topic)
                pbar.update(1)
        X_emb = X_emb.cpu().numpy()

        try:
            X_emb = X_emb[[t.lower() != "none" for t in y], :]
            y = [t for t in y if t.lower() != "none"]

            # to 2D
            #  The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
            #  Larger datasets usually require a larger perplexity.
            #  Consider selecting a value between 5 and 50. Different values can result in significantly different results.
            X_emb_2d = TSNE(n_components=2, verbose=1, perplexity=15).fit_transform(X_emb)

            def remove_outliers(data, r=2.0):
                outliers_data = abs(data - np.mean(data, axis=0)) >= r * np.std(data, axis=0)
                outliers = np.any(outliers_data, axis=1)
                keep = np.logical_not(outliers)
                return outliers, keep

            outliers, keep = remove_outliers(X_emb_2d)
            X_emb_2d = X_emb_2d[keep, :]
            y = [l for l, k in zip(y, keep.tolist()) if k]

            # plot
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_axes([0, 0, 1, 1])
            cc = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'tab:blue']
            for i, l in enumerate(sorted(set(y))):
                idx = [yl == l for yl in y]
                plt.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1], c=cc[i], s=10, edgecolor='none', alpha=0.5)
            ax.axis('off')  # adding it will get no axis
            plt.savefig(os.path.join(save_folder, 'tSNE_' + '{0}'.format(epoch) + '.png'))
            plt.close(fig)
        except:
            logging.info("Exception in t-SNE")
            print("Exception in t-SNE")
            pass

        VAE.train()

    def generate(val_loader, num_iters, generate_all=False, training_epoch=-1):
        VAE.eval()

        n_samples = 0
        bleu4_sum = 0.0
        rouge_scores_values_sum = [0.0] * 9
        eval_score = -1

        args.nsamples = 1
        args.batch_size = 1
        args.temperature = 0.95
        args.top_k = 100
        args.top_p = 0.95
        model_type = args.model_type

        # write samples to file
        samples_file = open(os.path.join(save_folder, 'generate-' + str(training_epoch) + '.txt'), 'w', encoding='utf8')

        questions = []
        generated_text = []
        actual_text = []
        with tqdm(total=len(val_loader)) as pbar:
            for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, topic) in enumerate(
                    val_loader):

                if i_test >= 10 and not generate_all: break

                length = -1
                if length == -1:
                    length = VAE.config.n_ctx - x_tokens.size(1) - 1
                elif length > VAE.config.n_ctx - x_tokens.size(1) - 1:
                    raise ValueError("Can't get samples longer than window size: %s" % VAE.config.n_ctx)

                eff_samples = []
                n, l = target_tokens.size()
                storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                storys = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in storys]
                storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                              storys]

                for _ in range(args.nsamples // args.batch_size):
                    out, _ = sample_sequence(
                        model=VAE,
                        tokenizer=tokenizer,
                        length=length,
                        batch_size=args.batch_size,
                        x_mask=x_mask,
                        x_tokens=x_tokens,
                        y_mask=y_mask,
                        y_tokens=y_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=device,
                        eos_token=tokenizer.encoder['<|endoftext|>'],
                        model_type=model_type
                    )
                    out = out.tolist()

                    # extract story, check metrics
                    for i in range(len(out)):
                        text = out[i]
                        text = text[text.index(endoftext) + 1:]

                        if endoftext in text:
                            idx = text.index(endoftext)
                            text = text[:idx]

                        text = tokenizer.decode(text).strip()

                        try:
                            # check bleu
                            bleu4 = sentence_bleu([storys_str[i].split()], text,
                                                  smoothing_function=SmoothingFunction().method7)

                            # check rouge
                            rouge = Rouge()
                            rouge_scores = rouge.get_scores(text, storys_str[i])
                            rouge_scores_values = [v for k in rouge_scores[0].keys() for v in
                                                   rouge_scores[0][k].values()]

                            bleu4_sum += bleu4
                            rouge_scores_values_sum = [v1 + v2 for v1, v2 in
                                                       zip(rouge_scores_values_sum, rouge_scores_values)]
                            n_samples += 1
                        except:
                            bleu4 = 0.0
                            rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                             'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                             'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

                        eff_samples.append((text, bleu4, rouge_scores))

                    pbar.update(1)

                for i in range(len(eff_samples)):
                    samples_file.write("=" * 50 + " SAMPLE " + str(i_test) + " " + "=" * 50)
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Question  " + "=" * 40)
                    samples_file.write('\n' * 2)
                    question = tokenizer.decode(x_tokens[i, :][x_mask[i, :] == 1].tolist())
                    samples_file.write(question)
                    questions.append(question)

                    samples_file.write('\n' * 2)
                    samples_file.write("=" * 40 + " Actual Explanation " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(storys_str[i])
                    actual_text.append(storys_str[i])
                    samples_file.write('\n' * 2)

                    samples_file.write("=" * 40 + " Generated Explanation " + "=" * 40)
                    samples_file.write('\n' * 2)
                    samples_file.write(eff_samples[i][0])
                    generated_text.append(eff_samples[i][0])
                    samples_file.write('\n' * 4)
                    samples_file.flush()

        if generate_all:
            predictions_and_actuals_df = pd.DataFrame({
                "Questions": questions,
                "Generated Text": generated_text,
                "Actual Text": actual_text
            })

            _, _, reference_text, _, _, _, generated_text_with_no_exact_repetitions, _, _, _ = preprocess_predictions_df(
                df=predictions_and_actuals_df)
            _, eval_score, _, _ = evaluate(metric_key="bleurt",
                                           generated=generated_text_with_no_exact_repetitions,
                                           references=reference_text,
                                           questions=questions,
                                           best_and_worst=False)
            v_writer.add_scalar("Validation - bleurt score", eval_score, training_epoch)

            predictions_and_actuals_df.to_csv(os.path.join("actual_vs_generated_{0}.csv".format(training_epoch)))

        print('Test complete with %05d samples.' % n_samples)
        logging.info("Test complete with %05d samples.", n_samples)
        logging.info("Iteration completed: %d" % num_iters)

        bleu4 = round(bleu4_sum / n_samples, 3)
        rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
        print(' bleu-4:', bleu4)
        print(' rouge :', rouge_scores_values)
        logging.info(' bleu-4: %f', bleu4)
        logging.info(' rouge : %s', str(rouge_scores_values))

        VAE.train()
        return eval_score

    test_plot(val_loader, e)
    val_step(val_loader)
    generate(val_loader, num_iters)
    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{0}'.format(e) + '.pt'))
    gpt2_model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)

    best_bleurt_score = -1

    no_cycles = 4
    total_no_beta_iters = args.iterations - args.iters_no_cyclic_annealing
    no_iter_in_cycle = math.ceil(total_no_beta_iters / no_cycles)

    while num_iters < args.iterations:
        # Run epoch
        st = time.time()

        # Training
        print('Training loop. Batches:', len(train_loader))
        logging.info('\n----------------------------------------------------------------------')
        logging.info("Training loop.       Batches: %d" % len(train_loader))

        with tqdm(total=len(train_loader)) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, topic) in enumerate(
                    train_loader):
                if num_iters < args.iters_no_cyclic_annealing:
                    beta = 1
                else:
                    # new cycle beta is zero
                    if (num_iters - args.iters_no_cyclic_annealing) % no_iter_in_cycle == 0:
                        logging.info('KL annealing restart')
                        beta = 0
                    else:
                        # anneal beta from 0 to 1 for first half of the cycle then fix it at 1
                        tau = ((num_iters - args.iters_no_cyclic_annealing - 1) % math.ceil(total_no_beta_iters / no_cycles)) / (total_no_beta_iters / no_cycles)
                        # proportion used to increase beta within a cycle
                        r = 0.5
                        if tau <= r:
                            beta = min(1, beta + (1 / r) * (1 / (total_no_beta_iters / no_cycles)))
                        else:
                            beta = 1

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    for name, parameter in VAE.named_parameters():
                        parameter.requires_grad = True
                    tuning_all = True

                output = train_step(device, VAE, optimizer, x_mask, x_tokens, y_mask, y_tokens,
                                    input_tokens, target_tokens, mask, loss_fn, beta, args.model_type, args.with_apex)
                loss, ce_loss, kl_loss = output[-1]

                lr = scheduler.get_last_lr()[0]
                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)
                t_writer.add_scalar('ppl', math.exp(min(ce_loss, 10)), num_iters)
                t_writer.add_scalar('lr', lr, num_iters)
                t_writer.add_scalar('iter_time', time.time() - st, num_iters)
                t_writer.add_scalar('kl', kl_loss, num_iters)
                t_writer.add_scalar('beta', beta, num_iters)

                if args.model_type == 'ae_vae_fusion':
                    loss, ce_loss, kl_loss = output[0]
                    # Log to Tensorboard
                    t_writer.add_scalar('ae_loss', loss, num_iters)
                    t_writer.add_scalar('ae_kl', kl_loss, num_iters)

                st = time.time()
                end = num_iters >= args.iterations

                if args.warmup != -1:
                    scheduler.step()

                if end: break
                num_iters += 1
                pbar.update(1)

                # # todo: what's this?
                # if args.switch_time > 0 and num_iters == int(args.iterations * args.switch_time):
                #     print('Switch to long sequence training')
                #     logging.info("Switch to long sequence training")
                #     cur_b_schedule += 1
                #     train_loader, val_loader, test_loader = prepare_dataset(
                #         args.data_dir, args.dataset, tokenizer,
                #         batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
                #         batch_schedule[-1][0], batch_schedule[-1][1],
                #         batch_schedule[-1][0], batch_schedule[-1][1],
                #         make_test=True,
                #         num_workers=args.workers, data_type=args.data_type
                #     )
        if not end:
            test_plot(val_loader, e)
            val_step(val_loader)
            current_bleurt_score = generate(val_loader, num_iters, generate_all=True, training_epoch=e)
            if current_bleurt_score > best_bleurt_score:
                best_bleurt_score = current_bleurt_score
                logging.info("best bleurt score is: {0}, and is from epoch: {1}".format(best_bleurt_score, e))

            print('Saving model...')
            logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
            logging.info("Saving model...")
            logging.info('\n------------------------------------------------------')
            torch.save(VAE.state_dict(),
                       os.path.join(save_folder, 'model_' + '{0}'.format(e) + '.pt'))
            gpt2_model.save_pretrained(save_folder)
            tokenizer.save_pretrained(save_folder)
            logging.info("Training loop. The ith epoch completed: %d" % e)
            e += 1

    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
    gpt2_model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)
    print('Training complete.')
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
