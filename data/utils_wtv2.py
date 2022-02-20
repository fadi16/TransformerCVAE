import random, re, os
from data.prompt_dataset import *
from data.plot_dataset import *
from data.arxiv_dataset import *
from data.yelp_dataset import *
from data.wtv2_dataset import *
import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from unidecode import unidecode
import functools
from rake_nltk import Rake
import urllib, sys
import urllib.request
import json, re
import numpy as np
from scipy.spatial.distance import cdist
from bert_serving.client import BertClient
from tqdm import trange
from random import shuffle
import pandas as pd
from data.retrieve_prompt_generate import retrieve


def compose(*functions):
    """ Executes a list of functions in order """
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)


def prefix_truncate(window):
    """ truncates text to the prefix window size """

    def f(text):
        if len(text) > window:
            text = text[:window]
        return text

    return f


class Preprocessor_base():
    def __init__(self):
        self.fn = None

    def make_fn(self):
        raise NotImplementedError()

    def __call__(self, x):
        try:
            if self.fn is None:
                self.fn = self.make_fn()
            x = self.fn(x)
            return x
        except Exception as e:
            print('Error in preprocessing', repr(e))
            raise e


def encode_tuple(tokenizer, t):
    # x, x + y, x + y, topic
    return tokenizer.encode(t[0]), tokenizer.encode(t[1]), tokenizer.encode(t[2]), t[3]


def truncate_tuple(truncator, t):
    return truncator(t[0]), truncator(t[1]), truncator(t[2]), t[3]


class Preprocessor(Preprocessor_base):
    def __init__(self, tokenizer, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def make_fn(self):
        return compose(
            insert_keywords(self.tokenizer),
            lambda input: encode_tuple(self.tokenizer, input) if isinstance(input, tuple) else [
                encode_tuple(self.tokenizer, inp) for inp in input],
            lambda input: truncate_tuple(prefix_truncate(self.seq_len), input) if isinstance(input, tuple) else [
                truncate_tuple(prefix_truncate(self.seq_len), inp) for inp in input]
        )


def insert_keywords(tokenizer):
    def f(text_raw_dict):
        # 'prompt' in text_raw_dict --> wp dataset; 'title' in text_raw_dict --> wi dataset and other well preprocessed dataset
        qna = text_raw_dict['question_and_answer']
        explanation = text_raw_dict['explanation']
        topic = text_raw_dict['major_question_topic']
        qna_and_explanation = qna + tokenizer.eos_token + explanation + tokenizer.eos_token
        # x, x + y, x + y
        return qna + tokenizer.eos_token, qna_and_explanation, tokenizer.eos_token + qna_and_explanation, topic

    return f


def collate_fn(samples):
    """ Creates a batch out of samples """
    x_max_len = max(map(lambda s: len(s[0]), samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(ss[0]) + [0] * (x_max_len - len(ss[0])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257, endoftext 50256, use 50257 here causes errors!!
    x = torch.LongTensor([ss[0] + [50256] * (x_max_len - len(ss[0])) for ss in samples])

    max_len = max(map(lambda s: len(s[1]), samples))
    # Zero pad mask
    y_mask = torch.ByteTensor([[1] * len(ss[1]) + [0] * (max_len - len(ss[1])) for ss in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257
    y = torch.LongTensor([ss[1] + [50256] * (max_len - len(ss[1])) for ss in samples])

    max_len = max(map(lambda s: len(s[2]), samples))
    # Zero pad mask
    input_mask = torch.ByteTensor([[1] * len(ip[2]) + [0] * (max_len - len(ip[2])) for ip in samples])
    # tokenizer.convert_tokens_to_ids('<|startoftext|>') = 50257
    input = torch.LongTensor([ip[2] + [50256] * (max_len - len(ip[2])) for ip in samples])

    topic = [p[3] for p in samples]

    return x_mask, x, y_mask, y, input[:, :-1], input[:, 1:].contiguous(), input_mask[:, 1:], topic


def prepare_dataset(data_dir, tokenizer, train_bsz, train_seq_len, val_bsz, val_seq_len, test_bsz=1,
                    test_seq_len=1024, num_workers=1, make_train=True, make_val=True, make_test=False,
                    with_retrieval=True, no_hypotheses=20, no_facts=6, central_only=False):
    loaders = []

    EXPLANATION = "explanation"
    QUESTION_AND_ANSWER = "question_and_answer"
    MAJOR_TOPIC = "major_question_topic"

    train_collate_fn = collate_fn
    val_collate_fn = collate_fn
    test_collate_fn = collate_fn

    print('Loading Word Tree v2 dataset...')
    train_data_path = os.path.join(data_dir, 'wordTreev2/train_data_wed.csv')
    val_data_path = os.path.join(data_dir, 'wordTreev2/dev_data_wed.csv')
    test_data_path = os.path.join(data_dir, 'wordTreev2/test_data_wed.csv')

    # train
    df_train = pd.read_csv(train_data_path, delimiter="\t")
    train_explanations = df_train[EXPLANATION]
    train_questions_and_answers = df_train[QUESTION_AND_ANSWER]
    train_topics = df_train[MAJOR_TOPIC]

    # val
    df_val = pd.read_csv(val_data_path, delimiter="\t")
    val_explanations = df_val[EXPLANATION]
    val_questions_and_answers = df_val[QUESTION_AND_ANSWER]
    val_topics = df_val[MAJOR_TOPIC]

    # test
    df_test = pd.read_csv(test_data_path, delimiter="\t")
    test_explanations = df_test[EXPLANATION]
    test_questions_and_answers = df_test[QUESTION_AND_ANSWER]
    test_topics = df_test[MAJOR_TOPIC]

    if with_retrieval:
        print("USING RETRIEVAL METHOD")
        train_retrieved_facts, dev_retrieved_facts, test_retrieved_facts = retrieve.retrieve(
            training_df=df_train,
            testing_df=df_test,
            val_df=df_val,
            no_similar_hypotheses=no_hypotheses,
            no_retrieved_facts=no_facts,
            only_central=central_only)

        for i in range(len(train_retrieved_facts)):
            train_questions_and_answers[i] += " @@ " + train_retrieved_facts[i]
        for i in range(len(dev_retrieved_facts)):
            val_questions_and_answers[i] += " @@ " + dev_retrieved_facts[i]
        for i in range(len(test_retrieved_facts)):
            test_questions_and_answers[i] += " @@ " + test_retrieved_facts[i]

    train_text = [(q, e, t) for q, e, t in zip(train_questions_and_answers, train_explanations, train_topics) if
                  q.strip() != '' and e.strip() != '']
    val_text = [(q, e, t) for q, e, t in zip(val_questions_and_answers, val_explanations, val_topics) if
                  q.strip() != '' and e.strip() != '']
    test_text = [(q, e, t) for q, e, t in zip(test_questions_and_answers, test_explanations, test_topics) if
                  q.strip() != '' and e.strip() != '']

    print('Done.')

    if make_train:
        train_preproc = Preprocessor(tokenizer, train_seq_len)
        d_train = WordTreev2Dataset(train_text, train_preproc)
        print('Train dataset size', len(d_train))
        loaders.append(data.DataLoader(d_train,
                                       batch_size=train_bsz,
                                       pin_memory=True,
                                       drop_last=True,
                                       num_workers=num_workers,
                                       collate_fn=train_collate_fn) if d_train else None)
    if make_val:
        val_preproc = Preprocessor(tokenizer, val_seq_len)
        d_val = WordTreev2Dataset(val_text, val_preproc)
        print('Val dataset size', len(d_val))
        loaders.append(data.DataLoader(d_val,
                                       batch_size=val_bsz,
                                       pin_memory=True,
                                       drop_last=True,
                                       num_workers=num_workers,
                                       collate_fn=val_collate_fn) if d_val else None)

    if make_test:
        test_preproc = Preprocessor(tokenizer, test_seq_len)
        d_test = WordTreev2Dataset(test_text, test_preproc)
        print('Test dataset size', len(d_test))
        loaders.append(data.DataLoader(d_test,
                                       batch_size=test_bsz,
                                       pin_memory=True,
                                       drop_last=True,
                                       num_workers=num_workers,
                                       collate_fn=test_collate_fn) if d_test else None)
    return loaders
