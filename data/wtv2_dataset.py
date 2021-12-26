import torch.utils.data


class WordTreev2Dataset(torch.utils.data.Dataset):
    """
    A dataset for Word Tree v2
    """

    def __init__(self, texts, preprocess=lambda x: x, sort=False):
        super().__init__()
        self.texts = texts
        self.preprocess = preprocess
        self.sort=sort

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        if self.sort:
            return self.data[i]
        else:
            question_answer_feature, explanation = self.texts[i]

            question_answer_feature = question_answer_feature.strip()
            explanation = explanation.strip()
            # todo: how to change it back
            #text_raw_dict = {'question_answer_feature': question_answer_feature, 'explanation': explanation}
            text_raw_dict = {'title': question_answer_feature, 'story': explanation}

            text = self.preprocess(text_raw_dict)
            return text
