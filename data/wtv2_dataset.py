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
            question_answer_feature, explanation, topic = self.texts[i]

            question_answer_feature = question_answer_feature.strip()
            explanation = explanation.strip()
            topic = topic.strip()
            text_raw_dict = {'question_and_answer': question_answer_feature, 'explanation': explanation, 'major_question_topic': topic}

            text = self.preprocess(text_raw_dict)
            return text
