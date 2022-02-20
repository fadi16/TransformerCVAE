import distutils.errors
import nltk

from bm25 import BM25


def fit_bm25_on_wtv2(utils, facts_bank, training_questions, facts_ids, training_questions_ids):
    model = BM25()

    # preprocessing facts bank
    lemmatized_facts_bank = []
    for fact in facts_bank:
        lemmatized_fact = utils.preprocess_fact(fact)
        if lemmatized_fact:
            lemmatized_facts_bank.append(lemmatized_fact)

    # preprocessing the q/a - explanations
    lemmatized_questions = []
    for question in training_questions:
        lemmatized_questions.append(utils.preprocess_question(question))

    model.fit(lemmatized_facts_bank, lemmatized_questions, facts_ids, training_questions_ids)
    return model
