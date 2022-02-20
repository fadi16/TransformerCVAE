import nltk

from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
class Utils:
    def init_explanation_bank_lemmatizer(self):
        lemmatization_file = open("./data/retrieve_prompt_generate/lemmatization-en.txt")
        self.lemmas = {}
        # saving lemmas
        for line in lemmatization_file:
            self.lemmas[line.split("\t")[1].lower().replace("\n", "")] = line.split("\t")[0].lower()
        return self.lemmas

    def explanation_bank_lemmatize(self, string: str):
        if self.lemmas == None:
            self.init_explanation_bank_lemmatizer()
        temp = []
        for word in string.split(" "):
            if word.lower() in self.lemmas:
                temp.append(self.lemmas[word.lower()])
            else:
                temp.append(word.lower())
        return " ".join(temp)

    def clean_fact(self, fact_explanation):
        fact = []
        for key in fact_explanation:
            if not "[SKIP]" in key and fact_explanation[key] != None:
                fact.append(str(fact_explanation[key]))
        return " ".join(fact)

    def preprocess_fact(self, fact):
            fact_tokens = nltk.word_tokenize(fact.replace("?", " ")
                                             .replace(".", " ")
                                             .replace(",", " ")
                                             .replace(";", " ")
                                             .replace("-", " "))
            lemmatized_fact_tokens = [self.explanation_bank_lemmatize(token.lower()) for token in fact_tokens]
            if len(lemmatized_fact_tokens) > 1:
                return " ".join(lemmatized_fact_tokens)

    def preprocess_question(self, question, remove_stopwords=False):
        question_tokens = nltk.word_tokenize(question)
        lemmatized_question_tokens = [self.explanation_bank_lemmatize(token.lower()) for token in question_tokens
                                      if not token.lower() in stopwords.words("english") or not remove_stopwords]
        return " ".join(lemmatized_question_tokens)

