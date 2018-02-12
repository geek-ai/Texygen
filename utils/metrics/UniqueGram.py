import os
from multiprocessing import Pool

import nltk
from utils.metrics.Metrics import Metrics
from nltk import ngrams

class UniqueGram(Metrics):
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'UniqueGram'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 500
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        return self.get_ng()

    def get_ng(self):
        document = self.get_reference()
        length = len(document)
        grams = list()
        for sentence in document:
            grams += self.get_gram(sentence)
        return len(set(grams))/length

    def get_gram(self, tokens):
        grams = list()
        if len(tokens) < self.gram:
            return grams
        gram_generator = ngrams(tokens, self.gram)
        for gram in gram_generator:
            grams.append(gram)
        return grams

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as test_text:
                for text in test_text:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    # def get_ng(self):
    #     ngram = self.gram
    #     ng = list()
    #     reference = self.get_reference()
    #     weight = tuple((1. / ngram for _ in range(ngram)))
    #     with open(self.test_data) as test_data:
    #         for hypothesis in test_data:
    #             hypothesis = nltk.word_tokenize(hypothesis)
    #             ng.append(nltk.translate.ng_score.sentence_ng(reference, hypothesis, weight,
    #                                                                 smoothing_function=SmoothingFunction().method1))
    #     return sum(ng) / len(ng)




    def calc_ng(self, reference, hypothesis, weight):
        if len(hypothesis) < self.gram:
            return 0


    def get_ng_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_ng_parallel(reference=reference)

    def get_ng_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                hypothesis = nltk.word_tokenize(hypothesis)
                result.append(pool.apply_async(self.calc_ng, args=(reference, hypothesis, weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt
