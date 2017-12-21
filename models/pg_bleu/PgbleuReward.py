import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
import random


class Reward(object):
    def __init__(self, oracle_file):
        self.oracle_file = oracle_file
        self.oracle_text = self.get_oracle()
        self.oracle_size = len(self.oracle_text)
        self.sample_size = 20

    def get_oracle(self):
        oracle_text = list()
        with open(self.oracle_file) as real_data:
            for text in real_data:
                text = nltk.word_tokenize(text)
                oracle_text.append(list(map(int, text)))
        return np.array(oracle_text)

    def get_reward(self, samples):
        rewards = []
        ngram = 3
        weight = tuple((1. / ngram for _ in range(ngram)))
        reference = self.oracle_text[np.random.choice(self.oracle_text.shape[0], self.sample_size, replace=False), :]
        for sample in samples:
            rewards.append(nltk.translate.bleu_score.sentence_bleu(reference, sample, weight,
                                                                   smoothing_function=SmoothingFunction().method1))
        rewards = np.array(rewards)
        rewards = np.zeros([len(samples[0]), 1]) + rewards.T
        return np.transpose(rewards)
