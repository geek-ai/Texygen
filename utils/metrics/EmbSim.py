import numpy as np
from scipy.spatial.distance import cosine

from utils.metrics.Metrics import Metrics


class EmbSim(Metrics):
    def __init__(self, model):
        super().__init__()
        self.name = 'EmbeddingSimilarity_LSTM'
        self.sess = model.sess
        self.oracle = model.oracle
        self.generator = model.generator
        self.oracle_sim = None
        self.gen_sim = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self):
        if self.is_first:
            self.get_oracle_sim()
            self.is_first = False
        self.get_gen_sim()
        return self.get_dis_corr()

    def get_oracle_sim(self):
        with self.sess.as_default():
            self.oracle_sim = self.oracle.similarity.eval()

    def get_gen_sim(self):
        with self.sess.as_default():
            self.gen_sim = self.generator.similarity.eval()

    def get_dis_corr(self):
        if len(self.oracle_sim) != len(self.gen_sim):
            raise ArithmeticError
        corr = 0
        for index in range(len(self.oracle_sim)):
            corr += (1 - cosine(np.array(self.oracle_sim[index]), np.array(self.gen_sim[index])))
        return np.log10(corr / len(self.oracle_sim))
