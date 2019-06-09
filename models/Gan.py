from abc import abstractmethod

from models.leakgan.LeakganDataLoader import DataLoader, DisDataloader
from utils.utils import *


class Gan:
    def __init__(self):
        self.oracle = None
        self.generator = None
        self.discriminator = None
        self.gen_data_loader = None
        self.dis_data_loader = None
        self.nll_data_loader = None
        self.oracle_data_loader = None
        self.sess = init_sess()
        self.metrics = list()
        self.epoch = 0
        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = None
        self.reward = None
        self.batch_size = 64
        self.sequence_length = None
        self.nll_test_loc = None
        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.oracle_test_file = 'save/oracle_test.txt'

    def set_oracle(self, oracle):
        self.oracle = oracle

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_data_loader(self):
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        nll_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.gen_data_loader = gen_dataloader
        self.dis_data_loader = dis_dataloader
        self.oracle_data_loader = oracle_dataloader
        self.nll_data_loader = nll_dataloader

    def set_sess(self, sess):
        self.sess = sess

    def add_metric(self, metric):
        self.metrics.append(metric)

    def add_epoch(self):
        self.epoch += 1

    def reset_epoch(self):
        # current not in use
        return
        self.epoch = 0

    def evaluate(self):
        from time import time
        log = "epoch:" + str(self.epoch) + '\t'
        scores = list()
        scores.append(self.epoch)
        for metric in self.metrics:
            tic = time()
            score = metric.get_score()
            log += metric.get_name() + ":" + str(score) + '\t'
            toc = time()
            print('time elapsed of ' + metric.get_name() + ': ' + str(toc - tic))
            scores.append(score)
        print(log)
        return scores

    def check_valid(self):
        # TODO
        pass

    @abstractmethod
    def train_oracle(self):
        pass

    def train_cfg(self):
        pass

    def train_real(self):
        pass
