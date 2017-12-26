import json
from time import time

from models.Gan import Gan
from models.gsgan.GsganDataLoader import DataLoader, DisDataloader
from models.gsgan.GsganDiscriminator import Discriminator
from models.gsgan.GsganGenerator import Generator
from models.gsgan.GsganReward import Reward
from utils.metrics.Bleu import Bleu
from utils.metrics.Cfg import Cfg
from utils.metrics.EmbSim import EmbSim
from utils.oracle.OracleCfg import OracleCfg
from utils.oracle.OracleLstm import OracleLstm
from utils.text_process import *
from utils.utils import *


class Gsgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        self.sequence_length = 20
        self.filter_size = [3, 4, 5]
        self.num_filters = 128
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 128
        self.start_token = 0

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'

    def init_oracle_trainng(self, oracle=None):
        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        # fixme bad taste
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      hidden_unit=20, embedding_size=self.emb_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters, non_static=True,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def init_metric(self):

        bleu = Bleu(test_text=self.generator_file, real_text=self.oracle_file)
        self.add_metric(bleu)

        self.generator.set_similarity()
        self.oracle.set_similarity()
        embsim = EmbSim(model=self)
        self.add_metric(embsim)

        # nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        # self.add_metric(nll)
        #
        # inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        # inll.set_name('i-nll')
        # self.add_metric(inll)

    def train_discriminator(self):

        def real_len(batches):
            return [np.ceil(np.argmin(batch + [0]) * 1.0 / 4) for batch in batches]

        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.input_x: x_batch,
                self.discriminator.input_y: y_batch,
                self.discriminator.dropout_keep_prob: 0.8,
                self.discriminator.batch_size: len(x_batch),
                self.discriminator.pad: np.zeros([len(x_batch), 1, 32, 1]),
                self.discriminator.real_len: real_len(x_batch),
            }
            # _ = self.sess.run(self.discriminator.train_op, feed)
            _, step, loss, accuracy = self.sess.run(
                [self.discriminator.train_op, self.discriminator.global_step, self.discriminator.loss,
                 self.discriminator.accuracy], feed)

    def evaluate(self):
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            scores = super().evaluate()
            for score in scores:
                self.log.write(str(score) + ',')
            self.log.write('\n')
            return scores
        return super().evaluate()

    def train_oracle(self):
        self.init_oracle_trainng()
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 10
        self.adversarial_epoch_num = 80
        self.log = open('experiment-log-seqgan.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        _ = self.sess.run(self.generator.start_token)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(epoch) + '\t time:' + str(start - end))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward()
        for epoch in range(self.adversarial_epoch_num):
            print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                z_h0 = np.random.uniform(low=0, high=1, size=[self.batch_size, self.emb_dim])
                z_c0 = np.random.uniform(low=0, high=1, size=[self.batch_size, self.emb_dim])

                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards,
                    self.generator.h_0: z_h0,
                    self.generator.c_0: z_c0,
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            print('epoch:' + str(epoch) + '\t time:' + str(start - end))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()

    def init_cfg_training(self, grammar=None):
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1

        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              start_token=self.start_token)
        self.set_generator(generator)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      hidden_unit=20, embedding_size=self.emb_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters, non_static=True,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)

    def train_cfg(self):
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x'
        """

        wi_dict_loc, iw_dict_loc = self.init_cfg_training(cfg_grammar)
        with open(iw_dict_loc, 'r') as file:
            iw_dict = json.load(file)

        def get_cfg_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.init_cfg_metric(grammar=cfg_grammar)
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 0
        self.adversarial_epoch_num = 200
        self.log = open('experiment-log-seqgan-cfg.csv', 'w')
        # generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(epoch) + '\t time:' + str(start - end))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward()
        for epoch in range(self.adversarial_epoch_num):
            print('epoch:' + str(epoch))
            start = time()
            for index in range(1):
                samples = self.generator.generate(self.sess)
                rewards = self.reward.get_reward(self.sess, samples, 16, self.discriminator)
                feed = {
                    self.generator.x: samples,
                    self.generator.rewards: rewards
                }
                _ = self.sess.run(self.generator.g_updates, feed_dict=feed)
            end = time()
            self.add_epoch()
            print('epoch:' + str(epoch) + '\t time:' + str(start - end))
            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()


if __name__ == '__main__':
    gan = Gsgan()
    gan.train_oracle()
    # gan.train_cfg()
