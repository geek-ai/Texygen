from time import time

from models.Gan import Gan
from models.leakgan.LeakganDataLoader import DataLoader, DisDataloader
from models.leakgan.LeakganDiscriminator import Discriminator
from models.leakgan.LeakganGenerator import Generator
from models.leakgan.LeakganReward import Reward
from utils.metrics.Bleu import Bleu
from utils.metrics.EmbSim import EmbSim
from utils.metrics.Nll import Nll
from utils.oracle.OracleLstm import OracleLstm
from utils.utils import *


def pre_train_epoch_gen(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss, _, _ = trainable_model.pretrain_step(sess, batch, .8)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def generate_samples_gen(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True, train=0):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess, 1.0, train))

    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)

    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes


class Leakgan(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20
        self.emb_dim = 32
        self.hidden_dim = 32
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        flags.DEFINE_boolean('restore', False, 'Training or testing a model')
        flags.DEFINE_boolean('resD', False, 'Training or testing a D model')
        flags.DEFINE_integer('length', 20, 'The length of toy data')
        flags.DEFINE_string('model', "", 'Model NAME')
        self.sequence_length = FLAGS.length
        self.filter_size = [2, 3]
        self.num_filters = [100, 200]
        self.l2_reg_lambda = 0.2
        self.dropout_keep_prob = 0.75
        self.batch_size = 64
        self.generate_num = 256
        self.start_token = 0
        self.dis_embedding_dim = 64
        self.goal_size = 16

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'

    def init_oracle_trainng(self, oracle=None):
        goal_out_size = sum(self.num_filters)

        if oracle is None:
            oracle = OracleLstm(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                                hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                                start_token=self.start_token)
        self.set_oracle(oracle)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters,
                                      batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                      start_token=self.start_token,
                                      goal_out_size=goal_out_size, step_size=4,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                              emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              filter_sizes=self.filter_size, start_token=self.start_token,
                              num_filters=self.num_filters, goal_out_size=goal_out_size, D_model=discriminator,
                              step_size=4)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=config)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)

    def init_metric(self):
        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

    def train_discriminator(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.dis_data_loader.load_train_data(self.oracle_file, self.generator_file)
        for _ in range(3):
            self.dis_data_loader.next_batch()
            x_batch, y_batch = self.dis_data_loader.next_batch()
            feed = {
                self.discriminator.D_input_x: x_batch,
                self.discriminator.D_input_y: y_batch,
            }
            _, _ = self.sess.run([self.discriminator.D_loss, self.discriminator.D_train_op], feed)
            self.generator.update_feature_function(self.discriminator)

    def evaluate(self):
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        if self.oracle_data_loader is not None:
            self.oracle_data_loader.create_batches(self.generator_file)
        if self.log is not None:
            if self.epoch == 0 or self.epoch == 1:
                for metric in self.metrics:
                    self.log.write(metric.get_name() + ',')
                self.log.write('\n')
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

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-leakgan.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        for a in range(1):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1})

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        print('adversarial training:')

        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        for epoch in range(self.adversarial_epoch_num//10):
            for epoch_ in range(10):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                for index in range(1):
                    start = time()
                    samples = self.generator.generate(self.sess, 1)
                    rewards = self.reward.get_reward(samples)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1
                    }
                    _, _, g_loss, w_loss = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, ], feed_dict=feed)
                    print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
                    end = time()
                    print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                    if self.epoch % 5 == 0 or self.epoch == self.adversarial_epoch_num - 1:
                        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                                             self.generator_file)
                        self.evaluate()
                    self.add_epoch()


                for _ in range(15):
                    self.train_discriminator()
            for epoch_ in range(5):
                start = time()
                loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
                end = time()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                self.add_epoch()
                if epoch % 5 == 0:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                                         self.generator_file)
                    # self.evaluate()
            for epoch_ in range(5):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator()

    def init_cfg_training(self, grammar=None):
        from utils.oracle.OracleCfg import OracleCfg
        oracle = OracleCfg(sequence_length=self.sequence_length, cfg_grammar=grammar)
        self.set_oracle(oracle)
        self.oracle.generate_oracle()
        self.vocab_size = self.oracle.vocab_size + 1
        goal_out_size = sum(self.num_filters)

        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters,
                                      batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                      start_token=self.start_token,
                                      goal_out_size=goal_out_size, step_size=4,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                              emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              filter_sizes=self.filter_size, start_token=self.start_token,
                              num_filters=self.num_filters, goal_out_size=goal_out_size, D_model=discriminator,
                              step_size=4)
        self.set_generator(generator)

        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        return oracle.wi_dict, oracle.iw_dict

    def init_cfg_metric(self, grammar=None):
        from utils.metrics.Cfg import Cfg
        cfg = Cfg(test_file=self.test_file, cfg_grammar=grammar)
        self.add_metric(cfg)

    def train_cfg(self):
        import json
        from utils.text_process import get_tokenlized
        from utils.text_process import code_to_text
        cfg_grammar = """
          S -> S PLUS x | S SUB x |  S PROD x | S DIV x | x | '(' S ')'
          PLUS -> '+'
          SUB -> '-'
          PROD -> '*'
          DIV -> '/'
          x -> 'x' | 'y'
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

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-leakganbasic-cfg.csv', 'w')
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_cfg_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num * 3):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        self.reset_epoch()
        print('adversarial training:')
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        for epoch in range(self.adversarial_epoch_num//10):
            for epoch_ in range(10):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                start = time()
                for index in range(1):
                    samples = self.generator.generate(self.sess, 1)
                    rewards = self.reward.get_reward(samples)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1
                    }
                    _, _, g_loss, w_loss = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, ], feed_dict=feed)
                    print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
                end = time()
                self.add_epoch()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                    get_cfg_test_file()
                    self.evaluate()

                for _ in range(15):
                    self.train_discriminator()
            for epoch_ in range(5):
                start = time()
                loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
                end = time()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                self.add_epoch()
                if epoch % 5 == 0:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                                         self.generator_file)
                    get_cfg_test_file()
                    self.evaluate()
            for epoch_ in range(5):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator()

    def init_real_trainng(self, data_loc=None):
        from utils.text_process import text_precess, text_to_code
        from utils.text_process import get_tokenlized, get_word_list, get_dict
        if data_loc is None:
            data_loc = 'data/image_coco.txt'
        self.sequence_length, self.vocab_size = text_precess(data_loc)

        goal_out_size = sum(self.num_filters)
        discriminator = Discriminator(sequence_length=self.sequence_length, num_classes=2, vocab_size=self.vocab_size,
                                      dis_emb_dim=self.dis_embedding_dim, filter_sizes=self.filter_size,
                                      num_filters=self.num_filters,
                                      batch_size=self.batch_size, hidden_dim=self.hidden_dim,
                                      start_token=self.start_token,
                                      goal_out_size=goal_out_size, step_size=4,
                                      l2_reg_lambda=self.l2_reg_lambda)
        self.set_discriminator(discriminator)

        generator = Generator(num_classes=2, num_vocabulary=self.vocab_size, batch_size=self.batch_size,
                              emb_dim=self.emb_dim, dis_emb_dim=self.dis_embedding_dim, goal_size=self.goal_size,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              filter_sizes=self.filter_size, start_token=self.start_token,
                              num_filters=self.num_filters, goal_out_size=goal_out_size, D_model=discriminator,
                              step_size=4)
        self.set_generator(generator)
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        oracle_dataloader = None
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)

        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader, oracle_loader=oracle_dataloader)
        tokens = get_tokenlized(data_loc)
        word_set = get_word_list(tokens)
        [word_index_dict, index_word_dict] = get_dict(word_set)
        with open(self.oracle_file, 'w') as outfile:
            outfile.write(text_to_code(tokens, word_index_dict, self.sequence_length))
        return word_index_dict, index_word_dict

    def init_real_metric(self):
        from utils.metrics.DocEmbSim import DocEmbSim
        docsim = DocEmbSim(oracle_file=self.oracle_file, generator_file=self.generator_file, num_vocabulary=self.vocab_size)
        self.add_metric(docsim)

        inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        inll.set_name('nll-test')
        self.add_metric(inll)

    def train_real(self, data_loc=None):
        from utils.text_process import code_to_text
        from utils.text_process import get_tokenlized
        wi_dict, iw_dict = self.init_real_trainng(data_loc)
        self.init_real_metric()

        def get_real_test_file(dict=iw_dict):
            with open(self.generator_file, 'r') as file:
                codes = get_tokenlized(self.generator_file)
            with open(self.test_file, 'w') as outfile:
                outfile.write(code_to_text(codes=codes, dictionary=dict))

        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 80
        self.adversarial_epoch_num = 100
        self.log = open('experiment-log-leakgan-real.csv', 'w')
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)

        for a in range(1):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1})

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                get_real_test_file()
                self.evaluate()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()


        self.reset_epoch()
        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        for epoch in range(self.adversarial_epoch_num//10):
            for epoch_ in range(10):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                start = time()
                for index in range(1):
                    samples = self.generator.generate(self.sess, 1)
                    rewards = self.reward.get_reward(samples)
                    feed = {
                        self.generator.x: samples,
                        self.generator.reward: rewards,
                        self.generator.drop_out: 1
                    }
                    _, _, g_loss, w_loss = self.sess.run(
                        [self.generator.manager_updates, self.generator.worker_updates, self.generator.goal_loss,
                         self.generator.worker_loss, ], feed_dict=feed)
                    print('epoch', str(epoch), 'g_loss', g_loss, 'w_loss', w_loss)
                end = time()
                self.add_epoch()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                    get_real_test_file()
                    self.evaluate()

                for _ in range(15):
                    self.train_discriminator()
            for epoch_ in range(5):
                start = time()
                loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
                end = time()
                print('epoch:' + str(epoch) + '--' + str(epoch_) + '\t time:' + str(end - start))
                if epoch % 5 == 0:
                    generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num,
                                         self.generator_file)
                    get_real_test_file()
                    # self.evaluate()
            for epoch_ in range(5):
                print('epoch:' + str(epoch) + '--' + str(epoch_))
                self.train_discriminator()

