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
        _, g_loss,_,_ = trainable_model.pretrain_step(sess, batch, .8)
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
        self.generate_num = 128
        self.start_token = 0
        self.dis_embedding_dim = 64
        self.goal_size = 16

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'

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

        bleu = Bleu(test_text=self.generator_file, real_text=self.oracle_file)
        self.add_metric(bleu)

        self.generator.set_similarity()
        self.oracle.set_similarity()
        embsim = EmbSim(model=self)
        self.add_metric(embsim)

        nll = Nll(data_loader=self.oracle_data_loader, rnn=self.oracle, sess=self.sess)
        self.add_metric(nll)

        # inll = Nll(data_loader=self.gen_data_loader, rnn=self.generator, sess=self.sess)
        # inll.set_name('i-nll')
        # self.add_metric(inll)

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
        self.oracle_data_loader.create_batches(self.generator_file)
        return super().evaluate()

    def train_oracle(self):
        self.init_metric()
        self.sess.run(tf.global_variables_initializer())

        self.pre_epoch_num = 10
        self.adversarial_epoch_num = 80
        log = open('experiment-log-leakgan.csv', 'w')
        generate_samples(self.sess, self.oracle, self.batch_size, self.generate_num, self.oracle_file)
        generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
        self.gen_data_loader.create_batches(self.oracle_file)
        self.oracle_data_loader.create_batches(self.generator_file)

        for a in range(1):
            g = self.sess.run(self.generator.gen_x, feed_dict={self.generator.drop_out: 1, self.generator.train: 1})

        # rollout = Reward(generator, update_rate)
        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()
            loss = pre_train_epoch_gen(self.sess, self.generator, self.gen_data_loader)
            end = time()
            print('epoch:' + str(epoch) + '\t time:' + str(start - end))
            if epoch % 5 == 0:
                generate_samples_gen(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
                self.evaluate()
            self.add_epoch()

        print('start pre-train discriminator:')
        self.reset_epoch()
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()

        print('adversarial training:')

        self.reward = Reward(model=self.generator, dis=self.discriminator, sess=self.sess, rollout_num=4)
        for epoch in range(self.adversarial_epoch_num):
            print('epoch:' + str(epoch))
            for index in range(100):
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

            if epoch % 5 == 0 or epoch == self.adversarial_epoch_num - 1:
                self.evaluate()

            for _ in range(15):
                self.train_discriminator()


if __name__ == '__main__':
    gan = Leakgan()
    gan.train_oracle()
