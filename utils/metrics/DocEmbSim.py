import collections
import math
import random

import nltk
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine

from utils.metrics.Metrics import Metrics


class DocEmbSim(Metrics):
    def __init__(self, oracle_file=None, generator_file=None, num_vocabulary = None):
        super().__init__()
        self.name = 'EmbeddingSimilarity'
        self.oracle_sim = None
        self.gen_sim = None
        self.is_first = True
        self.oracle_file = oracle_file
        self.generator_file = generator_file
        self.num_vocabulary = num_vocabulary
        self.batch_size = 64
        self.embedding_size = 32
        self.data_index = 0
        self.valid_examples = None

    def get_score(self):
        if self.is_first:
            self.get_oracle_sim()
            self.is_first = False
        self.get_gen_sim()
        return self.get_dis_corr()

    def get_frequent_word(self):
        if self.valid_examples is not None:
            return self.valid_examples

        import collections
        words = []
        with open(self.oracle_file, 'r') as file:
            for line in file:
                text = nltk.word_tokenize(line)
                text = list(map(int, text))
                words += text
        counts = collections.Counter(words)
        new_list = sorted(words, key=lambda x: -counts[x])
        word_set = list(set(new_list))
        if len(word_set) < self.num_vocabulary//10:
            self.valid_examples = word_set
            return word_set
        else:
            self.valid_examples = word_set[0:self.num_vocabulary//10]
            return word_set[0:self.num_vocabulary//10]

    def read_data(self, file):
        words = []
        with open(file, 'r') as file:
            for line in file:
                text = nltk.word_tokenize(line)
                words.append(text)
        return words

    def generate_batch(self, batch_size, num_skips, skip_window, data=None):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        return batch, labels

    def get_wordvec(self, file):
        graph = tf.Graph()
        batch_size = self.batch_size
        embedding_size = self.embedding_size
        vocabulary_size = self.num_vocabulary
        num_sampled = 64
        if num_sampled > vocabulary_size:
            num_sampled = vocabulary_size
        num_steps = 2
        skip_window = 1  # How many words to consider left and right.
        num_skips = 2  # How many times to reuse an input to generate a label.
        if self.valid_examples is None:
            self.get_frequent_word()

        with graph.as_default():

            # Input data.
            train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # initial Variables.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, seed=11))
            softmax_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size), seed=12))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, train_dataset)
            # Compute the softmax loss, using a sample of the negative labels each time.
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

            data = self.read_data(file)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            average_loss = 0
            generate_num = len(data)
            for step in range(num_steps):
                for index in range(generate_num):
                    cur_batch_data, cur_batch_labels = self.generate_batch(
                        batch_size, num_skips, skip_window, data[index])
                    feed_dict = {train_dataset: cur_batch_data, train_labels: cur_batch_labels}
                    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += l
            similarity_value = similarity.eval()
            return similarity_value

    def get_oracle_sim(self):
        self.oracle_sim = self.get_wordvec(self.oracle_file)

    def get_gen_sim(self):
        self.gen_sim = self.get_wordvec(self.generator_file)

    def get_dis_corr(self):
        if len(self.oracle_sim) != len(self.gen_sim):
            raise ArithmeticError
        corr = 0
        for index in range(len(self.oracle_sim)):
            corr += (1 - cosine(np.array(self.oracle_sim[index]), np.array(self.gen_sim[index])))
        return np.log10(corr / len(self.oracle_sim))
