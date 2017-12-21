import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Reward(object):
    def __init__(self):
        self.mu_r = None
        self.sigma_r = None

    def set_vec_real(self, sess, input_x, discriminator):
        feed = {discriminator.input_x: input_x}
        f_r = sess.run(discriminator.h_highway, feed)
        self.mu_r = np.mean(f_r, axis=0)
        self.sigma_r = np.cov(np.transpose(f_r))

    def get_reward(self, sess, input_x, discriminator):
        feed = {discriminator.input_x: input_x}
        f_s = sess.run(discriminator.h_highway, feed)
        mu_s = np.mean(f_s, axis=0)
        sigma_s = np.cov(np.transpose(f_s))
        if np.linalg.matrix_rank(sigma_s) == sigma_s.shape[0]:
            print('non-singular!')

        # add noise when sigma_s singular
        while np.linalg.matrix_rank(sigma_s) != sigma_s.shape[0]:
            print('singular!')
            print(np.linalg.matrix_rank(sigma_s))
            print(np.linalg.matrix_rank(f_s))
            sig = (sigma_s.max()- sigma_s.min()) / 100
            noise = np.random.normal(scale=sig, size=sigma_s.shape)
            sigma_s += noise

        term_1 = np.trace(
            (np.matmul(np.linalg.inv(sigma_s), self.sigma_r)) + (np.matmul(np.linalg.inv(self.sigma_r), sigma_s)))
        term_2 = np.subtract(mu_s, self.mu_r)
        term_3 = np.add(np.linalg.inv(sigma_s), np.linalg.inv(self.sigma_r))
        loss = term_1 + np.matmul(np.matmul(np.transpose(term_2), term_3), term_2)
        neg_loss = np.zeros(shape=input_x.shape)
        return np.subtract(neg_loss,loss)


def logistic_kernel(x, y, param):
    # useful for calculate_logistic_mmd, same symbol as https://en.wikipedia.org/wiki/Logistic_distribution
    s = param['logistic_s']
    numerator = tf.exp(-(x - y) / s)
    denominator = s * tf.square(1 + tf.exp(-(x - y) / s))
    return numerator / denominator


def calculate_logistic_mmd(x0, y0, x1, y1, param, batch_size):
    kxx = tf.reduce_sum(logistic_kernel(x0, x1, param)) / batch_size ** 2
    kxy = tf.reduce_sum(logistic_kernel(x0, y1, param)) / batch_size ** 2
    kyy = tf.reduce_sum(logistic_kernel(y0, y1, param)) / batch_size ** 2
    return kxx - 2 * kxy + kyy


def calculate_gaussian_mmd(x0, y0, x1, y1, param, batch_size):
    kxx = tf.reduce_sum(tf.exp(-tf.square(x0 - x1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
    kxy = tf.reduce_sum(tf.exp(-tf.square(x0 - y1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
    kyy = tf.reduce_sum(tf.exp(-tf.square(y0 - y1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
    return kxx - 2 * kxy + kyy


def calculate_mmd(x, y, param, batch_size):
    xt = tf.transpose(x)
    yt = tf.transpose(y)
    x0 = tf.identity(x)
    y0 = tf.identity(y)
    x1 = tf.identity(xt)
    y1 = tf.identity(yt)
    for i in range(batch_size - 1):
        x0 = tf.concat([x0, x], axis=1)
        y0 = tf.concat([y0, y], axis=1)
        x1 = tf.concat([x1, xt], axis=0)
        y1 = tf.concat([y1, yt], axis=0)
    gaussian_mmd = calculate_gaussian_mmd(x0, y0, x1, y1, param, batch_size)
    logistic_mmd = calculate_logistic_mmd(x0, y0, x1, y1, param, batch_size)
    mmd = param['logistic'] * logistic_mmd + param['gaussian'] * gaussian_mmd
    return mmd
