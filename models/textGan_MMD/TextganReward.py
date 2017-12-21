import numpy as np
import tensorflow as tf


class Reward(object):
    def __init__(self):
        self.f_r = None
        self.f_r_size = None
        self.mu_r = None
        self.sigma_r = None

        self.sample_size = 100

    def set_vec_real(self, sess, input_x, discriminator):
        feed = {discriminator.input_x: input_x}
        self.f_r = sess.run(discriminator.h_highway, feed)
        self.f_r_size = len(self.f_r)

    def get_reward(self, sess, input_x, discriminator):
        feed = {discriminator.input_x: input_x}
        f_s = sess.run(discriminator.h_highway, feed)
        mmd_param = {'logistic': 0, 'logistic_s': 5,  # Weight in multi-kernel MMD
                     'gaussian': 1, 'gaussian_sigma': 20}
        l_mmd = []
        f_r = self.f_r[np.random.choice(self.f_r.shape[0], self.sample_size, replace=False), :]

        for x in f_s:
            reward = 0
            for y in f_r:
                reward += calculate_mmd(x, y, mmd_param, 1)
                pass
            reward /= self.sample_size
            l_mmd.append(reward)
        l_mmd = np.array(l_mmd)
        l_mmd = np.zeros([len(input_x[0]), 1]) + l_mmd.T
        return np.subtract(0.0, l_mmd).T


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


# def calculate_gaussian_mmd(x0, y0, x1, y1, param, batch_size):
#     kxx = tf.reduce_sum(tf.exp(-tf.square(x0 - x1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
#     kxy = tf.reduce_sum(tf.exp(-tf.square(x0 - y1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
#     kyy = tf.reduce_sum(tf.exp(-tf.square(y0 - y1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
#     return kxx - 2 * kxy + kyy


def calculate_gaussian_mmd(x0, y0, x1, y1, param, batch_size):
    kxx = np.sum(np.exp(-np.square(x0 - x1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
    kxy = np.sum(np.exp(-np.square(x0 - y1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
    kyy = np.sum(np.exp(-np.square(y0 - y1) / 2 / param['gaussian_sigma'])) / batch_size ** 2
    return kxx - 2 * kxy + kyy

def calculate_mmd(x, y, param, batch_size):
    xt = np.transpose(x)
    yt = np.transpose(y)
    # x0 = np.identity(x)
    # y0 = np.identity(y)
    # x1 = np.identity(xt)
    # y1 = np.identity(yt)
    x0 = x
    y0 = y

    x1 = xt
    y1 = yt
    for i in range(batch_size - 1):
        x0 = np.concatenate([x0, x], axis=1)
        y0 = np.concatenate([y0, y], axis=1)
        x1 = np.concatenate([x1, xt], axis=0)
        y1 = np.concatenate([y1, yt], axis=0)
    gaussian_mmd = calculate_gaussian_mmd(x0, y0, x1, y1, param, batch_size)
    # logistic_mmd = calculate_logistic_mmd(x0, y0, x1, y1, param, batch_size)
    # mmd = param['logistic'] * logistic_mmd + param['gaussian'] * gaussian_mmd
    return gaussian_mmd
