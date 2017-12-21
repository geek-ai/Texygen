import numpy as np


class Reward(object):
    def __init__(self):
        return

    def get_reward(self, sess, input_x, rollout_num, discriminator):
        rewards = []
        seq_len = len(input_x[0])
        for i in range(rollout_num):
            feed = {discriminator.input_x: input_x}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            rewards.append(ypred)

        rewards = np.mean(rewards, axis=0)
        rewards = np.divide(rewards, np.sum(rewards))
        rewards -= np.mean(rewards)
        rewards = np.zeros([seq_len, 1]) + rewards.T
        return np.transpose(rewards)
