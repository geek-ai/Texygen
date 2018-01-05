import numpy as np


class Reward():
    def __init__(self):
        return

    def get_reward(self, sess, input_x, rollout_num, discriminator):

        def real_len(batches):
            return [np.ceil(np.argmin(batch + [0]) * 1.0 / 4) for batch in batches]

        rewards = []
        seq_len = len(input_x[0])
        for i in range(rollout_num):
            feed = {
                discriminator.input_x: input_x,
                discriminator.dropout_keep_prob: 0.8,
                discriminator.batch_size: len(input_x),
                discriminator.pad: np.zeros(([len(input_x), 1, 32, 1]),),
                discriminator.real_len: real_len(input_x),
            }
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            rewards.append(ypred)

        rewards = np.mean(rewards, axis=0)
        rewards = np.divide(rewards, np.sum(rewards))
        rewards -= np.mean(rewards)
        rewards = np.zeros([seq_len, 1]) + rewards.T
        return np.transpose(rewards)

