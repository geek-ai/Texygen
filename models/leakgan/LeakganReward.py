import numpy as np


def redistribution(idx, total, min_v):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward, rollout_num=1.0):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l), min_s)
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
    return ret


class Reward(object):
    def __init__(self, model, dis, sess, rollout_num):
        self.model = model
        self.dis = dis
        self.sess = sess
        self.rollout_num = rollout_num

    def get_reward(self, input_x):
        rewards = []
        for i in range(self.rollout_num):
            for given_num in range(1, self.model.sequence_length // self.model.step_size):
                real_given_num = given_num * self.model.step_size
                feed = {self.model.x: input_x, self.model.given_num: real_given_num, self.model.drop_out: 1.0}
                samples = self.sess.run(self.model.gen_for_reward, feed)
                # print samples.shape
                feed = {self.dis.D_input_x: samples}
                ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            feed = {self.dis.D_input_x: input_x}
            ypred_for_auc = self.sess.run(self.dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self.model.sequence_length // self.model.step_size - 1] += ypred
        rewards = rescale(np.array(rewards), self.rollout_num)
        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length
        return rewards
