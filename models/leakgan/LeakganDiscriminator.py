import tensorflow as tf
import  numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

def cosine_similarity(a,b):
    normalize_a = tf.nn.l2_normalize(a, -1)
    normalize_b = tf.nn.l2_normalize(b, -1)
    cos_similarity = (tf.multiply(normalize_a, normalize_b))
    return cos_similarity
# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output

class Discriminator(object):
    def __init__(self, sequence_length, num_classes, vocab_size,dis_emb_dim,filter_sizes, num_filters,batch_size,hidden_dim,
                 start_token,goal_out_size,step_size,l2_reg_lambda=0.0, dropout_keep_prob=0.75):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.l2_reg_lambda = l2_reg_lambda
        self.num_filters_total = sum(self.num_filters)
        self.temperature = 1.0
        self.grad_clip = 5.0
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.dropout_keep_prob = dropout_keep_prob

        self.D_input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.D_input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")

        with tf.name_scope('D_update'):
            self.D_l2_loss = tf.constant(0.0)
            self.FeatureExtractor_unit = self.FeatureExtractor()

            # Train for Discriminator
            with tf.variable_scope("feature") as self.feature_scope:
                D_feature = self.FeatureExtractor_unit(self.D_input_x,self.dropout_keep_prob)#,self.dropout_keep_prob)
                self.feature_scope.reuse_variables()

            D_scores, D_predictions,self.ypred_for_auc = self.classification(D_feature)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=D_scores, labels=self.D_input_y)
            self.D_loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.D_l2_loss

            self.D_params = [param for param in tf.trainable_variables() if
                             'Discriminator' or 'FeatureExtractor' in param.name]
            d_optimizer = tf.train.AdamOptimizer(5e-5)
            D_grads_and_vars = d_optimizer.compute_gradients(self.D_loss, self.D_params, aggregation_method=2)
            self.D_train_op = d_optimizer.apply_gradients(D_grads_and_vars)


    # This module used to Extract sentence's Feature
    def FeatureExtractor(self):
        # Embedding layer
        # scope.reuse_variables()
        def unit(Feature_input,dropout_keep_prob):#,dropout_keep_prob):
            with tf.variable_scope('FeatureExtractor') as scope:
                with tf.device('/cpu:0'), tf.name_scope("embedding") as scope:
                    #
                    W_fe = tf.get_variable(
                        name="W_fe",
                        initializer=tf.random_uniform([self.vocab_size + 1, self.dis_emb_dim], -1.0, 1.0))
                    # scope.reuse_variables()
                    embedded_chars = tf.nn.embedding_lookup(W_fe, Feature_input + 1)
                    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

                # Create a convolution + maxpool layer for each filter size
                pooled_outputs = []
                for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                    with tf.name_scope("conv-maxpool-%s" % filter_size) as scope:
                        # Convolution Layer
                        filter_shape = [filter_size, self.dis_emb_dim, 1, num_filter]
                        W = tf.get_variable(name="W-%s" % filter_size,
                                            initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                        b = tf.get_variable(name="b-%s" % filter_size,
                                            initializer=tf.constant(0.1, shape=[num_filter]))
                        # scope.reuse_variables()
                        conv = tf.nn.conv2d(
                            embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv-%s" % filter_size)
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-%s" % filter_size)
                        # Maxpooling over the outputs
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool-%s" % filter_size)
                        pooled_outputs.append(pooled)
                        #
                # Combine all the pooled features
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])

                # Add highway
                with tf.name_scope("highway"):
                    h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

                    # Add dropout
                with tf.name_scope("dropout"):
                    h_drop = tf.nn.dropout(h_highway,dropout_keep_prob)

            return h_drop

        return unit

    def classification(self, D_input):
        with tf.variable_scope('Discriminator'):
            W_d = tf.Variable(tf.truncated_normal([self.num_filters_total, self.num_classes], stddev=0.1), name="W")
            b_d = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.D_l2_loss += tf.nn.l2_loss(W_d)
            self.D_l2_loss += tf.nn.l2_loss(b_d)
            self.scores = tf.nn.xw_plus_b(D_input, W_d, b_d, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        return self.scores, self.predictions, self.ypred_for_auc