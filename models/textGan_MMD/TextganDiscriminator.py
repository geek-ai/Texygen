import tensorflow as tf


def id_num():
    try:
        id_num.counter += 1
    except AttributeError:
        id_num.counter = 1
    return id_num.counter


def linear(input_, output_size, scope=None, name=''):
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


    name = str(id_num())
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope+name or "SimpleLinear"+name):
        matrix = tf.get_variable("Matrix"+name, [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias"+name, [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway', name=''):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope+name):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx, name=name))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx, name=name) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            emd_dim, filter_sizes, num_filters, g_embeddings=None,
            batch_size = 64, l2_reg_lambda=0.0, dropout_keep_prob=1):
        self.embbeding_mat = g_embeddings

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_x_lable = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.input_y = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y_lable = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.zh = tf.placeholder(tf.float32, [None, emd_dim], name="zh")
        # self.zc = tf.placeholder(tf.float32, [None], name="zc")

        self.dropout_keep_prob = dropout_keep_prob
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        # self.embbeding_mat = None
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([emd_dim, emd_dim], -1.0, 1.0),
                    name="W")
                # self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            self.W_conv = list()
            self.b_conv = list()
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    self.W_conv.append(W)
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    self.b_conv.append(b)

            num_filters_total = sum(self.num_filters)
            with tf.name_scope("output"):
                self.Wo = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name="W")
                self.bo = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")

            # recon layer
            with tf.name_scope("recon"):
                self.Wzh = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="Wz")
                self.bzh = tf.Variable(tf.constant(0.0, shape=[1]), name="bz")
                # self.Wzc = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="Wz")
                # self.bzc = tf.Variable(tf.constant(0.0, shape=[1]), name="bz")

            # Create a convolution + maxpool layer for each filter size
            # pooled_outputs = []
            # index = 0
            # for filter_size, num_filter in zip(filter_sizes, num_filters):
            #     with tf.name_scope("conv-maxpool-%s-midterm" % filter_size):
            #         # Convolution Layer
            #         conv = tf.nn.conv2d(
            #             self.embedded_chars_expanded,
            #             self.W_conv[index],
            #             strides=[1, 1, 1, 1],
            #             padding="VALID",
            #             name="conv")
            #         # Apply nonlinearity
            #         h = tf.nn.relu(tf.nn.bias_add(conv, self.b_conv[index]), name="relu")
            #         # Maxpooling over the outputs
            #         pooled = tf.nn.max_pool(
            #             h,
            #             ksize=[1, sequence_length - filter_size + 1, 1, 1],
            #             strides=[1, 1, 1, 1],
            #             padding='VALID',
            #             name="pool")
            #         pooled_outputs.append(pooled)
            #
            # # Combine all the pooled features
            # num_filters_total = sum(num_filters)
            # self.h_pool = tf.concat(pooled_outputs, 3)
            # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            #
            # # Add highway
            # with tf.name_scope("highway"):
            #     self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)
            #
            # # Add dropout
            # with tf.name_scope("dropout"):
            #     self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)
            #
            # # Final (unnormalized) scores and predictions
            # with tf.name_scope("output"):
            #     W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #     self.ypred_for_auc = tf.nn.softmax(self.scores)
            #     self.predictions = tf.argmax(self.scores, 1, name="predictions")

            input_xy = tf.concat([self.input_x, self.input_y], axis=0)
            input_label = tf.concat([self.input_x_lable, self.input_y_lable], axis=0)

            input_x = tf.nn.embedding_lookup(self.embbeding_mat, input_xy)  # batch_size x seq_length x g_emb_dim
            scores, ypred_for_auc, predictions = self.predict(input_x=input_x)

            def compute_pairwise_distances(x, y):
                """Computes the squared pairwise Euclidean distances between x and y.
                Args:
                  x: a tensor of shape [num_x_samples, num_features]
                  y: a tensor of shape [num_y_samples, num_features]
                Returns:
                  a distance matrix of dimensions [num_x_samples, num_y_samples].
                Raises:
                  ValueError: if the inputs do no matched the specified dimensions.
                """

                if not len(x.get_shape()) == len(y.get_shape()) == 2:
                    raise ValueError('Both inputs should be matrices.')

                if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
                    raise ValueError('The number of features should be the same.')

                norm = lambda x: tf.reduce_sum(tf.square(x), 1)

                # By making the `inner' dimensions of the two matrices equal to 1 using
                # broadcasting then we are essentially substracting every pair of rows
                # of x and y.
                # x will be num_samples x num_features x 1,
                # and y will be 1 x num_features x num_samples (after broadcasting).
                # After the substraction we will get a
                # num_x_samples x num_features x num_y_samples matrix.
                # The resulting dist will be of shape num_y_samples x num_x_samples.
                # and thus we need to transpose it again.
                return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

            def gaussian_kernel_matrix(x, y, sigmas=None):
                r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
                We create a sum of multiple gaussian kernels each having a width sigma_i.
                Args:
                  x: a tensor of shape [num_samples, num_features]
                  y: a tensor of shape [num_samples, num_features]
                  sigmas: a tensor of floats which denote the widths of each of the
                    gaussians in the kernel.
                Returns:
                  A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
                """
                if sigmas is None:
                    sigmas = [
                        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                        1e3, 1e4, 1e5, 1e6
                    ]
                beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

                dist = compute_pairwise_distances(x, y)

                s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

                return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

            def calc_mmd(x, y):
                cost = tf.reduce_mean(gaussian_kernel_matrix(x, x))
                cost += tf.reduce_mean(gaussian_kernel_matrix(y, y))
                cost -= 2 * tf.reduce_mean(gaussian_kernel_matrix(x, y))

                # We do not allow the loss to become negative.
                cost = tf.where(cost > 0, cost, 0, name='value')

                return cost

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                gan_loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_label)

                x_feature = self.feature(input_x=self.input_x, name='x')
                y_feature = self.feature(input_x=self.input_y, name='y')
                mmd_loss = calc_mmd(x_feature, y_feature)

                z_hat = tf.matmul(x_feature, self.Wzh)
                recon_loss = - tf.norm(tf.subtract(z_hat, self.zh), axis=1)
                self.loss = tf.reduce_mean(gan_loss) + l2_reg_lambda * l2_loss + 0.1 * mmd_loss + 0.1 * recon_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def set_embbeding_mat(self, generator):
        raise NotImplementedError



    def feature(self, input_x, name = ''):
        if len(input_x.get_shape()) == 2:
            # incase input_x : batch_size x seq_length [tokens]
            input_x = tf.nn.embedding_lookup(self.embbeding_mat, input_x)
        # input_x:  batch_size x seq_length x g_emb_dim
        pooled_outputs = []
        index = -1
        ## embedded_chars = tf.nn.embedding_lookup(self.W, input_x)
        # embedded_chars = tf.matmul(input_x, self.W)
        embedded_chars = tf.scan(lambda a, x: tf.matmul(x, self.W), input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
            index += 1
            with tf.name_scope("conv-maxpool-%s-midterm" % filter_size):
                # Convolution Layer
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    self.W_conv[index],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, self.b_conv[index]), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(self.num_filters)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add highway
        with tf.name_scope("highway" + name):
            h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, )
        return h_highway

    def predict(self, input_x):
        # input_x:  batch_size x seq_length x g_emb_dim
        l2_loss = tf.constant(0.0)
        h_highway = self.feature(input_x)
        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_highway, self.dropout_keep_prob)
        num_filters_total = sum(self.num_filters)
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            l2_loss += tf.nn.l2_loss(self.Wo)
            l2_loss += tf.nn.l2_loss(self.bo)
            scores = tf.nn.xw_plus_b(h_drop, self.Wo, self.bo, name="scores")
            ypred_for_auc = tf.nn.softmax(scores)
            predictions = tf.argmax(scores, 1, name="predictions")

        return scores, ypred_for_auc, predictions
