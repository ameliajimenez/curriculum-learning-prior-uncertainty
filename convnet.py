from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
NET_VARIABLES = 'net_variables'
UPDATE_OPS_COLLECTION = 'update_ops'  # must be grouped with training op

activation = tf.nn.relu


def create_convnet(x, n_class, is_training, weights_seed=0):
    """
    Creates a new convolutional net for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param n_class: number of output labels
    :param is_training: boolean tf.Variable, true indicates training phase
    :param weights_seed: tensorflow seed for the initialization of the weights
    """

    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]

    x_image = tf.reshape(x, tf.stack([-1, nx, ny, 1]))

    c = Config()
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['use_bias'] = True  # if we use batch norm, this param is set to False
    c['fc_units_out'] = n_class

    # conv1a
    with tf.variable_scope('conv1a'):
        c['conv_filters_out'] = 128
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x_image, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)

    # conv1b
    with tf.variable_scope('conv1b'):
        c['conv_filters_out'] = 128
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)

    # conv1c
    with tf.variable_scope('conv1c'):
        c['conv_filters_out'] = 128
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)

    # pool1
    x = _max_pool(x, ksize=2, stride=2)

    # drop1
    x = control_flow_ops.cond(c['is_training'],
                              lambda: tf.nn.dropout(x, 0.5),
                              lambda: x)

    # conv2a
    with tf.variable_scope('conv2a'):
        c['conv_filters_out'] = 256
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)

    # conv2b
    with tf.variable_scope('conv2b'):
        c['conv_filters_out'] = 256
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)

    # conv2c
    with tf.variable_scope('conv2c'):
        c['conv_filters_out'] = 256
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'SAME')
        x = bn(x, c)
        x = activation(x)

    # pool2
    x = _max_pool(x, ksize=2, stride=2)

    # drop2
    x = control_flow_ops.cond(c['is_training'],
                              lambda: tf.nn.dropout(x, 0.5),
                              lambda: x)

    # conv3a
    with tf.variable_scope('conv3a'):
        c['conv_filters_out'] = 512
        c['ksize'] = 3
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'VALID')
        x = bn(x, c)
        x = activation(x)

    # conv3b
    with tf.variable_scope('conv3b'):
        c['conv_filters_out'] = 256
        c['ksize'] = 1
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'VALID')
        x = bn(x, c)
        x = activation(x)

    # conv3c
    with tf.variable_scope('conv3c'):
        c['conv_filters_out'] = 128
        c['ksize'] = 1
        c['stride'] = 1
        x = conv(x, c, weights_seed, 'VALID')
        x = bn(x, c)
        x = activation(x)

    # pool3
    x = _avg_pool2d(x, pool_size=[6, 6], strides=[1, 1])
    x = tf.squeeze(x, axis=[1, 2])
    x_embedding = x

    # dense
    with tf.variable_scope('fc6'):
        x = fc(x, c, keep_prob=1., weights_seed=weights_seed)
    logits = x

    tf.summary.histogram("logits/activations", x)

    return logits, x_embedding


def bn(x, c):
    """
    Batch normalization layer.

    :param x: input tensor, shape [?, nx, ny, channels]
    :param c: layer configuration
    """
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer)
        return x + bias

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer)
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer)

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer,
                                    trainable=False)

    # These ops will only be performed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

    return x


def fc(x, c, keep_prob, weights_seed):
    """
    Fully connected layer.

    :param x: input tensor, shape [?, nx, ny, channels]
    :param c: layer configuration
    :param keep_prob: dropout rate
    :param weights_seed: tensorflow seed for the convolutional weights
    """
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV, seed=weights_seed)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_DECAY)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)

    x = control_flow_ops.cond(c['is_training'],
                              lambda: tf.nn.dropout(x, keep_prob),
                              lambda: x)
    return x


def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    """
    A little wrapper around tf.get_variable to do weight decay and add to net collection.

    :param name: layer name
    :param shape:
    :param initializer:
    :param weight_decay (optional):
    :param dtype (optional):
    :param trainable (optional):
    """
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, NET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c, weights_seed, padding='SAME'):
    """
    Convolutional layer.

    :param x: input tensor, shape [?, nx, ny, channels]
    :param c: layer configuration
    :param weights_seed: tensorflow seed for the convolutional weights
    :param padding (optional): type of padding to aply to convolution ('SAME' or 'VALID')
    """
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]

    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV, seed=weights_seed)

    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)

    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)


def _max_pool(x, ksize=3, stride=2, name=None):
    """
    Max pooling layer.

    :param x: input tensor, shape [?, nx, ny, channels]
    :param ksize (optional): kernel size for the pooling operation
    :param stride (optional): stride for the pooling operation
    :param name (optional): layer name
    """
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)


def _avg_pool2d(x, pool_size, strides, padding='VALID', name=None):
    """
    Average pooling layer.

    :param x: input tensor, shape [?, nx, ny, channels]
    :param pool_size: pooling size
    :param strides: stride for the pooling operation
    :param padding (optional): type of padding to aply to convolution ('SAME' or 'VALID')
    :param name (optional): layer name
    """
    return tf.layers.average_pooling2d(x,
                                       pool_size=pool_size,  # size of the pooling window
                                       strides=strides,
                                       padding=padding,
                                       name=name)


class ConvNet(object):
    """
    A Convolutional Network (ConvNet) implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param is_training: (optional) flag to differentiate train and test stages
    """

    def __init__(self, channels=1, n_class=10, is_training=False, use_mask=False, cost_name='cross_entropy'):

        tf.reset_default_graph()

        self.n_class = n_class
        self.is_training = is_training
        self.use_mask = use_mask

        self.x = tf.placeholder("float", shape=[None, None, None, channels])  # image
        self.y = tf.placeholder("float", shape=[None, n_class])  # one-hot encoding for labels
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
        self.mask = tf.placeholder("float", shape=[None])

        logits, x_embedding = create_convnet(self.x, n_class, self.is_training)

        self.cost, self.individual_losses = self._get_cost(logits, cost_name)
        self.predicter_embedding = x_embedding
        self.predicter = tf.nn.softmax(logits)
        self.predicter_logits = logits

        self.correct_pred = tf.equal(tf.argmax(self.predicter, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits, cost_name):
        """
        Constructs the cost function: cross_entropy

        :param logits: output logits of the network
        :returns average loss and individual losses for the batch
        """

        if cost_name == 'cross_entropy':
            individual_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=self.y)
            if self.use_mask:
                individual_losses = tf.multiply(self.mask, individual_losses)
            loss = tf.reduce_mean(individual_losses)

        elif 'weights' in cost_name:
            individual_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=self.y)
            weighted_losses = tf.multiply(self.mask, individual_losses)
            loss = tf.reduce_mean(weighted_losses)

        return loss, individual_losses

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: probability distribution over the classes
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test,
                                                             self.y: y_dummy,
                                                             self.keep_prob: 1.})
        return prediction

    def predict_embedding(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: feature embedding space
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(self.predicter_embedding, feed_dict={self.x: x_test,
                                                                       self.y: y_dummy,
                                                                       self.keep_prob: 1.})
        return prediction

    def predict_logits(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: logits
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(self.predicter_logits, feed_dict={self.x: x_test,
                                                                    self.y: y_dummy,
                                                                    self.keep_prob: 1.})
        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)
