from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import logging
import os
import shutil
import tensorflow as tf

UPDATE_OPS_COLLECTION = 'update_ops'  # must be grouped with training op


class Trainer(object):
    """
    Trains a net instance

    :param net: the net instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    verification_batch_size = 4

    def __init__(self, net, batch_size=16, optimizer="adam", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, global_step):

        loss_ = self.net.cost

        if self.optimizer == "momentum":
            print('momentum optimizer')
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.01)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.5)
            momentum = self.opt_kwargs.pop("momentum", 0.9)
            decay_steps = self.opt_kwargs.pop("decay_steps", 100)
            type_decay = self.opt_kwargs.pop("type_decay", 'exponential')

            if type_decay == 'exponential':
                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=global_step,
                                                                     decay_steps=decay_steps,
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)

            opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                             **self.opt_kwargs)
            grads = opt.compute_gradients(loss_)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        elif self.optimizer == 'adam':
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.01)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.1)
            beta1 = self.opt_kwargs.pop("beta1", 0.9)
            beta2 = self.opt_kwargs.pop("beta2", 0.99999)
            decay_steps = self.opt_kwargs.pop("decay_steps", 100)
            type_decay = self.opt_kwargs.pop("type_decay", 'exponential')

            if type_decay == 'exponential':
                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=global_step,
                                                                     decay_steps=decay_steps,
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)

            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                         beta1=beta1,
                                         beta2=beta2)

            grads = opt.compute_gradients(loss_)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        else:
            print('Optimizer not available')

        return train_op

    def _initialize(self, output_path, restore):
        """
        Initializes the network

        :param output_path: path to model location
        :param restore: flag to restore previous model (True or False)
        """
        global_step = tf.Variable(0)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, training_iters, strategy, epochs=100, np_seed=0, dropout=0.8,
              restore=False, compute_uncertainty=False, anti_curriculum=False, replacement=False):
        """
        Launches the training process

        :param data_provider: callable returning training and validation data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param np_seed: numpy seed for the sampling
        :param dropout: dropout rate
        :param restore: flag to restore previous model (True or False)
        """
        train_batch_size = self.batch_size

        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path

        init = self._initialize(output_path, restore)

        with tf.Session() as sess:
            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            summary_writer_train = tf.summary.FileWriter(output_path + '/logs/train', graph=sess.graph)
            summary_writer_val = tf.summary.FileWriter(output_path + '/logs/val', graph=sess.graph)

            logging.info("Start optimization")

            # early stop
            best_loss_val = np.infty
            wait_epochs = 20  # number of epochs to wait before stopping training if there is no improvement
            best_epochs = []
            last_epoch = 0

            n_iterations_validation = data_provider.val.num_examples // train_batch_size
            n_iterations_per_epoch = training_iters

            for epoch in range(epochs):
                total_loss = 0.0

                # if (epoch - last_epoch) > wait_epochs:
                #     print('Training has been stopped. Model did not improve in last {} epochs'.format(wait_epochs))
                #     break

                if compute_uncertainty and epoch != 0:
                    if epoch == 1:
                        data_provider.train.remove_epoch()
                    print('recalculate uncertainty at epoch {:2d}'.format(epoch))
                    dropout_unc = 0.7
                    inference_times = 5
                    n_train = data_provider.train._num_examples
                    predictions = np.zeros((n_train, self.net.n_class, inference_times), np.float32)

                    for n_run in range(0, inference_times):
                        start = 0
                        end = train_batch_size
                        for step in range(1, n_iterations_per_epoch + 1):
                            # load data
                            batch_x, batch_y, batch_probs, batch_indices = data_provider.train.next_batch(train_batch_size)
                            batch_x = np.pad(batch_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # pad input
                            batch_mask = np.ones(train_batch_size, np.float32)

                            # Measure uncertainty
                            preds = sess.run(self.net.predicter,  # self.net.predicter_logits
                                             feed_dict={self.net.x: batch_x,
                                                        self.net.y: batch_y,
                                                        self.net.mask: batch_mask,
                                                        self.net.keep_prob: dropout_unc})

                            if end > n_train:
                                end = n_train
                                preds = preds[:(end - start)]

                            # dense predictions
                            predictions[start:end, :, n_run] = preds

                            start += batch_x.shape[0]
                            end += batch_x.shape[0]
                            # after every inference run, start=0 and subtract one epoch from the count
                        data_provider.train.remove_epoch()

                    metrics = measure_uncertainty(predictions)
                    unc = metrics['pe_norm']
                    if anti_curriculum:
                        unc = 1.0 - unc
                    data_provider.train.change_probs(unc)

                if strategy == 'baseline':
                    # baseline
                    for step in range(1, n_iterations_per_epoch + 1):
                        batch_x, batch_y, batch_probs, batch_indices = data_provider.train.next_batch(train_batch_size)
                        batch_probs = np.random.rand(train_batch_size)
                        batch_x = np.pad(batch_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # padded input
                        batch_probs /= batch_probs.max()
                        batch_mask = batch_probs

                        # Run optimization op (backprop)
                        _, loss, lr = sess.run(
                            (self.optimizer, self.net.cost, self.learning_rate_node),
                            feed_dict={self.net.x: batch_x,
                                       self.net.y: batch_y,
                                       self.net.mask: batch_mask,
                                       self.net.keep_prob: dropout})

                        total_loss += loss

                elif strategy == 'reorder':
                    # prior REORDER
                    for step in range(1, n_iterations_per_epoch + 1):
                        batch_x, batch_y, batch_probs, batch_indices = \
                            data_provider.train.next_batch_probs(train_batch_size, np_seed=np_seed, decay=True,
                                                                 replacement=replacement)
                        batch_x = np.pad(batch_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # padded input
                        batch_probs /= batch_probs.max()
                        batch_mask = batch_probs

                        # Run optimization op (backprop)
                        _, loss, lr = sess.run(
                            (self.optimizer, self.net.cost, self.learning_rate_node),
                            feed_dict={self.net.x: batch_x,
                                       self.net.y: batch_y,
                                       self.net.mask: batch_mask,
                                       self.net.keep_prob: dropout})

                        total_loss += loss

                elif strategy == 'subsets':
                    # prior SUBSETS
                    n_iterations_per_epoch_subset = int(np.ceil(data_provider.train._subset_size * 1.0 / train_batch_size))

                    for step in range(1, n_iterations_per_epoch_subset + 1):
                        batch_x, batch_y, batch_probs, batch_indices = \
                            data_provider.train.next_subset_batch(train_batch_size, np_seed=np_seed, grow=True,
                                                                  replacement=replacement, random=True, decay=False)
                        batch_x = np.pad(batch_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # padded input
                        batch_probs /= batch_probs.max()
                        batch_mask = batch_probs

                        # Run optimization op (backprop)
                        _, loss, lr = sess.run(
                            (self.optimizer, self.net.cost, self.learning_rate_node),
                            feed_dict={self.net.x: batch_x,
                                       self.net.y: batch_y,
                                       self.net.mask: batch_mask,
                                       self.net.keep_prob: dropout})

                        total_loss += loss

                elif strategy == 'weights':
                    # prior WEIGHTS  # cost_name = 'weighted_cross_entropy
                    for step in range(1, n_iterations_per_epoch + 1):
                        batch_x, batch_y, batch_probs, batch_indices = data_provider.train.next_batch_weights_only_decay(train_batch_size)
                        batch_x = np.pad(batch_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # padded input
                        batch_probs /= batch_probs.max()
                        batch_mask = batch_probs

                        # Run optimization op (backprop)
                        _, loss, lr = sess.run(
                            (self.optimizer, self.net.cost, self.learning_rate_node),
                            feed_dict={self.net.x: batch_x,
                                       self.net.y: batch_y,
                                       self.net.mask: batch_mask,
                                       self.net.keep_prob: dropout})

                        total_loss += loss

                # summary train
                self.output_minibatch_stats(sess, summary_writer_train, epoch, batch_x, batch_y, batch_mask,
                                            dropout, phase='Train')

                loss_vals = []
                for step in range(1, n_iterations_validation + 1):
                    # validation samples
                    val_x, val_y, val_probs, val_indices = data_provider.val.next_batch(train_batch_size)
                    val_x = np.pad(val_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')  # padded input

                    val_probs = np.ones(train_batch_size, np.float32)

                    loss_val = sess.run(self.net.cost,
                                        feed_dict={self.net.x: val_x,
                                                   self.net.y: val_y,
                                                   self.net.mask: val_probs,
                                                   self.net.keep_prob: 1.})

                    loss_vals.append(loss_val)

                # summary validation
                self.output_minibatch_stats(sess, summary_writer_val, epoch, val_x, val_y, val_probs, dropout, phase='Val')

                loss_val = np.mean(loss_vals)

                # save model for minimum validation loss
                if loss_val < best_loss_val and epoch != 0:
                    best_epochs.append([epoch, loss_val])
                    last_epoch = epoch
                    save_path = self.net.save(sess, save_path)
                    best_loss_val = loss_val
                    print('Saved at epoch: {}, Validation loss: {:.4f}'.format(epoch, loss_val))

            logging.info("Optimization Finished!")

            return save_path

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, batch_mask, dropout, phase):

        """
        Evaluation after epoch

        :param sess: current session
        :param summary_writer: writer for the logs
        :param step: number of training mini batch iteration
        :param batch_x: data to predict on. Shape [batch_size, nx, ny, channels]
        :param batch_y: classification label. Shape [batch_size, n_class]
        :param dropout: dropout rate
        :param phase: training or test phase
        """

        if phase == 'Train':
            # Calculate batch loss and accuracy
            summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                            self.net.cost,
                                                            self.net.accuracy,
                                                            self.net.predicter],
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.mask: batch_mask,
                                                                      self.net.keep_prob: dropout})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step, loss, acc, error_rate(predictions, batch_y)))
        else:
            self.net.is_training = False
            # Calculate batch loss and accuracy
            summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                            self.net.cost,
                                                            self.net.accuracy,
                                                            self.net.predicter],
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.mask: batch_mask,
                                                                      self.net.keep_prob: 1.})
            self.net.is_training = True
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            logging.info(
                "Iter {:}, Minibatch Loss= {:.4f}, Validation Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step, loss, acc, error_rate(predictions, batch_y)))


def error_rate(predictions, labels):
    """
    Returns the error rate based on dense predictions and 1-hot labels.

    :param predictions: labels predicted by the network
    :param labels: ground truth classification labels
    """
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / (predictions.shape[0]))


def measure_uncertainty(predictions):
    """
    Returns uncertainty measurements of each training sample: predictive entropy (pe), expected entropy (ee), mutual information (mi)
    :param predictions: input tensor, shape [n_train, n_class, n_inference_dropouts]
    """
    epsilon = 1e-8

    mean_qim = np.mean(predictions, axis=2)
    indiv_pe = np.multiply(mean_qim, - np.log(np.clip(mean_qim, epsilon, 1.)))
    pe = np.sum(indiv_pe, 1)
    pe_norm = (pe - pe.min()) / (pe.max() - pe.min() + epsilon).astype(np.float32)

    indiv_ee = np.sum(np.multiply(predictions, - np.log(np.clip(predictions, epsilon, 1.))), 1)
    ee = np.mean(indiv_ee, 1)
    ee_norm = (ee - ee.min()) / (ee.max() - ee.min() + epsilon).astype(np.float32)

    mi = pe - ee
    mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + epsilon).astype(np.float32)

    metrics = {'pe_norm': pe_norm, 'pe': pe,
               'ee_norm': ee_norm,  'ee': ee,
               'mi_norm': mi_norm, 'mi': mi}

    return metrics
