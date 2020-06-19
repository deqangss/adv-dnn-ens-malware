"""Projected gradient decsent, l2, l-infinity norm based attack"""

import os
import sys
import warnings

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM = {
    'max_iteration': 50,
    'batch_size': 50
}

class BGA_K(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer=None,
                 verbose = False, **kwargs):
        super(BGA_K, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.parse(**kwargs)
        self.scaled_max_extended = tf.placeholder(dtype = tf.float32, shape = [None, input_dim], name = "MAX_BOUND")
        self.scaled_min_extended = tf.placeholder(dtype = tf.float32, shape = [None, input_dim], name = "MIN_BOUND")

    def graph_(self, x_input, y_input):
        # sqrt_m
        sqrt_m = tf.sqrt(tf.to_float(x_input.get_shape().as_list()[1]))
        # boundary
        scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) + # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)), # may be useful to reset the lower bound
            x_input # upper bound for positions no perturbations allowed
        )
        scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            x_input
        )

        def _cond(i, _):
            return tf.less(i, self.iterations)

        def _body(i, x_adv_tmp):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits= self.model.get_logits(x_adv_tmp),
                    labels= y_input
                )
            )
            grad = tf.gradients(loss, x_adv_tmp)[0]

            grad_l2norm = tf.sqrt(tf.reduce_sum(tf.square(grad),
                                             axis =-1,
                                             keepdims=True))

            perturbations = tf.cast((tf.greater(sqrt_m * (1. - 2 * x_adv_tmp) * grad,
                                                grad_l2norm)),
                                    tf.float32
                                    )
            x_adv_tmp = x_adv_tmp + perturbations
            x_adv_tmp = tf.clip_by_value(x_adv_tmp, clip_value_min= scaled_min_extended, clip_value_max=scaled_max_extended)

            return i + 1, x_adv_tmp

        _, adv_x_batch = tf.while_loop(_cond, _body, (tf.zeros([]), x_input),
                                       maximum_iterations=self.iterations,
                                       back_prop= False
                                       )

        # map to discrete domain
        if self.normalizer is not None:
            # projection in the discrete domain with the threshold: 0.5
            x_adv = tf.rint(tf.divide(adv_x_batch - self.normalizer.min_, self.normalizer.scale_))
            # re-project back
            x_adv_normalized = tf.multiply(x_adv, self.normalizer.scale_) + self.normalizer.min_
        else:
            x_adv_normalized = tf.rint(adv_x_batch)

        return x_adv_normalized

    def parse(self, max_iteration = 50, batch_size = 50, **kwargs):
        self.iterations = max_iteration
        self.batch_size = batch_size

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def graph(self, x_input, y_input):
        # sqrt_m
        sqrt_m = tf.sqrt(tf.to_float(x_input.get_shape().as_list()[1]))

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits= self.model.get_logits(x_input),
                labels= y_input
            )
        )
        grad = tf.gradients(loss, x_input)[0]

        grad_l2norm = tf.sqrt(tf.reduce_sum(tf.square(grad),
                                         axis =-1,
                                         keepdims=True))

        perturbations = tf.cast((tf.greater(sqrt_m * (1. - 2 * x_input) * grad,
                                            grad_l2norm)),
                                tf.float32
                                )
        x_adv_tmp = x_input + perturbations
        x_adv_tmp = tf.clip_by_value(x_adv_tmp,
                                     clip_value_min= self.scaled_min_extended,
                                     clip_value_max= self.scaled_max_extended)

        # map to discrete domain
        if self.normalizer is not None:
            # projection in the discrete domain with the threshold: 0.5
            x_adv = tf.rint(tf.divide(x_adv_tmp - self.normalizer.min_, self.normalizer.scale_))
            # re-project back
            x_adv_normalized = tf.multiply(x_adv, self.normalizer.scale_) + self.normalizer.min_
        else:
            x_adv_normalized = tf.rint(x_adv_tmp)

        return x_adv_normalized

    def perturb(self, dataX, ground_truth_labels, sess = None):
        # TF tensor
        self.x_adv_batch = self.graph(self.model.x_input, self.model.y_input)

        try:
            input_data = utils.DataProducer(dataX, ground_truth_labels, batch_size= self.batch_size, name = 'test')

            # load baseline model parameters
            sess_close_flag = False
            if sess is None:
                cur_checkpoint = tf.train.latest_checkpoint(self.model.save_dir)
                config_gpu = tf.ConfigProto(log_device_placement=True)
                config_gpu.gpu_options.allow_growth = True
                sess = tf.Session(config=config_gpu)
                saver = tf.train.Saver()
                saver.restore(sess, cur_checkpoint)
                sess_close_flag = True
        except IOError as ex:
            raise IOError("BGA-K attack: Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []

            for idx, X_batch, y_batch in input_data.next_batch():
                loss = sess.run(self.model.y_xent, feed_dict={
                    self.model.x_input: X_batch,
                    self.model.y_input: y_batch,
                    self.model.is_training: False,
                })

                worst_loss = loss
                X_worst = np.copy(X_batch)

                # boundary
                _scaled_max_extended = np.maximum(
                    np.multiply(self.scaled_clip_max, self.insertion_perm_array) +
                    np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
                    X_batch
                )
                _scaled_min_extended = np.minimum(
                    np.multiply(self.scaled_clip_min, self.removal_perm_array) +
                    np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
                    X_batch
                )

                for iter_i in range(self.iterations):
                    _x_adv_tmp = sess.run(self.x_adv_batch, feed_dict={
                        self.model.x_input: X_worst,
                        self.model.y_input: y_batch,
                        self.model.is_training: False,
                        self.scaled_max_extended: _scaled_max_extended,
                        self.scaled_min_extended: _scaled_min_extended
                    })

                    _loss = sess.run(self.model.y_xent, feed_dict={
                        self.model.x_input: _x_adv_tmp,
                        self.model.y_input: y_batch,
                        self.model.is_training: False,
                        })
                    change_indicator = _loss > worst_loss
                    X_worst[change_indicator] = _x_adv_tmp[change_indicator]
                    worst_loss[change_indicator] = _loss[change_indicator]

                x_adv.append(X_worst)
                # accuracy
                curr_accuracy = utils.test_func(sess, self.model, X_worst, y_batch, batch_size=50)
                if self.verbose:
                    print(
                        "\tBGA-K attack: mini-batch {}/{}, the current accuracy is {:.5} on a batch of samples".format(idx + 1,
                                                                                                                       input_data.mini_batches,
                                                                                                                        curr_accuracy))
            x_adv_normalized = np.concatenate(x_adv)
            if self.verbose:
                accuracy = utils.test_func(sess, self.model, x_adv_normalized, ground_truth_labels, batch_size=50)
                print("The classification accuracy is {:.5} on adversarial feature vectors.".format(accuracy))
                assert np.all(x_adv_normalized - dataX >= 0)
                perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_normalized - dataX) > 1e-6, axis=1))
                perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_normalized - dataX), axis=1))
                perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv_normalized - dataX), axis=1)))
                print("\t The average l0 norm of perturbations is {:5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:5}".format(perturbations_amount_l2))
                print(np.mean(np.sum(x_adv_normalized - dataX, axis=1)))
            if sess_close_flag:
                sess.close()

        return dataX, x_adv_normalized, ground_truth_labels



