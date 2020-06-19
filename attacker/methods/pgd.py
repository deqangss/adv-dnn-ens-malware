"""
Projected gradient descent, l2, l-infinity norm based attack
link: https://adversarial-ml-tutorial.org
malware related paper: https://arxiv.org/abs/2004.07919
"""

import os
import sys
import warnings

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM = {
    'step_size': 0.02,
    'ord': 'l2', # 'l2', 'linfinity',
    'rand_round': False,
    'max_iteration': 50,
    'batch_size': 50
}

class PGD(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer=None,
                 verbose = False, **kwargs):
        super(PGD, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)
        self.step_size = DEFAULT_PARAM['step_size']
        self.ord = DEFAULT_PARAM['ord']
        self.rand_round = DEFAULT_PARAM['rand_round']
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.is_init_graph = False
        self.parse(**kwargs)


    def project_perturbations(self, x_input, perturbations):
        # boundary
        scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) +
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            x_input
        )
        scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            x_input
        )

        return tf.clip_by_value(x_input + perturbations, clip_value_min= scaled_min_extended, clip_value_max= scaled_max_extended)

    def init_graph(self):
        # TF tensor
        self.x_adv_batch = self.graph(self.model.x_input, self.model.y_input)
        self.is_init_graph = True

    def graph(self, x_input, y_input, name = 'pgd'):
        def _cond(i, _):
            return tf.less(i, self.iterations)

        # boundary
        scaled_max_extended = tf.maximum( # broadcasting
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) + # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)), # may be useful to reset the lower bound
            x_input # upper bound for positions no perturbations allowed
        )
        scaled_min_extended = tf.minimum( # broadcasting
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            x_input
        )

        def _body(i, x_adv_tmp):
            loss = tf.losses.sparse_softmax_cross_entropy(logits=self.model.get_logits(x_adv_tmp),
                                                          labels=y_input) # average loss, may cause leakage issue
            grad = tf.gradients(loss, x_adv_tmp)[0]
            if self.ord == 'l2':
                perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size), ord = 2)
            elif self.ord == 'l-infinity':
                perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size))
            elif self.ord == 'l1':
                raise NotImplementedError("L1 norm based attack is not implemented here.")
            else:
                raise ValueError("'l-infinity' are supported.")

            x_adv_tmp = x_adv_tmp + perturbations
            x_adv_tmp = tf.clip_by_value(x_adv_tmp,
                                         clip_value_min= scaled_min_extended,
                                         clip_value_max=scaled_max_extended)

            # l2 norm ball based constraint, negected
            # avoid_zero_div = 1e-6
            # if self.ord == 'l2':
            #     norm = tf.sqrt(tf.maximum(avoid_zero_div,
            #                               tf.reduce_sum(tf.square(x_adv_tmp - self.x_input),axis=1,keepdims=True)))
            #
            #     factor = tf.minimum(1., tf.divide(1., norm))
            #     x_adv_tmp = self.x_input + (x_adv_tmp - self.x_input) * factor

            return i + 1, x_adv_tmp

        _, adv_x_batch = tf.while_loop(_cond, _body, (tf.zeros([]), x_input),
                                       maximum_iterations=self.iterations)

        # map to discrete domain
        if self.normalizer is not None:
            unnorm_x = tf.divide(x_input - self.normalizer.min_, self.normalizer.scale_)
            unnorm_x_adv = tf.divide(adv_x_batch - self.normalizer.min_,self.normalizer.scale_)
            if not self.rand_round:
                # project to discrete domain with the threshold:0.5
                x_adv = tf.rint(unnorm_x_adv)
                # re-project back
                x_adv_normalized = tf.multiply(x_adv, self.normalizer.scale_) + self.normalizer.min_
            else:
                real_valued_perturb = unnorm_x_adv - unnorm_x
                abs_real_perturb = tf.abs(real_valued_perturb)
                abs_integer_perturb = tf.cast(tf.cast(abs_real_perturb, tf.int32), tf.float32)
                decimal_part = abs_real_perturb - abs_integer_perturb
                random_threshold = tf.random_uniform(tf.shape(decimal_part), minval=0., maxval=1.)
                round_decimal = utils.round_x(decimal_part, random_threshold)
                x_adv = tf.sign(real_valued_perturb)*(abs_integer_perturb + round_decimal) + unnorm_x
                # re-project back
                x_adv_normalized = tf.multiply(x_adv, self.normalizer.scale_) + self.normalizer.min_
        else:
            if not self.rand_round:
                x_adv_normalized = tf.rint(adv_x_batch)
            else:
                random_threshold = tf.random_uniform(adv_x_batch.shape, minval=0., maxval=1.)
                x_adv_normalized = utils.round_x(adv_x_batch, random_threshold)

        return x_adv_normalized

    def parse(self, step_size = 0.01, ord ='l2', rand_round = False, max_iteration = 50, batch_size = 50, **kwargs):
        self.step_size = step_size
        self.ord = ord
        self.rand_round = rand_round
        self.iterations = max_iteration
        self.batch_size = batch_size

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def perturb(self, dataX, ground_truth_labels, sess = None):
        # TF tensor
        if not self.is_init_graph:
            self.x_adv_batch = self.graph(self.model.x_input, self.model.y_input)

        try:
            input_data = utils.DataProducer(dataX, ground_truth_labels, batch_size= self.batch_size, name = 'test')

            # load model parameters
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
            raise IOError("PGD attack: Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []

            for idx, X_batch, y_batch in input_data.next_batch():
                natural_loss = sess.run(self.model.y_xent, feed_dict={
                    self.model.x_input: X_batch,
                    self.model.y_input: y_batch,
                    self.model.is_training: False,
                })

                _x_adv_tmp = sess.run(self.x_adv_batch, feed_dict={
                    self.model.x_input: X_batch,
                    self.model.y_input: y_batch,
                    self.model.is_training: False
                })

                adv_loss = sess.run(self.model.y_xent, feed_dict={
                    self.model.x_input: _x_adv_tmp,
                    self.model.y_input: y_batch,
                    self.model.is_training: False,
                })
                replace_flag = (adv_loss < natural_loss)
                _x_adv_tmp[replace_flag] = X_batch[replace_flag]
                # accuracy
                curr_accuracy = utils.test_func(sess, self.model, _x_adv_tmp, y_batch, batch_size=50)
                if self.verbose:
                    print("\tPGD attack: mini-batch {}/{}, the current accuracy is {:.5} on a batch of samples".format(idx + 1, input_data.mini_batches, curr_accuracy))
                x_adv.append(_x_adv_tmp)
            x_adv_normalized = np.concatenate(x_adv)

            if self.verbose:
                accuracy = utils.test_func(sess, self.model, x_adv_normalized, ground_truth_labels, batch_size=50)
                print("The classification accuracy is {:.5} on adversarial feature vectors.".format(accuracy))

                perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_normalized - dataX) > 1e-6, axis=1))
                perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_normalized - dataX), axis=1))
                perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv_normalized - dataX), axis=1)))
                print("\t The average l0 norm of perturbations is {:5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:5}".format(perturbations_amount_l2))
            if sess_close_flag:
                sess.close()

        return dataX, x_adv_normalized, ground_truth_labels



