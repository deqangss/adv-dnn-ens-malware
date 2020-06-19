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
    'k': 1, # we'd better set k = 1 if input is binary representations
    'step_size': 1.,
    'max_iteration': 50,
    'batch_size': 50,
    'force_iteration': True # do not terminate the iteration even the miss-classification happens
}

class PGDl1(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer=None, verbose = False, **kwargs):
        super(PGDl1, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer,
                                      verbose)

        self.k = DEFAULT_PARAM['k']
        self.step_size = DEFAULT_PARAM['step_size']
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.is_init_graph = False
        self.parse(**kwargs)

    def parse(self, k =1, step_size = 1., max_iteration = 50, batch_size = 128, force_iteration = True, **kwargs):
        self.k = k
        self.step_size = step_size
        self.iterations = max_iteration
        self.batch_size = batch_size
        self.force_iteration = force_iteration

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def project_perturbations(self, x_input, perturbations):
        # boundary
        scaled_max_extended = tf.maximum( # broadcast
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) + # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)), # may be useful to reset the lower bound
            x_input # upper bound for positions no perturbations allowed
        )
        scaled_min_extended = tf.minimum(
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            x_input
        )
        return tf.clip_by_value(x_input + perturbations,
                                clip_value_min= scaled_min_extended,
                                clip_value_max= scaled_max_extended)

    def init_graph(self):
        # TF tensor
        self.x_adv_batch = self.graph(self.model.x_input, self.model.y_input)
        self.is_init_graph = True

    def graph(self, x_input, y_input):
        nb_classes = self.model.output_dim

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

        def _cond(i, cond_in, *_):
            return tf.logical_and(tf.less(i, self.iterations), cond_in)

        increase_domain = tf.reshape(x_input < scaled_max_extended,
                                     [-1, self.input_dim])
        decrease_domian = tf.reshape(x_input > scaled_min_extended,
                                     [-1, self.input_dim])

        search_domain = tf.cast(tf.logical_or(increase_domain, decrease_domian), tf.float32)

        def _body(i, cond_in, domain_in, x_adv_tmp):
            logits = self.model.get_logits(x_adv_tmp)
            loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                          labels=y_input)
            preds_onehot = tf.one_hot(tf.argmax(logits, axis=1), depth=nb_classes)

            grad = tf.gradients(loss, x_adv_tmp)[0]
            abs_grad = tf.reshape(tf.abs(grad), (-1, self.input_dim))
            threshold = 0.

            tmp_increase_domain = tf.reshape(tf.less(x_adv_tmp, scaled_max_extended),(-1, self.input_dim))
            tmp_increase_domain = tf.logical_and(tf.cast(domain_in, tf.bool), tmp_increase_domain)
            tmp_domain1 = tf.logical_and(tf.greater(grad, tf.to_float(threshold)),
                                         tmp_increase_domain)

            tmp_decrease_domain = tf.reshape(tf.greater(x_adv_tmp, scaled_min_extended), (-1, self.input_dim))
            tmp_decrease_domain = tf.logical_and(tf.cast(domain_in, tf.bool), tmp_decrease_domain)
            tmp_domain2 = tf.logical_and(tf.less(grad, tf.to_float(-1 * threshold)),
                                         tmp_decrease_domain)

            tmp_search_domain = tf.cast(tf.logical_or(tmp_domain1, tmp_domain2), tf.float32)
            score_mask = tf.cast(abs_grad > 0., tf.float32) * tmp_search_domain

            abs_grad_mask = abs_grad * score_mask
            top_k_v, top_k_idx = tf.nn.top_k(abs_grad_mask, k = self.k)
            changed_pos = tf.reduce_sum(tf.one_hot(top_k_idx, depth= self.input_dim), axis = 1)
            perturbations = tf.sign(grad) * changed_pos * tmp_search_domain
            # positions corresponds to the changed value will be neglected
            domain_in = domain_in - changed_pos
            mod_not_done = tf.equal(preds_onehot[:, 0], 0)

            if self.force_iteration:
                cond = (tf.reduce_sum(domain_in, axis=1) >= 1)
            else:
                cond = mod_not_done & (tf.reduce_sum(domain_in, axis=1) >= 1)

            cond_float = tf.reshape(tf.cast(cond, tf.float32), shape=[-1, 1])
            to_mod = perturbations * cond_float
            to_mod_reshape = tf.reshape(
                to_mod, shape=([-1] + x_adv_tmp.shape[1:].as_list()))

            x_out = x_adv_tmp + to_mod_reshape * self.step_size / tf.maximum(
                tf.reduce_sum(to_mod_reshape, -1, keepdims=True), self.k)

            x_out = tf.clip_by_value(x_out,
                                     clip_value_min=scaled_min_extended,
                                     clip_value_max=scaled_max_extended)

            cond_out = tf.reduce_any(cond)

            return i + 1, cond_out, domain_in, x_out

        _1, _2, _3, adv_x_batch = tf.while_loop(_cond,
                                                _body,
                                                (tf.zeros([]), True, search_domain, x_input),
                                                maximum_iterations=self.iterations,
                                                back_prop=False)

        # map to discrete domain
        if self.normalizer is not None:
            x_adv = tf.rint(tf.divide(adv_x_batch - self.normalizer.min_, self.normalizer.scale_))
            # re-project back
            x_adv_normalized = tf.multiply(x_adv, self.normalizer.scale_) + self.normalizer.min_
        else:
            x_adv_normalized = tf.rint(adv_x_batch)

        return x_adv_normalized

    def perturb(self, dataX, ground_truth_labels, sess=None):
        # TF tensor
        if not self.is_init_graph:
            self.x_adv_batch = self.graph(self.model.x_input, self.model.y_input)

        try:
            input_data = utils.DataProducer(dataX, ground_truth_labels, batch_size=self.batch_size, name='test')

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
        except Exception:
            raise IOError("l1 norm PGD attack: Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []

            for idx, X_batch, y_batch in input_data.next_batch():

                _x_adv_tmp = sess.run(self.x_adv_batch, feed_dict={
                    self.model.x_input: X_batch,
                    self.model.y_input: y_batch,
                    self.model.is_training: False
                })

                # accuracy
                curr_accuracy = utils.test_func(sess, self.model, _x_adv_tmp, y_batch, batch_size=50)
                if self.verbose:
                    print(
                        "\tPGD attack (l1 norm): mini-batch {}/{}, the current accuracy is {:.5} on a batch of samples".format(
                            idx + 1,
                            input_data.mini_batches,
                            curr_accuracy))
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