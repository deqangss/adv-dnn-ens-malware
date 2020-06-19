"""
Projected gradient decsent, adam solver
malware paper link: https://arxiv.org/abs/1812.08108
"""
from __future__ import print_function

import os
import sys
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *
from tools.adam_optimizer import TensorAdam, NadamOptimizer

DEFAULT_PARAM = {
    'learning_rate': 0.01,
    'max_iteration': 55,
    'batch_size': 50
}

class PGDAdam(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer=None, verbose = False, **kwargs):
        super(PGDAdam, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)
        self.lr = DEFAULT_PARAM['learning_rate']
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.optimizer = TensorAdam(lr = self.lr)
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

    def graph(self, x_input, y_input):
        def _cond(i, *_):
            return tf.less(i, self.iterations)

        init_state = self.optimizer.init_state([tf.zeros_like(x_input, dtype = tf.float32)])
        nest = tf.contrib.framework.nest

        # boundary
        scaled_max_extended = tf.maximum(  # broadcasting
            tf.multiply(self.scaled_clip_max,
                        tf.to_float(self.insertion_perm_array)) +  # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)), # may be useful to reset the lower bound
            x_input  # upper bound for positions no perturbations allowed
        )
        scaled_min_extended = tf.minimum(  # broadcasting
            tf.multiply(self.scaled_clip_min, tf.to_float(self.removal_perm_array)) +
            tf.multiply(self.scaled_clip_max, 1. - tf.to_float(self.removal_perm_array)),
            x_input
        )

        def _body(i, x_adv_tmp, flat_optim_state):
            curr_state = nest.pack_sequence_as(structure = init_state,
                                               flat_sequence = flat_optim_state)

            def _loss_fn_wrapper(x_):
                return -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model.get_logits(x_),
                                                                           labels=y_input)
            x_adv_tmp_list, new_optim_state = self.optimizer.minimize(_loss_fn_wrapper, [x_adv_tmp], curr_state)

            x_adv_tmp_clip = tf.clip_by_value(x_adv_tmp_list[0], clip_value_min= scaled_min_extended, clip_value_max= scaled_max_extended)

            return i + 1, x_adv_tmp_clip, nest.flatten(new_optim_state)

        flat_init_state = nest.flatten(init_state)
        _, adv_x_batch, _ = tf.while_loop(_cond,
                                          _body,
                                          (tf.zeros([]), x_input, flat_init_state),
                                          maximum_iterations=self.iterations,
                                          back_prop=False #
                                          )

        # map to discrete domain
        if self.normalizer is not None:
            x_adv = tf.rint(tf.divide(adv_x_batch - self.normalizer.min_, self.normalizer.scale_))
            # re-project back
            x_adv_normalized = tf.multiply(x_adv, self.normalizer.scale_) + self.normalizer.min_
        else:
            x_adv_normalized = tf.rint(adv_x_batch)

        return x_adv_normalized

    def parse(self, learning_rate = 0.01, max_iteration = 55, batch_size = 50, **kwargs):
        self.lr = learning_rate
        self.iterations = max_iteration
        self.batch_size = batch_size
        self.optimizer = TensorAdam(lr=self.lr)
        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def clip_adv(self, x):
        """
        clip x into feasible range
        :param x: 2-D np.ndarray
        :return: clipped x
        :rtype: 2D np.ndarray
        """
        scaled_min_expand = np.tile(self.scaled_clip_min, (x.shape[0], 1))
        scaled_max_expand = np.tile(self.scaled_clip_max, (x.shape[0], 1))  # exchange time with space

        scaled_max_expand[:, ~(self.insertion_perm_array == 1)] = x[:, ~(
                self.insertion_perm_array == 1)]  # update upper bound for each sample
        scaled_min_expand[:, ~(self.removal_perm_array == 1)] = x[:, ~(
                self.removal_perm_array == 1)]  # update lower bound
        return np.clip(x, a_min= scaled_min_expand, a_max=scaled_max_expand)


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
            raise IOError("PGD adam attack: Failed to load data and model parameters.")

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
                        "\tPGDAdam attack: mini-batch {}/{}, the current accuracy is {:.5} on a batch of samples".format(idx + 1,
                                                                                                                 input_data.mini_batches,
                                                                                           curr_accuracy))
                x_adv.append(_x_adv_tmp)
            x_adv_normalized = np.concatenate(x_adv)

            perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_normalized - dataX) > 1e-6, axis=1))
            perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_normalized - dataX), axis=1))
            perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv_normalized - dataX), axis=1)))

            if self.verbose:
                accuracy = utils.test_func(sess, self.model, x_adv_normalized, ground_truth_labels, batch_size=50)
                print("The classification accuracy is {:.5} on adversarial feature vectors.".format(accuracy))
                print("\t The average l0 norm of perturbations is {:5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:5}".format(perturbations_amount_l2))
            if sess_close_flag:
                sess.close()

        return dataX, x_adv_normalized, ground_truth_labels


