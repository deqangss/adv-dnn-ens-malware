"""
@inproceedings{grosse2017adversarial,
  title={Adversarial examples for malware detection},
  author={Grosse, Kathrin and Papernot, Nicolas and Manoharan, Praveen and Backes, Michael and McDaniel, Patrick},
  booktitle={European Symposium on Research in Computer Security},
  pages={62--79},
  year={2017},
  organization={Springer}
}
"""

import os
import sys

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM = {
    'max_iteration': 50, # maximum iterations
    'batch_size': 50,
    'force_iteration': True # do not terminate the iteration even the miss-classification happens
}

class GrosseAttack(Attack):
    """Grosse et al. attack"""
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer = None, verbose = False, **kwargs):
        super(GrosseAttack, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.parse(**kwargs)

        # TF tensor
        self.clip_max_input = tf.placeholder(dtype=tf.float32, shape=(None, self.model.input_dim), name='CLIP_MAX')

    def graph(self, x_tensor):
        nb_classes = self.model.output_dim
        nb_features = self.model.input_dim
        # Compute our initial search domain. We optimize the initial search domain
        # by removing all features that are already at their maximum values (if
        # increasing input features).
        search_domain = tf.reshape(tf.cast(x_tensor < self.clip_max_input, tf.float32),
                                   [-1, nb_features])

        # y_in_init = tf.reshape(tf.one_hot(self.targed_y_input, depth=nb_classes), [-1, nb_classes])

        # Loop variables
        # x_in: the tensor that holds the latest adversarial outputs that are in
        #       progress.
        # y_in: the tensor for target labels
        # domain_in: the tensor that holds the latest search domain
        # cond_in: the boolean tensor to show if more iteration is needed for
        #          generating adversarial samples
        def _cond(x_in, domain_in, i, cond_in):
            # Repeat the loop until we have achieved misclassification or
            # reaches the maximum iterations
            return tf.logical_and(tf.less(i, self.iterations), cond_in)

        def _body(x_in, domain_in, i, cond_in):
            logits = self.model.get_logits(x_in)
            preds = tf.nn.softmax(logits)
            preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

            # get corresponding derivatives
            derivatives, = tf.gradients(tf.reduce_mean(preds[:, 0]), x_in) # malicious samples are labeled as '1'

            # Remove the already-used input features from the search space
            # Subtract 2 times the maximum value from those value so that
            # they won't be picked later
            increase_coef = 2 * tf.cast(tf.equal(domain_in, 0), tf.float32)

            derivatives -= increase_coef \
                          * tf.reduce_max(tf.abs(derivatives), axis=1, keepdims=True)
            derivatives = tf.reshape(derivatives, shape=[-1, nb_features])

            # Create a mask to only keep features that match conditions
            scores_mask = derivatives > 0

            # Extract the best malware feature
            scores = tf.cast(scores_mask, tf.float32) * derivatives
            best = tf.argmax(scores, axis=1)
            p1_one_hot = tf.one_hot(best, depth=nb_features)

            # Check if more modification is needed for each sample
            # mod_not_done = tf.equal(tf.reduce_sum(y_in * preds_onehot, axis=1), 0)
            mod_not_done = tf.equal(preds_onehot[:,0], 0)

            if self.force_iteration:
                cond = (tf.reduce_sum(domain_in * tf.cast(scores_mask, tf.float32), axis=1) >= 1)
            else:
                cond = mod_not_done & (tf.reduce_sum(domain_in * tf.cast(scores_mask, tf.float32), axis=1) >= 1)

            cond_float = tf.reshape(tf.cast(cond, tf.float32), shape=[-1, 1])
            to_mod = p1_one_hot * cond_float

            domain_out = domain_in - to_mod

            to_mod_reshape = tf.reshape(
                to_mod, shape=([-1] + x_in.shape[1:].as_list()))
            x_out = tf.minimum(x_in + to_mod_reshape, self.scaled_clip_max)

            # Increase the iterator, and check if all miss-classifications are done
            i_out = tf.add(i, 1)
            cond_out = tf.reduce_any(cond)

            return x_out, domain_out, i_out, cond_out

        x_adv_batch, _2, _3, _4 = tf.while_loop(
            _cond,
            _body,
            [x_tensor, search_domain, 0, True]
        )

        return x_adv_batch

    def parse(self, iterations=50, batch_size=50, force_iteration = True, **kwargs):
        self.iterations = iterations
        self.batch_size = batch_size
        self.force_iteration = force_iteration
        if len(kwargs) > 0:
            print("unused hyper parameters.")

    def perturb(self, dataX, ground_truth_labels, sess = None):
        """
        generate the adversarial feature vectors
        :param dataX: feature vector
        :param ground_truth_labels: label
        :param kwargs: parameters
        :return: pristine feature vectors, adversarial feature vectors, labels
        """
        self.batch_x_adv = self.graph(self.model.x_input)
        try:
            input_data = utils.DataProducer(dataX, ground_truth_labels, batch_size= self.batch_size, name = 'test')

            # load model parameters
            sess_close_flag = False
            if sess is None:
                cur_checkpoint = tf.train.latest_checkpoint(self.model.save_dir)
                print(self.model.save_dir)
                config_gpu = tf.ConfigProto(log_device_placement=True)
                config_gpu.gpu_options.allow_growth = True
                sess = tf.Session(config=config_gpu)
                saver = tf.train.Saver()
                saver.restore(sess, cur_checkpoint)
                sess_close_flag = True
        except IOError as ex:
            raise IOError("Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []

            for idx, X_batch, y_batch in input_data.next_batch():
                # boundary
                _scaled_max_extended = np.maximum(
                    np.multiply(self.scaled_clip_max,
                                self.insertion_perm_array) +  # upper bound for positions allowing perturbations
                    np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
                    # may be useful to reset the lower bound
                    X_batch  # upper bound for positions no perturbations allowed
                )

                _batch_x_adv = sess.run(self.batch_x_adv,feed_dict={
                    self.model.x_input: X_batch,
                    self.model.is_training: False,
                    self.clip_max_input: _scaled_max_extended
                })

                # accuracy
                if self.verbose:
                    curr_accuracy = utils.test_func(sess, self.model, _batch_x_adv, y_batch, batch_size=50)
                    print("\t Batch {}/{}, the current accuracy is {:.5} on adversarial examples".format(
                        idx,
                        input_data.mini_batches,
                        curr_accuracy)
                    )
                x_adv.append(_batch_x_adv)

            x_adv = np.concatenate(x_adv)
            if self.normalizer is not None:
                x_adv = np.rint(normalize_inverse(x_adv, self.normalizer))
                # check again
                x_adv_normalized = normalize_transform(x_adv, self.normalizer)
            else:
                x_adv_normalized = np.rint(x_adv)

            if self.verbose:
                accuracy = utils.test_func(sess, self.model, x_adv_normalized, ground_truth_labels, batch_size=50)
                print("The classification accuracy is {:.5} on adversarial feature vectors.".format(accuracy))

                perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_normalized - dataX) > 1e-6, axis=1))
                perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_normalized - dataX), axis=1))
                perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv_normalized - dataX), axis=1)))
                print("\t The average l0 norm of perturbations is {:.5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:.5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:.5}".format(perturbations_amount_l2))
            if sess_close_flag:
                sess.close()

        return dataX, x_adv_normalized, ground_truth_labels






