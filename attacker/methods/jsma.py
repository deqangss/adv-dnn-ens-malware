"""
The Jacobian-based Saliency Map Method
paper link: https://arxiv.org/pdf/1511.07528.pdf
malware paper link: ieeexplore.ieee.org/document/8782574
"""

import os
import sys

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM = {
    'theta': 1,
    'max_iteration': 50,
    'batch_size': 50,
    'force_iteration': True # do not terminate the itartion even the miss-classification happens
}

class JSMA(Attack):
    """JAMA attack"""
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer = None, verbose = False, **kwargs):
        super(JSMA, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)
        self.theta = DEFAULT_PARAM['theta']
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.increase = bool(self.theta > 0)
        self.parse(**kwargs)

        # TF tensor
        self.targed_y_input = tf.placeholder(dtype = tf.int64, shape=(None,), name='TARGET_Y')
        self.clip_min_input = tf.placeholder(dtype=tf.float32, shape=(None, self.model.input_dim), name='CLIP_MIN')
        self.clip_max_input = tf.placeholder(dtype=tf.float32, shape=(None, self.model.input_dim), name='CLIP_MAX')
        self.batch_x_adv = self.graph()

    def graph(self):
        """the code snippet is adapted from clevshans:https://github.com/tensorflow/cleverhans"""
        nb_classes = self.model.output_dim
        nb_features = self.model.input_dim
        # Compute our initial search domain. We optimize the initial search domain
        # by removing all features that are already at their maximum values (if
        # increasing input features---otherwise, at their minimum value).
        if self.increase:
            search_domain = tf.reshape(tf.cast(self.model.x_input < self.clip_max_input, tf.float32),
                                       [-1, nb_features])
        else:
            search_domain = tf.reshape(tf.cast(self.model.x_input > self.clip_min_input, tf.float32),
                                       [-1, nb_features])

        y_in_init = tf.reshape(tf.one_hot(self.targed_y_input, depth=nb_classes), [-1, nb_classes])

        # Loop variables
        # x_in: the tensor that holds the latest adversarial outputs that are in
        #       progress.
        # y_in: the tensor for target labels
        # domain_in: the tensor that holds the latest search domain
        # cond_in: the boolean tensor to show if more iteration is needed for
        #          generating adversarial samples
        def _cond(x_in, y_in, domain_in, i, cond_in):
            # Repeat the loop until we have achieved misclassification or
            # reaches the maximum iterations
            return tf.logical_and(tf.less(i, self.iterations), cond_in)

        def _body(x_in, y_in, domain_in, i, cond_in):
            logits = self.model.get_logits(x_in)
            preds = tf.nn.softmax(logits)
            preds_onehot = tf.one_hot(tf.argmax(preds, axis=1), depth=nb_classes)

            # create the Jacobian graph
            list_derivatives = []
            for class_ind in range(nb_classes):
                derivatives = tf.gradients(preds[:, class_ind], x_in)
                list_derivatives.append(derivatives[0])
            grads = tf.reshape(
                tf.stack(list_derivatives), shape=[nb_classes, -1, nb_features])

            # Compute the Jacobian components
            # To help with the computation later, reshape the target_class
            # and other_class to [nb_classes, -1, 1].
            # The last dimention is added to allow broadcasting later.
            target_class = tf.reshape(
                tf.transpose(y_in, perm=[1, 0]), shape=[nb_classes, -1, 1])
            other_classes = tf.cast(tf.not_equal(target_class, 1), tf.float32)

            grads_target = tf.reduce_sum(grads * target_class, axis=0)
            grads_other = tf.reduce_sum(grads * other_classes, axis=0)

            # Remove the already-used input features from the search space
            # Subtract 2 times the maximum value from those value so that
            # they won't be picked later
            increase_coef = (4 * int(self.increase) - 2) \
                            * tf.cast(tf.equal(domain_in, 0), tf.float32)

            target_tmp = grads_target
            target_tmp -= increase_coef \
                          * tf.reduce_max(tf.abs(grads_target), axis=1, keepdims=True)
            # target_sum = tf.reshape(target_tmp, shape=[-1, nb_features, 1]) \
            #             + tf.reshape(target_tmp, shape=[-1, 1, nb_features]) # Save RAM for high dimensional malware features. This will induce some shortcoming, see: https://arxiv.org/abs/1608.04644
            target_sum = tf.reshape(target_tmp, shape=[-1, nb_features])

            other_tmp = grads_other
            other_tmp += increase_coef \
                         * tf.reduce_max(tf.abs(grads_other), axis=1, keepdims=True)
            # other_sum = tf.reshape(other_tmp, shape=[-1, nb_features, 1]) \
            #            + tf.reshape(other_tmp, shape=[-1, 1, nb_features])
            other_sum = tf.reshape(other_tmp, shape=[-1, nb_features])

            # Create a mask to only keep features that match conditions
            if self.increase:
                scores_mask = ((target_sum > 0) & (other_sum < 0))
            else:
                scores_mask = ((target_sum < 0) & (other_sum > 0))

            # Create a 2D numpy array of scores for each pair of candidate features
            # scores = tf.cast(scores_mask, tf.float32) \
            #         * (-target_sum * other_sum) * zero_diagonal
            ## Extract the best two pixels
            # best = tf.argmax(
            #        tf.reshape(scores, shape=[-1, nb_features * nb_features]), axis=1)

            # p1 = tf.mod(best, nb_features)
            # p2 = tf.floordiv(best, nb_features)
            # p1_one_hot = tf.one_hot(p1, depth=nb_features)
            # p2_one_hot = tf.one_hot(p2, depth=nb_features)

            # Extract the best malware feature
            scores = tf.cast(scores_mask, tf.float32) * (-target_sum * other_sum)
            best = tf.argmax(scores, axis=1)
            p1_one_hot = tf.one_hot(best, depth=nb_features)

            # Check if more modification is needed for each sample
            mod_not_done = tf.equal(tf.reduce_sum(y_in * preds_onehot, axis=1), 0)

            if self.force_iteration:
                cond = (tf.reduce_sum(search_domain, axis=1) >= 1)
            else:
                cond = mod_not_done & (tf.reduce_sum(domain_in, axis=1) >= 1)

            cond_float = tf.reshape(tf.cast(cond, tf.float32), shape=[-1, 1])
            to_mod = p1_one_hot * cond_float
            domain_out = domain_in - to_mod

            to_mod_reshape = tf.reshape(
                to_mod, shape=([-1] + x_in.shape[1:].as_list()))

            if self.increase:
                x_out = tf.minimum(x_in + to_mod_reshape * self.theta,
                                   self.scaled_clip_max)
            else:
                x_out = tf.maximum(x_in + to_mod_reshape * self.theta,
                                   self.scaled_clip_min)

            # Increase the iterator, and check if all miss-classifications are done
            i_out = tf.add(i, 1)
            cond_out = tf.reduce_any(cond)

            return x_out, y_in, domain_out, i_out, cond_out

        x_adv_batch, _1, _2, _3, _4 = tf.while_loop(
            _cond,
            _body,
            [self.model.x_input, y_in_init, search_domain, 0, True]
        )

        return x_adv_batch

    def parse(self, theta= 1, iterations=50, batch_size=50, force_iteration = True, **kwargs):
        self.theta = theta
        self.increase = bool(self.theta > 0)
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
        self.target_labels = utils.get_other_classes_batch(self.model.output_dim, ground_truth_labels)
        try:
            input_data = utils.DataProducer(dataX, self.target_labels, batch_size= self.batch_size, name = 'test')

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
            raise IOError("Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []

            for idx, X_batch, tar_y_batch in input_data.next_batch():
                # boundary
                _scaled_max_extended = np.maximum(
                    np.multiply(self.scaled_clip_max,
                                self.insertion_perm_array) +  # upper bound for positions allowing perturbations
                    np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array),
                    # may be useful to reset the lower bound
                    X_batch  # upper bound for positions no perturbations allowed
                )
                _scaled_min_extended = np.minimum(
                    np.multiply(self.scaled_clip_min, self.removal_perm_array) +
                    np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
                    X_batch
                )

                _batch_x_adv_container = []
                for cls_idx in range(self.target_labels.shape[1]): #if there are multiple classes, we tile the dataX
                    tar_label = tar_y_batch[:,cls_idx]
                    _batch_x_adv = sess.run(self.batch_x_adv,feed_dict={
                        self.model.x_input: X_batch,
                        self.targed_y_input: tar_label,
                        self.model.is_training: False,
                        self.clip_min_input: _scaled_min_extended,
                        self.clip_max_input: _scaled_max_extended
                    })

                    # accuracy
                    if self.verbose:
                        curr_accuracy = utils.test_func(sess, self.model, _batch_x_adv, tar_label, batch_size=50)
                        print("\t Batch {}/{}, the current accuracy is {:.5} on classifying samples as targeted labels".format(
                            idx,
                            input_data.mini_batches,
                            curr_accuracy)
                        )
                    _batch_x_adv_container.append(_batch_x_adv)
                _batch_x_adv_container = np.stack(_batch_x_adv_container)
                x_adv.append(_batch_x_adv_container)

            x_adv = np.reshape(np.hstack(x_adv), (-1, self.model.input_dim))
            if self.normalizer is not None:
                x_adv = np.rint(normalize_inverse(x_adv, self.normalizer))
                # check again
                x_adv_normalized = normalize_transform(x_adv, self.normalizer)
            else:
                x_adv_normalized = np.rint(x_adv)

            if self.verbose:
                accuracy = utils.test_func(sess, self.model, x_adv_normalized, np.concatenate([ground_truth_labels] * self.target_labels.shape[1]), batch_size=50)
                print("The classification accuracy is {:.5} on adversarial feature vectors.".format(accuracy))

                perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_normalized - dataX) > 1e-6, axis=1))
                perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_normalized - dataX), axis=1))
                perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv_normalized - dataX), axis=1)))
                print("\t The average l0 norm of perturbations is {:.5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:.5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:.5}".format(perturbations_amount_l2))
            if sess_close_flag:
                sess.close()

        rtn_dataX = np.concatenate([dataX] * self.target_labels.shape[1])
        rtn_label = np.concatenate([ground_truth_labels] * self.target_labels.shape[1])
        return rtn_dataX, x_adv_normalized, rtn_label






