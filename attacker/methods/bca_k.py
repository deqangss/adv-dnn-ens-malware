"""
@inproceedings{al2018adversarial,
  title={Adversarial deep learning for robust detection of binary encoded malware},
  author={Al-Dujaili, Abdullah and Huang, Alex and Hemberg, Erik and O'Reilly, Una-May},
  booktitle={2018 IEEE Security and Privacy Workshops (SPW)},
  pages={76--82},
  year={2018},
  organization={IEEE}
}
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM = {
    'max_iteration': 50, # parameter k
    'batch_size': 50,
    'force_iteration': True # do not terminate the iteration even the miss-classification happens
}

class BCA_K(Attack):
    # Multi-step bit coordinate ascent
    # We modify bck_k a little bit to accommodate the non-binary feature

    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer = None, verbose = False, **kwargs):
        super(BCA_K, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)
        self.force_iteration = DEFAULT_PARAM['force_iteration']
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.is_init_graph = False
        self.parse(**kwargs)

        # TF tensor
        self.clip_max_input = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name='CLIP_MAX')

    def graph(self, x_tensor, y_tensor = None):
        nb_classes = self.model.output_dim
        nb_features = self.model.input_dim
        scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) +
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)),
            x_tensor
        )
        # Compute our initial search domain. We optimize the initial search domain
        # by removing all features that are already at their maximum values (if
        # increasing input features).
        # search_domain = tf.reshape(tf.cast(x_tensor < self.clip_max_input, tf.float32),
        #                            [-1, nb_features])
        search_domain = tf.reshape(tf.cast(x_tensor < scaled_max_extended, tf.float32),
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

            # get corresponding grads
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits= logits,
                    labels= y_tensor
                )
            )
            grads, = tf.gradients(loss, x_in)

            # Remove the already-used input features from the search space
            # Subtract 2 times the maximum value from those value so that
            # they won't be picked later
            increase_coef = 2 * tf.cast(tf.equal(domain_in, 0), tf.float32)

            grads -= increase_coef \
                          * tf.reduce_max(tf.abs(grads), axis=1, keepdims=True)
            grads = tf.reshape(grads, shape=[-1, nb_features])

            # Create a mask to only keep features that match conditions
            scores_mask = tf.greater(grads, tf.to_float(0.))

            # Extract the best malware feature
            scores = tf.cast(scores_mask, tf.float32) * grads
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

    def project_perturbations(self, x_input, perturbations):
        # boundary
        scaled_max_extended = tf.maximum(
            tf.multiply(self.scaled_clip_max, tf.to_float(self.insertion_perm_array)) + # upper bound for positions allowing perturbations
            tf.multiply(self.scaled_clip_min, 1. - tf.to_float(self.insertion_perm_array)), # may be useful to reset the lower bound
            x_input # upper bound for positions no perturbations allowed
        ) # broadcast leveraged
        return tf.clip_by_value(x_input + perturbations,
                                clip_value_min= x_input,
                                clip_value_max= scaled_max_extended)

    def init_graph(self):
        # TF tensor
        self.batch_x_adv = self.graph(self.model.x_input, self.model.y_input)
        self.is_init_graph = True

    def perturb(self, dataX, ground_truth_labels, sess = None):
        """
        generate the adversarial feature vectors for dataX
        :param dataX: feature vectors
        :param ground_truth_labels: label
        :param kwargs: parameters
        :return: tuple of pristine feature vectors, adversarial feature vectors, labels
        """
        if not self.is_init_graph:
            self.batch_x_adv = self.graph(self.model.x_input, self.model.y_input)

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
            raise IOError("Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []

            for idx, X_batch, y_batch in input_data.next_batch():
                # update upper bound for each sample
                scaled_max_expand = np.tile(self.scaled_clip_max, (X_batch.shape[0], 1))  # exchange time with space
                scaled_max_expand[:, ~(self.insertion_perm_array == 1)] = X_batch[:, ~(
                            self.insertion_perm_array == 1)]
                _batch_x_adv = sess.run(self.batch_x_adv, feed_dict={
                    self.model.x_input: X_batch,
                    self.model.y_input: y_batch,
                    self.model.is_training: False
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






