"""Fast gradient method for discrete features"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *


class FGSM(Attack):
    """FGSM attack"""

    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer,
                 verbose=False, **kwargs):
        super(FGSM, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer,
                                   verbose)
        self.epsilon = 1.  # default magnitude
        self.ord = 'l-infinity'
        self.batch_size = 50
        self.parse(**kwargs)

        self.perturbations = self.graph()

    def graph(self):
        preds = tf.reduce_max(self.model.y_proba, axis=1, keepdims=True)
        y = tf.to_float(tf.equal(preds, self.model.y_proba))
        y = tf.stop_gradient(y)
        y = y / tf.reduce_sum(y, axis=1, keepdims=True)

        # label_masking = tf.one_hot(self.model.y_input, 2, on_value=1., off_value=0., dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.model.logits)

        # gradient
        grad, = tf.gradients(loss, self.model.x_input)
        if self.ord == 'l-infinity':
            perturbations = utils.optimize_linear(grad, eps=self.epsilon)
        elif self.ord == 'l1':
            perturbations = utils.optimize_linear(grad, eps=self.epsilon, ord=1)
        elif self.ord == 'l2':
            perturbations = utils.optimize_linear(grad, eps=self.epsilon, ord=2)
        else:
            raise ValueError("Only 'l1', 'l2', 'l-infinity' are supported.")
        return perturbations

    def parse(self, epsilon=0.01, ord='l-infinity', batch_size=50, **kwargs):
        self.epsilon = epsilon
        self.ord = ord
        self.batch_size = batch_size
        if len(kwargs) > 0:
            print("unused hyper parameters.")

    def perturb(self, dataX, ground_truth_labels, sess=None):
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
        except IOError as ex:
            raise IOError("Failed to load data and model parameters.")

        with sess.as_default():
            x_adv = []

            for idx, X_batch, y_batch in input_data.next_batch():
                x_adv_tmp = np.copy(X_batch)

                loss = sess.run(self.model.y_xent, feed_dict={
                    self.model.x_input: X_batch,
                    self.model.y_input: y_batch,
                    self.model.is_training: False,
                })

                worst_loss = loss
                X_worst = np.copy(X_batch)

                # boundary
                _scaled_max_extended = np.maximum(
                    np.multiply(self.scaled_clip_max, self.insertion_perm_array) + # upper bound for positions allowing perturbations
                    np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array), # may be useful to reset the lower bound
                    X_batch # upper bound for positions no perturbations allowed
                )
                _scaled_min_extended = np.minimum(
                    np.multiply(self.scaled_clip_min, self.removal_perm_array) +
                    np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array),
                    X_batch
                )

                _perturbations = sess.run(self.perturbations, feed_dict={
                    self.model.x_input: x_adv_tmp
                })

                x_adv_tmp = x_adv_tmp + _perturbations

                # projection
                x_adv_tmp = np.clip(x_adv_tmp, a_max= _scaled_max_extended, a_min= _scaled_min_extended)

                _loss = sess.run(self.model.y_xent, feed_dict={
                    self.model.x_input: x_adv_tmp,
                    self.model.y_input: y_batch,
                    self.model.is_training: False,
                })

                change_indicator = (_loss >= worst_loss)
                X_worst[change_indicator] = x_adv_tmp[change_indicator]
                worst_loss[change_indicator] = _loss[change_indicator]

                # accuracy
                if self.verbose:
                    curr_accuracy = utils.test_func(sess, self.model, x_adv_tmp, y_batch, batch_size=50)
                    print("\t {} norm based FGSM attack: the current accuracy is {:.5} on a batch of samples".format(
                        self.ord, curr_accuracy))
                x_adv.append(X_worst)

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
                print("\t The average l0 norm of perturbations is {:5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:5}".format(perturbations_amount_l2))
            if sess_close_flag:
                sess.close()

        return dataX, x_adv_normalized, ground_truth_labels
