import os
import sys
import warnings

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM = {
    'epsilon': 1000,
    'max_eta': 1.,
    'repetition': 1,
    'random_seed': 0
}

class SaltAndPepper(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer = None, verbose = False, **kwargs):
        super(SaltAndPepper, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)

        self.epsilon = DEFAULT_PARAM['epsilon']
        self.max_eta = DEFAULT_PARAM['max_eta']
        self.repetition = DEFAULT_PARAM['repetition']
        self.random_seed = DEFAULT_PARAM['random_seed']

        self.parse(**kwargs)

    def parse(self, max_eta = 1., repetition = 1, random_seed = 0, **kwargs):
        # self.epsilon = epsilon
        self.max_eta = max_eta
        self.repetition = repetition
        self.random_seed = random_seed

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def perturb(self, dataX, ground_truth_labels, sess = None):
        try:
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
            x_adv = np.copy(dataX)
            for idx in range(len(x_adv)):
                feat_vector = dataX[idx: idx + 1]
                # boundary
                scaled_max_extended = np.maximum(
                    np.multiply(self.scaled_clip_max, self.insertion_perm_array.astype(np.float32)) + # upper bound for positions allowing perturbations
                    np.multiply(self.scaled_clip_min, 1. - self.insertion_perm_array.astype(np.float32)), # may be useful to reset the lower bound
                    feat_vector # upper bound for positions no perturbations allowed
                )
                scaled_min_extended = np.minimum(
                    np.multiply(self.scaled_clip_min, self.removal_perm_array.astype(np.float32)) +
                    np.multiply(self.scaled_clip_max, 1. - self.removal_perm_array.astype(np.float32)),
                    feat_vector
                )
                r = scaled_max_extended - scaled_min_extended
                shape = feat_vector.shape
                np.random.seed(self.random_seed)
                max_eta = self.max_eta
                epsilons = min(self.epsilon, shape[1])

                for _ in range(self.repetition):
                    for eta in np.linspace(0, max_eta, num=epsilons)[1:]:
                        p = eta
                        uni_noises = np.random.uniform(0, 1, size=shape)
                        salt = (uni_noises >= 1. - p / 2).astype(dataX.dtype) * r
                        pepper = -(uni_noises < p / 2).astype(dataX.dtype) * r

                        perturbed_x = feat_vector + salt + pepper
                        perturbed_x = np.clip(perturbed_x, scaled_min_extended, scaled_max_extended)

                        _acc = sess.run(self.model.accuracy, feed_dict={
                            self.model.x_input: perturbed_x,
                            self.model.y_input: ground_truth_labels[idx: idx + 1],
                            self.model.is_training: False
                        })

                        if _acc <= 0.:
                            print("success!", idx)
                            x_adv[idx: idx + 1] = perturbed_x
                            max_eta = min(1., eta * 1.5)
                            break

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

        # dump to disk
        return dataX, x_adv_normalized, ground_truth_labels

