import os
import sys
import warnings

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *
from attacker.methods.mimicry import Mimicry

DEFAULT_PARAM = {
    'repetition': 1,
    'random_seed': 0
}

class PointWise(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer = None, verbose = False, **kwargs):
        super(PointWise, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)

        self.repetition = DEFAULT_PARAM['repetition']
        self.random_seed = DEFAULT_PARAM['random_seed']

        self.parse(**kwargs)

        set_param = {
            'trial': self.repetition,
            'random_seed': self.random_seed,
            'is_reducing_pert': False
        }
        self.init_method = Mimicry(targeted_model,
                                   input_dim,
                                   insertion_perm_array,
                                   removal_perm_array,
                                   normalizer,
                                   False,
                                   **set_param
                                   )

    def parse(self, repetition = 1, random_seed = 0, **kwargs):
        self.repetition = repetition
        self.random_seed = random_seed

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def get_start_point(self, x, y, sess = None):
        _1, adv_x, _2 = \
            self.init_method.perturb(x, y, sess)
        return adv_x

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
            x_adv_init = self.get_start_point(dataX, ground_truth_labels, sess)
            assert x_adv_init.shape == dataX.shape

            x_adv = np.copy(x_adv_init)
            for idx in range(len(x_adv_init)):
                feat_vector = dataX[idx: idx + 1]
                adv_feat_vector = x_adv_init[idx: idx + 1]

                shape = feat_vector.shape
                N = feat_vector.size

                orig_feats = feat_vector.reshape(-1)
                adv_feats = adv_feat_vector.reshape(-1)

                np.random.seed(self.random_seed)
                while True:
                    # draw random shuffling of all indices
                    indices = list(range(N))
                    np.random.shuffle(indices)

                    for index in indices:
                        # start trial
                        old_value = adv_feats[index]
                        new_value = orig_feats[index]
                        if old_value == new_value:
                            continue
                        adv_feats[index] = new_value

                        # check if still adversarial
                        _acc = sess.run(self.model.accuracy, feed_dict={
                            self.model.x_input: adv_feats.reshape(shape),
                            self.model.y_input: ground_truth_labels[idx: idx + 1],
                            self.model.is_training: False
                        })

                        if _acc <= 0.:
                            x_adv[idx: idx+ 1] = adv_feats.reshape(shape)
                            break

                        # if not, undo change
                        adv_feats[index] = old_value
                    else:
                        print("No features can be flipped by adversary successfully")
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

