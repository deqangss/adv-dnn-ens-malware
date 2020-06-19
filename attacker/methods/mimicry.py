import os
import sys
import warnings

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM = {
    'trial': 10,
    'random_seed': 0,
    'is_reducing_pert': False
}


class Mimicry(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer=None,
                 verbose=False, **kwargs):
        super(Mimicry, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer,
                                      verbose)

        self.trial = DEFAULT_PARAM['trial']
        self.random_seed = DEFAULT_PARAM['random_seed']
        self.is_reducing_pert = DEFAULT_PARAM['is_reducing_pert']
        self.small_data, self.small_label = self._load_data(500)

        self.parse(**kwargs)

    def _load_data(self, max_num=500):
        if not isinstance(max_num, int) and max_num < 0:
            raise TypeError("Input must be an positive interger.")
        if not os.path.exists(config.get("feature." + self.model.info.feature_type, "dataX")) \
                and not os.path.exists(config.get("feature." + self.model.info.feature_type, "datay")):
            raise Exception("No " + " feature." + self.model.info.feature_type + ".")
        else:
            dataX, _, _ = utils.read_joblib(config.get("feature." + self.model.info.feature_type, "dataX"))
            datay, _, _ = utils.read_joblib(config.get("feature." + self.model.info.feature_type, "datay"))

        small_dataX = []
        small_datay = []
        for l in range(self.model.output_dim):
            _label_idx_arr = (datay == l)
            max_num = max_num if np.sum(_label_idx_arr) > max_num else np.sum(_label_idx_arr)
            small_dataX.append(dataX[_label_idx_arr][:max_num])
            small_datay.append(datay[_label_idx_arr][:max_num])
        return np.concatenate(small_dataX), np.concatenate(small_datay)

    def get_trials(self, label):
        '''
        get the trial samples for one sample
        :param label: ground truth labels
        :return: trails for corresponding label
        '''
        if isinstance(label, int):
            cur_label = (self.small_label == label)

            other_data = self.small_data[~cur_label]
            number_of_other_data = other_data.shape[0]
            replace = False
            if self.trial > number_of_other_data:
                replace = True
            np.random.seed(self.random_seed)
            return other_data[np.random.choice(number_of_other_data, self.trial, replace=replace)]
        else:
            raise TypeError("Input error, only Int type support.")

    def parse(self, trial=10, random_seed=0, is_reducing_pert=False, **kwargs):
        self.trial = trial
        self.random_seed = random_seed
        self.is_reducing_pert = is_reducing_pert

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def perturb(self, dataX, ground_truth_labels, sess=None):

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

            x_adv = []

            for idx in range(len(dataX)):
                feat_vector = dataX[idx: idx + 1]
                label = ground_truth_labels[idx: idx + 1]
                labels_ = np.tile(label, self.trial)

                trials_vector = self.get_trials(int(label[0]))

                # modify
                modified_feat_vector = np.tile(feat_vector, (self.trial, 1))
                # recompute boundary
                scaled_min_expand = np.tile(self.scaled_clip_min, (self.trial, 1))
                scaled_max_expand = np.tile(self.scaled_clip_max, (self.trial, 1))  # exchange time with space

                scaled_max_expand[:, ~(self.insertion_perm_array == 1)] = modified_feat_vector[:, ~(
                        self.insertion_perm_array == 1)]  # update upper bound for each sample
                scaled_min_expand[:, ~(self.removal_perm_array == 1)] = modified_feat_vector[:, ~(
                        self.removal_perm_array == 1)]  # update lower bound

                modified_feat_vector[:] = trials_vector
                x_adv_tmp = np.clip(modified_feat_vector, a_min=scaled_min_expand, a_max=scaled_max_expand)

                _y_pred, _y_prob = sess.run([self.model.y_pred, self.model.y_proba],
                                            feed_dict={self.model.x_input: x_adv_tmp,
                                                       self.model.y_input: labels_,
                                                       self.model.is_training: False
                                                       }
                                            )

                attack_success_indicator = (_y_pred != label)
                modif_scale = np.sum(np.abs(x_adv_tmp - feat_vector), axis=-1)
                if np.any(attack_success_indicator) and self.is_reducing_pert:
                    modif_scale[~attack_success_indicator] = np.max(modif_scale) + 1.
                    idx_min_scale = np.argsort(modif_scale)[0]
                else:
                    idx_min_scale = np.argmin(_y_prob[:, int(label[0])])

                x_adv.append(np.reshape(x_adv_tmp[idx_min_scale], (1, -1)))

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

        # dump to disk
        return dataX, x_adv_normalized, ground_truth_labels
