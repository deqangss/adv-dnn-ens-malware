import os
import sys
import warnings

import tensorflow as tf
import numpy as np
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *
from attacker.methods.pgd import PGD
from attacker.methods.pgdl1 import PGDl1
from attacker.methods.pgd_adam import PGDAdam
from attacker.methods.gdkde import GDKDE

DEFAULT_PARAM = {
    'random_seed' : 0,
    'iteration': 5,
    'call_sp' : True,
    'use_fast_version': False,
    'varepsilon': 1e-9,
    'attack_names': ['pgdl1', 'pgdl2', 'pgdlinf', 'pgd_adam']
}

attack_params_dict = {
    'pgdl1' : {'k': 1, 'step_size': 1., 'max_iteration': 100, 'batch_size': 50, 'force_iteration' : False},
    'pgdl2' : {'step_size': 1., 'ord': 'l2', 'max_iteration': 1000, 'batch_size': 50},
    'pgdlinf' : {'step_size': 0.01, 'ord': 'l-infinity', 'max_iteration': 1000, 'batch_size': 50},
    'pgd_adam' : {'learning_rate': 0.01, 'max_iteration': 1000, 'batch_size': 50},
    'gdkde' : {'step_size': 1.,'max_iteration': 1000,'negative_data_num': 5000,'kernel_width': 20.,
               'lambda_factor': 1.,'distance_max': 20., 'xi': 0., 'batch_size': 50},
}

attack_methods_dict = defaultdict()

class MAX(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer = None, verbose = False, **kwargs):
        super(MAX, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)

        self.iteration = DEFAULT_PARAM['iteration']
        self.attack_names = DEFAULT_PARAM['attack_names']
        self.call_sp = DEFAULT_PARAM['call_sp']
        self.use_fast_version = DEFAULT_PARAM['use_fast_version']
        self.varepsilon = DEFAULT_PARAM['varepsilon']

        self.attack_seq_selected = defaultdict(list)
        self.attack_mthds_dict = attack_methods_dict

        self.parse(**kwargs)

        # attacks
        self.attack_mthds_dict['pgdl1'] = \
            PGDl1(targeted_model,
                  input_dim,
                  insertion_perm_array,
                  removal_perm_array,
                  normalizer,
                  False,
                  **attack_params_dict['pgdl1']
                  )
        self.attack_mthds_dict['pgdl2'] = \
            PGD(targeted_model,
                input_dim,
                insertion_perm_array,
                removal_perm_array,
                normalizer,
                False,
                **attack_params_dict['pgdl2']
                )
        self.attack_mthds_dict['pgdlinf'] = \
            PGD(targeted_model,
                input_dim,
                insertion_perm_array,
                removal_perm_array,
                normalizer,
                False,
                **attack_params_dict['pgdlinf']
                )
        self.attack_mthds_dict['pgd_adam'] =\
            PGDAdam(targeted_model,
                    input_dim,
                    insertion_perm_array,
                    removal_perm_array,
                    normalizer,
                    False,
                    **attack_params_dict['pgd_adam']
                    )
        self.attack_mthds_dict['gdkde'] = \
            GDKDE(targeted_model,
                  input_dim,
                  insertion_perm_array,
                  removal_perm_array,
                  normalizer,
                  False,
                  **attack_params_dict['gdkde']
                  )

    def parse(self, random_seed = 0, iteration = 1, call_saltandpepper = True, use_fast_version = False, varepsilon = 1e-9, attack_names = None, **kwargs):
        self.random_seed = random_seed
        self.iteration = iteration
        self.call_sp = call_saltandpepper
        self.use_fast_version = use_fast_version
        self.varepsilon = varepsilon
        self.attack_names = attack_names

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def get_start_point(self, x, y, sess = None):
        x_perterb = x.copy()
        min_prob = np.ones(y.shape)
        for iter_idx in range(self.iteration):
            adv_x_list = []
            spec_names = []
            for a in self.attack_names:
                if a in self.attack_mthds_dict.keys():
                    _1, adv_x, _2 = self.attack_mthds_dict[a].perturb(x_perterb, y, sess)
                    adv_x_list.append(adv_x)
                    spec_names.append(a)
                else:
                    warnings.warn("No attack:{}".format(a))

            attack_num = len(adv_x_list)
            all_adv_x = np.concatenate(adv_x_list)

            _y_prob = sess.run(self.model.y_proba,
                               feed_dict={self.model.x_input: all_adv_x,
                                          self.model.is_training: False
                                          })

            all_adv_trans = np.transpose(np.reshape(all_adv_x, (attack_num, x.shape[0], x.shape[1])),
                                         (1,0,2))
            _y_prob_trans = np.transpose(np.reshape(_y_prob, (attack_num, x.shape[0], _y_prob.shape[1])),
                                         (1,0,2))

            for i in range(len(x)):
                gt = int(y[i])
                min_v = np.min(_y_prob_trans[i, :, gt])
                if min_v < min_prob[i]:
                    min_prob[i] = min_v
                    min_ind = int(np.argmin(_y_prob_trans[i, :, gt]))
                    x_perterb[i, :] = all_adv_trans[i, min_ind, :]
                    self.attack_seq_selected[i].append(spec_names[min_ind])
                else:
                    self.attack_seq_selected[i].append(' ')
                    continue

        return x_perterb

    def get_start_point_cvg(self, x, y, sess=None):
        x_perterb = x.copy()
        loop_indicator = np.ones((x.shape[0],), np.bool)
        _last_prob = sess.run(self.model.y_proba,
                              feed_dict={self.model.x_input: x,
                                         self.model.is_training: False
                                         })
        for iter_idx in range(self.iteration):
            adv_x_list = []
            spec_names = []
            x_loop = x_perterb[loop_indicator]
            for a in self.attack_names:
                if a in self.attack_mthds_dict.keys():
                    _1, adv_x, _2 = self.attack_mthds_dict[a].perturb(x_loop, y, sess)
                    adv_x_list.append(adv_x)
                    spec_names.append(a)
                else:
                    warnings.warn("No attack:{}".format(a))

            attack_num = len(adv_x_list)
            all_adv_x = np.concatenate(adv_x_list)

            _y_prob = sess.run(self.model.y_proba,
                               feed_dict={self.model.x_input: all_adv_x,
                                          self.model.is_training: False
                                          })

            all_adv_trans = np.transpose(np.reshape(all_adv_x, (attack_num, x_loop.shape[0], x_loop.shape[1])),
                                         (1, 0, 2))
            _y_prob_trans = np.transpose(np.reshape(_y_prob, (attack_num, x_loop.shape[0], _y_prob.shape[1])),
                                         (1, 0, 2))

            l_i = 0
            for g_i, e in enumerate(loop_indicator):
                if not e:
                    continue

                gt = int(y[g_i])
                min_v = np.min(_y_prob_trans[l_i, :, gt])
                min_ind = int(np.argmin(_y_prob_trans[l_i, :, gt]))
                x_perterb[g_i, :] = all_adv_trans[l_i, min_ind, :]
                if np.abs(min_v - _last_prob[g_i, gt]) <= self.varepsilon:
                    loop_indicator[g_i] = False
                else:
                    _last_prob[g_i, gt] = min_v
                l_i += 1

            if not np.any(loop_indicator):
                break
        return x_perterb

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
            if self.use_fast_version:
                x_adv_init = self.get_start_point_cvg(dataX, ground_truth_labels, sess)
            else:
                x_adv_init = self.get_start_point(dataX, ground_truth_labels, sess)

            assert x_adv_init.shape == dataX.shape
            x_adv = np.copy(x_adv_init)

            if self.call_sp:
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
        utils.dump_json(self.attack_seq_selected, './metainfo.json')
        # dump to disk
        return dataX, x_adv_normalized, ground_truth_labels

