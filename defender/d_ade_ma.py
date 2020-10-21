"""
The script is for diversified Adversarial deep ensemble incorporating attacks a mixture of attacks
we consider the ''max'' attack, including pgd-l1, pgd-l2, pgd-linf, pgd-adam
"""

import os
import sys
from datetime import datetime
from timeit import default_timer
import random

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score

proj_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(proj_dir)

from config import config
from tools import utils
from learner.feature_extractor import get_droid_feature, FeatureMapping, feature_type_scope_dict
from learner.basic_DNN import BasicDNNModel, DNN_HP, INFO
from attacker.feature_reverser import DrebinFeatureReverse
from defender.at import MAXIMIZER_PARAM_DICT, MAXIMIZER_METHOD_DICT
from defender.at import AdversarialTrainingDNN
from defender.at import ADV_TRAIN_HP
from defender.at_ma import MAX_ADV_TRAIN_HP


ADV_ENS_INFO = {
    'dataset_dir': config.get('dataset', 'dataset_root'),
    'feature_tp': list(feature_type_scope_dict.keys())[0], #'drebin',
    'feature_mapping_type': config.get('feature.drebin', 'feature_mp'),
    'learning_algorithm': 'ADV_ENS_BASE'
}

ADV_ENS_HP = {
    'lambda_2': 1. # the balanced factor for regularizing the base models
}


class DAdversarialDeepEnsembleMax(BasicDNNModel):
    def __init__(self,
                 info_dict = None,
                 hyper_params = None,
                 reuse=False,
                 is_saving = True,
                 init_graph = True,
                 mode = 'train',
                 name = 'DADV_NN_ENSEMBLE_MAX'):
        """
        hardened deep ensemble incorporated with ''max'' attack and a diversifying method
        @param info_dict: None,
        @param hyper_params: hyper parameters,
        @param reuse: reuse the variables or not
        @param is_saving: option for saving weights
        @param init_graph: initialize graph
        @param mode: enable a mode for run the model, 'train' or 'test'
        @param name: model name
        """
        self.is_saving = is_saving
        self.init_graph = init_graph
        self.mode = mode
        if info_dict is None:
            ADV_ENS_INFO.update(INFO)
            info_dict = ADV_ENS_INFO
        self.clf_info = utils.ParamWrapper(info_dict)
        if hyper_params is None:
            ADV_ENS_HP.update(MAX_ADV_TRAIN_HP)
            ADV_ENS_HP.update(DNN_HP)
            hyper_params = ADV_ENS_HP
        self.hp_params = utils.ParamWrapper(hyper_params)
        self.model_name = name

        self.base_model_method = [AdversarialTrainingDNN] * len(MAXIMIZER_METHOD_DICT)
        self.base_model_method.append(BasicDNNModel)
        self.base_model_count = len(self.base_model_method)
        assert self.base_model_count > 1, 'one base model at least'

        # initialization
        if self.clf_info.feature_tp == feature_type_scope_dict.keys()[0]:
            self.normalizer = utils.read_pickle(config.get('feature.' + self.clf_info.feature_tp, 'normalizer'))
        else:
            raise ValueError("Feature type is incompatible.")
        input_dim = len(utils.read_pickle(config.get('feature.' + self.clf_info.feature_tp, 'vocabulary')))
        self.eta = self.hp_params.eta
        feature_reverser = DrebinFeatureReverse()
        allow_insert_array, allow_removal_array = feature_reverser.get_mod_array()

        # build attack graph
        maximizer_name_list = self.hp_params.maximizer_name_list
        self.inner_maximizers = []
        self.trial_list = []
        for maximizer_name in maximizer_name_list:
            maximizer_method = MAXIMIZER_METHOD_DICT[maximizer_name]
            maximizer_param = MAXIMIZER_PARAM_DICT[maximizer_name]
            inner_maximizer = maximizer_method(self,
                                               input_dim,
                                               allow_insert_array,
                                               allow_removal_array,
                                               self.normalizer,
                                               verbose=False,
                                               **maximizer_param
                                               )

            self.inner_maximizers.append(inner_maximizer)
            self.trial_list.append(self.hp_params.trials_dict[maximizer_name])

        # record the number of malware examples in a training batch
        self.batch_size_mal = tf.Variable(0, dtype=tf.int64, trainable=False)

        super(DAdversarialDeepEnsembleMax, self).__init__(info_dict,
                                                          hyper_params,
                                                          reuse = reuse,
                                                          is_saving=self.is_saving,
                                                          init_graph= self.init_graph,
                                                          mode = self.mode,
                                                          name = name)

    def model_graph(self, reuse = False):
        """Continue to conduct initialization"""
        self.base_model_names = ["{}_{}".format(self.model_name, k) for k in range(self.base_model_count)]
        self.base_models = []
        for k in range(self.base_model_count - 1):
            ADV_TRAIN_HP['maximizer_name'] = self.hp_params.maximizer_name_list[k]
            adv_train_hp = ADV_TRAIN_HP.update(DNN_HP)
            self.base_models.append(
                self.base_model_method[k](hyper_params = adv_train_hp,
                                          name=self.base_model_names[k],
                                          is_saving=False,
                                          init_graph=False,
                                          reuse=False,
                                          mode=self.mode
                                          )
            )
            self.base_model_names[k] = self.base_models[k].model_name  # accommodate the changed model name
        self.base_models.append(
            self.base_model_method[-1](name=self.base_model_names[-1],
                                       is_saving=False,
                                       init_graph=False,
                                       reuse=False,
                                       mode=self.mode)
        )

        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='ENS_X')
        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None, ], name='ENS_Y')
        self.is_training = tf.placeholder(tf.bool, name="ENS_TRAIN")

        w_shape = [self.base_model_count * self.hp_params.output_dim, ]
        init_v = tf.constant_initializer(1. / w_shape[0], dtype=tf.float32)
        with tf.variable_scope(self.model_name, reuse=reuse):
            self.W = tf.get_variable(name="COMB_W", shape=w_shape, initializer=init_v, trainable=False) # update manually

        # useful information for base models
        for m in range(self.base_model_count):
            self.base_models[m].is_training = self.is_training

        self.sub_logits_list, \
        self.sub_y_list, \
        self.logits, \
        self.y_tensor = \
            self.forward(self.x_input, self.y_input, reuse)
        self.model_inference()

    def forward(self, x_tensor, y_tensor, reuse = False):
        base_model_logits = []
        base_model_y_tensor = []
        for k in range(self.base_model_count - 1):
            self.base_models[k].mode = self.mode
            _logits, _y_tensor = \
                self.base_models[k].forward(x_tensor, y_tensor, reuse)

            base_model_logits.append(_logits[self.hp_params.batch_size:])
            base_model_y_tensor.append(_y_tensor[self.hp_params.batch_size:])
        # basic dnn models
        _basic_logits, _basic_y = self.base_models[-1].forward(x_tensor, y_tensor, reuse)

        base_model_logits.append(_basic_logits)
        base_model_y_tensor.append(_basic_y)

        # nn ensemble
        def ens_graph(x_input, is_training=True):
            logits_list = []
            for k in range(self.base_model_count):
                _1, _2, _logits = self.base_models[k].nn(x_input,
                                                         is_training = is_training,
                                                         name = self.base_model_names[k],
                                                         reuse = True)
                logits_list.append(_logits)
            logits_cat = tf.concat(logits_list, axis=1)
            logits_weighted = logits_cat * self.W
            logits_ens = tf.reduce_sum(
                tf.reshape(logits_weighted, [-1, self.base_model_count, self.hp_params.output_dim]), axis=1)
            return logits_ens

        self.ens_nn = ens_graph
        _ = self.ens_nn(x_tensor, is_training= False)

        if self.mode == 'train':
            adv_x, rtn_x, rtn_y = self.gen_max_adv_mal_graph(x_tensor, y_tensor)

            self.adv_x = tf.cond(self.is_training,
                                 lambda: tf.concat([x_tensor, adv_x], axis=0),
                                 lambda: x_tensor)
            self.adv_y = tf.cond(self.is_training,
                                 lambda: tf.concat([y_tensor, rtn_y], axis=0),
                                 lambda: y_tensor)

        elif self.mode == 'test':
            self.adv_x = x_tensor
            self.adv_y = y_tensor
        else:
            pass

        ens_logits = ens_graph(self.adv_x, is_training=self.is_training)
        ens_y_tensor = self.adv_y

        return base_model_logits, base_model_y_tensor, ens_logits, ens_y_tensor

    def model_inference(self):

        # loss function of base models
        self.cross_entropy_base_model = 0
        for sub_i in range(self.base_model_count):
            cross_entropy_aug = tf.losses.sparse_softmax_cross_entropy(
                labels= self.sub_y_list[sub_i],
                logits = self.sub_logits_list[sub_i]
            )
            self.cross_entropy_base_model = self.cross_entropy_base_model + cross_entropy_aug

        # loss definition
        cross_entropy_orig = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor[:self.hp_params.batch_size],
            logits=self.logits[:self.hp_params.batch_size]
        )
        cross_entropy_aug = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor[self.hp_params.batch_size:],
            logits=self.logits[self.hp_params.batch_size:]
        )
        self.cross_entropy_ens = self.hp_params.lambda_ * cross_entropy_aug + (
                    1. - self.hp_params.lambda_) * cross_entropy_orig
        self.cross_entropy = self.cross_entropy_ens + self.hp_params.lambda_2 * self.cross_entropy_base_model


        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_tensor,
            logits=self.logits
        )

        # prediction
        self.y_proba = tf.nn.softmax(self.logits)
        self.y_pred = tf.argmax(self.logits, axis=1)

        # some information
        self.accuracy = tf.reduce_mean(
            tf.to_float(tf.equal(self.y_pred, self.y_tensor))
        )

    def gen_max_adv_mal_graph(self, x_tensor, y_tensor):
        """
        graph for waging attacks
        :param x_tensor: batch of input data
        :param y_tensor: batch of ground truths
        :return: the perturbed examples
        """
        mal_indices = tf.where(y_tensor)  # '1' denotes the malicious sample
        mal_x_tensor = tf.gather_nd(x_tensor, mal_indices)
        mal_y_tensor = tf.gather_nd(y_tensor, mal_indices)

        ben_indices = tf.where(tf.equal(y_tensor, 0))
        ben_x_tensor = tf.gather_nd(x_tensor, ben_indices)
        ben_y_tensor = tf.gather_nd(y_tensor, ben_indices)
        with tf.control_dependencies([mal_x_tensor, mal_y_tensor]):
            self.batch_size_mal = tf.assign(self.batch_size_mal, tf.reduce_sum(y_tensor))
            x_shape = mal_x_tensor.get_shape().as_list()

        def filter(adv_mal_x):
            """
            filter the perturbed samples that dnn can classify correctly
            @param adv_mal_x: perturbed input
            """
            logits = self.ens_nn(adv_mal_x, False)
            pred_y_adv = tf.argmax(logits, axis=1)
            incorrect_case = tf.reshape(tf.to_float(tf.logical_not(
                tf.equal(pred_y_adv, mal_y_tensor))), (-1, 1))
            return tf.stop_gradient((adv_mal_x - mal_x_tensor) * incorrect_case + mal_x_tensor)

        def  _get_random_noises(x_batch):
            eta = tf.random_uniform([1, ], 0, self.hp_params.eta)
            init_perturbations = tf.random_uniform(tf.shape(x_batch),
                                                   minval=-1.,
                                                   maxval=1.,
                                                   dtype=tf.float32)
            init_perturbations = tf.multiply(
                tf.sign(init_perturbations),
                tf.to_float(
                    tf.abs(init_perturbations) >= 1. - eta),
            )
            return init_perturbations

        def _loss_fn(x, y):
            logits = self.ens_nn(x, False)
            return -1 * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=y)

        def _get_single_perturb(inner_maximizer, mal_x, mal_y, trials = 0):
            if trials == 0:
                return tf.stop_gradient(inner_maximizer.graph(mal_x, mal_y))
            elif trials > 0:
                mal_x_batch_ext = tf.tile(mal_x, [trials, 1])
                mal_y_batch_ext = tf.tile(mal_y, [trials, ])

                init_perturbations = _get_random_noises(mal_x_batch_ext)

                init_x_batch_ext = inner_maximizer.project_perturbations(
                    mal_x_batch_ext,
                    init_perturbations
                )

                adv_mal_batch_ext = tf.stop_gradient(
                    inner_maximizer.graph(
                    init_x_batch_ext,
                    mal_y_batch_ext
                ))

                adv_mal_losses = tf.stop_gradient(_loss_fn(adv_mal_batch_ext, mal_y_batch_ext))

                adv_mal_x_pool = tf.reshape(adv_mal_batch_ext, [trials, -1, x_shape[1]])
                adv_losses = tf.reshape(adv_mal_losses, [trials, -1])

                idx_selected = tf.stack([tf.argmin(adv_losses),
                                         tf.range(self.batch_size_mal, dtype=tf.int64)], axis=1)

                return tf.gather_nd(adv_mal_x_pool, idx_selected)
            else:
                raise ValueError

        adv_mal_list = []
        N = len(self.inner_maximizers)
        for index in range(N):
            adv_mal_list.append(
                _get_single_perturb(self.inner_maximizers[index], mal_x_tensor, mal_y_tensor, trials=self.trial_list[index])
            )

        adv_mal_instances_final = tf.concat(adv_mal_list, axis=0)

        y_adv_mal = tf.tile(mal_y_tensor, [N,])
        losses = _loss_fn(adv_mal_instances_final, y_adv_mal)

        adv_mal_instances_final = tf.reshape(adv_mal_instances_final, [N, -1, x_shape[1]])
        losses = tf.reshape(losses, [N, -1])
        max_adv_indices = tf.stack([tf.argmin(losses), tf.range(self.batch_size_mal, dtype=tf.int64)],
                                   axis=1)
        max_adv = filter(
            tf.gather_nd(adv_mal_instances_final, max_adv_indices))

        pben_x_tensor = tf.clip_by_value(ben_x_tensor + _get_random_noises(ben_x_tensor),
                                         clip_value_max=1.,
                                         clip_value_min=0.)

        rtn_adv_batch = tf.concat([max_adv, pben_x_tensor], axis= 0)
        rtn_prist_batch = tf.concat([mal_x_tensor, ben_x_tensor], axis=0)
        rtn_y_tensor = tf.concat([mal_y_tensor, ben_y_tensor], axis = 0)
        return rtn_adv_batch, rtn_prist_batch, rtn_y_tensor

    def update_w(self, optimizer, vars):
        opt = optimizer.minimize(loss = self.cross_entropy_ens, var_list = vars)
        dim = tf.reduce_prod(tf.shape(self.W))
        with tf.control_dependencies([opt]):
            U = tf.reshape(self.W, [-1])
            sorted_U = tf.nn.top_k(U, k = dim, sorted = True)[0]
            cumsum_U = tf.cumsum(sorted_U)
            rho = tf.where(
                tf.greater(sorted_U * tf.to_float(tf.range(1, dim + 1)), cumsum_U - 1)
            )[-1, -1]

            theta = (cumsum_U[rho] - 1.) / tf.to_float(rho)
            W_clipped = tf.clip_by_value(self.W - theta, clip_value_min= 0., clip_value_max=np.infty)
            self.W = tf.assign(self.W, W_clipped)
            return self.W


    def train(self, trainX = None, trainy = None, valX = None, valy = None):
        """train deep ensemble"""
        if trainX is None or trainy is None or valX is None or valy is None:
            trainX, valX, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'dataX'))
            trainy, valy, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'datay'))

        train_input = utils.DataProducer(trainX, trainy,self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)
        val_input = utils.DataProducer(valX, valy, self.hp_params.batch_size*20, name='val')

        # perturb the malware representations
        val_mal_indicator = (valy == 1.)
        val_malX = valX[val_mal_indicator]
        val_maly = valy[val_mal_indicator]

        # attack initialization
        for inner_maximizer in self.inner_maximizers:
            inner_maximizer.init_graph()

        # record information
        global_train_step = tf.train.get_or_create_global_step()
        saver = tf.train.Saver()
        tf.summary.scalar('accuracy_adv_train', self.accuracy)
        tf.summary.scalar('loss_adv_train', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()

        # optimizers
        var_w = [var for var in tf.global_variables() if 'COMB_W' in var.name]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,
                                                                                      global_step=global_train_step)
            optimizer_w = tf.train.AdamOptimizer(self.hp_params.learning_rate)
            updated_w = self.update_w(optimizer_w, var_w)

        tf_cfg = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        tf_cfg.gpu_options.allow_growth = True
        tf_cfg.gpu_options.per_process_gpu_memory_fraction = 1.
        sess = tf.Session(config=tf_cfg)

        with sess.as_default():
            summary_writer = tf.summary.FileWriter(self.save_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            training_time = 0.0
            train_input.reset_cursor()
            output_steps = 500
            best_avg_score_val = 0.
            for step_idx, X_batch, y_batch in train_input.next_batch():

                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx + 1) % output_steps == 0:
                    print('Step {}/{}:{}'.format(step_idx + 1, train_input.steps, datetime.now()))
                    val_res_list = []
                    _adv_malX_list = []
                    val_input.reset_cursor()
                    for _, valX_batch, valy_batch in val_input.next_batch():
                        val_res_batch = sess.run([self.accuracy, self.y_pred],
                                                 feed_dict={self.x_input: valX_batch,
                                                            self.y_input: valy_batch,
                                                            self.is_training: False})
                        val_res_list.append(val_res_batch)

                        val_mal_indicator_batch = (valy_batch == 1.)
                        val_malX_batch = valX_batch[val_mal_indicator_batch]
                        val_maly_batch = valy_batch[val_mal_indicator_batch]
                        _adv_valX_batch = self.perturbations_of_max_attack(val_malX_batch, val_maly_batch, sess)
                        _adv_malX_list.append(_adv_valX_batch)
                    val_res = np.array(val_res_list, dtype=object)
                    _acc = np.mean(val_res[:, 0])
                    _pred_y = np.concatenate(val_res[:, 1])
                    from sklearn.metrics import f1_score
                    _f1_score = f1_score(valy, _pred_y[:valy.shape[0]])

                    _adv_valX = np.vstack(_adv_malX_list)[:val_maly.shape[0]]

                    _adv_acc_val = sess.run(self.accuracy, feed_dict={self.x_input: _adv_valX,
                                                                      self.y_input: val_maly,
                                                                      self.is_training: False})
                    _avg_score = (_f1_score + _adv_acc_val) / 2.
                    print('    validation accuracy {:.5}%'.format(_acc * 100))
                    print('    validation f1 score {:.5}%'.format(_f1_score * 100))
                    print('    validation accuracy on adversarial malware samples {:.5}%'.format(_adv_acc_val * 100))

                    if step_idx != 0:
                        print('    {} samples per second'.format(
                            output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))

                    if best_avg_score_val <= _avg_score:
                        best_avg_score_val = _avg_score
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        saver.save(sess, os.path.join(self.save_dir, 'checkpoint'),
                                   global_step=global_train_step)

                start = default_timer()
                sess.run(optimizer, feed_dict=train_dict)
                sess.run(updated_w, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()

    def get_logits(self, dataX_tf):
        """
        return logits for given data
        @param dataX 2D tensor
        """
        return self.ens_nn(dataX_tf, is_training=False)

    def perturbations_of_max_attack(self, val_malX, val_maly, sess):
        perturbed_mal_list = []
        N = len(self.inner_maximizers)
        S = val_malX.shape[0]
        D = val_malX.shape[1]
        for inner_maximizer in self.inner_maximizers:
            _1, adv_x, _2 = inner_maximizer.perturb(val_malX, val_maly, sess)
            perturbed_mal_list.append(adv_x)
        perturbed_mal_instances = np.concatenate(perturbed_mal_list, axis=0)
        with sess.as_default():
            _y_xent = sess.run(
                self.y_xent,
                feed_dict= {
                    self.x_input: perturbed_mal_instances,
                    self.y_input: np.tile(val_maly, [N,]),
                    self.is_training: False
                }
            )

        perturbed_mal_instances = np.reshape(perturbed_mal_instances, [N, -1, D]).transpose([1, 0, 2])
        _losses = np.reshape(_y_xent, [N, -1]).transpose([1, 0])
        return perturbed_mal_instances[range(S), np.argmax(_losses, axis=-1), :]


def _main():
    rsm_dnn = DAdversarialDeepEnsembleMax()
    rsm_dnn.train()
    rsm_dnn.mode = 'test'
    rsm_dnn.test_rpst()


if __name__ == "__main__":
    _main()

