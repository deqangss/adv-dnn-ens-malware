"""GD-KED:https://arxiv.org/abs/1708.06131"""

import os
import sys
import warnings

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from attacker.methods.attack_method import *

DEFAULT_PARAM={
    'step_size': 1.,
    'max_iteration': 50,
    'negative_data_num': 100,
    'kernel_width': 10.,
    'lambda_factor': 100.,
    'distance_max': 20., # indicator of projection during the iteration
    'xi': 1e-6, # terminate the iteration
    'batch_size': 50
}

class GDKDE(Attack):
    def __init__(self, targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer = None,
                 verbose = False, **kwargs):
        super(GDKDE, self).__init__(targeted_model, input_dim, insertion_perm_array, removal_perm_array, normalizer, verbose)
        self.step_size = DEFAULT_PARAM['step_size']
        self.neg_data_num = DEFAULT_PARAM['negative_data_num']
        self.kernel_width = DEFAULT_PARAM['kernel_width']
        self.lambda_factor = DEFAULT_PARAM['lambda_factor']
        self.d_max = DEFAULT_PARAM['distance_max']
        self.xi = DEFAULT_PARAM['xi']
        self.batch_size = DEFAULT_PARAM['batch_size']
        self.iterations = DEFAULT_PARAM['max_iteration']
        self.neg_dataX = self._load_neg_data(self.neg_data_num)
        self.parse(**kwargs)

        # TF tensor
        self.clip_min_input = tf.placeholder(dtype=tf.float32, shape=(None, self.model.input_dim), name='CLIP_MIN')
        self.clip_max_input = tf.placeholder(dtype=tf.float32, shape=(None, self.model.input_dim), name='CLIP_MAX')
        self.batch_x_adv = self.graph()

    def _load_neg_data(self, max_num = 100):
        if not isinstance(max_num, int) and max_num < 0:
            raise TypeError("Input must be an positive interger.")
        if not os.path.exists(config.get("feature." + self.model.info.feature_type, "dataX")) \
            and not os.path.exists(config.get("feature." + self.model.info.feature_type, "datay")):
            raise Exception("No " + " feature." + self.model.info.feature_type + ".")
        else:
            dataX = utils.read_joblib(config.get("feature." + self.model.info.feature_type, "dataX"))
            datay = utils.read_joblib(config.get("feature." + self.model.info.feature_type, "datay"))

            negative_data_idx = (datay == 0)
            neg_dataX = dataX[negative_data_idx]
            if len(neg_dataX) == 0:
                raise ValueError("No negative data.")
            elif len(neg_dataX) < max_num:
                np.random.seed(0)
                return neg_dataX[np.random.choice(len(neg_dataX), max_num, replace=True)]
            else:
                np.random.seed(0)
                return neg_dataX[np.random.choice(len(neg_dataX), max_num, replace=False)]


    @staticmethod
    def _laplician_kernel(x1,x2,w):
        '''
        calculate the laplician kernel with x1 and x2
        :param x1: input one, shape is [batch_size, input_dim]
        :param x2: input two, shape is [number_of_negative_data, input_dim]
        :param w: the width of kernal
        :return: the laplician kernel value, shape is [batch_size, number_of_negative_data]
        '''
        return tf.exp(-1 * tf.reduce_sum(tf.abs(tf.expand_dims(x1, 1) - tf.expand_dims(x2, 0)),
                                         axis = 2) / w)

    def graph(self):
        self.negX = tf.constant(self.neg_dataX, dtype=tf.float32)

        def _cond(i, _):
            return tf.less(i, self.iterations)

        def _body(i, x_adv_tmp):
            def F(x_in):
                logits = self.model.get_logits(x_in)
                y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.model.y_input)
                kernel = self._laplician_kernel(x_in, self.negX, self.kernel_width)
                kde = tf.divide(self.lambda_factor, self.neg_data_num) * tf.reduce_sum(kernel, -1)
                return tf.reduce_mean(y_xent - kde)
            loss1 = F(x_adv_tmp)
            grad = tf.gradients(loss1, x_adv_tmp)[0]
            perturbations = utils.optimize_linear(grad, tf.to_float(self.step_size), ord = 2)
            x_adv_tmp = x_adv_tmp - perturbations

            x_adv_tmp = tf.clip_by_value(x_adv_tmp, clip_value_min=self.clip_min_input, clip_value_max=self.clip_max_input)

            i_out = tf.add(i, 1)
            return i_out,  x_adv_tmp

        _0,adv_x_batch = tf.while_loop(_cond,
                                       _body,
                                       [0, self.model.x_input])

        return adv_x_batch

    def parse(self, step_size = 1, max_iteration = 50, negative_data_num = 100,
              kernel_width = 10., lambda_factor = 100, distance_max = 20., xi = 1e-6,
              batch_size = 50, **kwargs):
        self.step_size = step_size
        self.iterations = max_iteration
        self.neg_data_num = negative_data_num
        self.kernel_width = kernel_width
        self.lambda_factor = lambda_factor
        self.d_max = distance_max
        self.xi = xi
        self.batch_size = batch_size

        if len(kwargs) > 0:
            warnings.warn("unused hyper parameters.")

    def perturb(self, dataX, ground_truth_labels, sess = None):
        self.target_labels = utils.get_other_classes_batch(self.model.output_dim, ground_truth_labels)

        try:
            save_dir = config.get('attack', 'gdkde')
            if not os.path.exists(save_dir):
                utils.mkdir(save_dir)

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
                for cls_idx in range(self.target_labels.shape[1]):
                    tar_label = tar_y_batch[:, cls_idx]
                    _batch_x_adv = sess.run(self.batch_x_adv, feed_dict={
                        self.model.x_input: X_batch,
                        self.model.y_input: tar_label,
                        self.model.is_training: False,
                        self.clip_max_input: _scaled_max_extended,
                        self.clip_min_input: _scaled_min_extended
                    })

                    # accuracy
                    if self.verbose:
                        curr_accuracy = utils.test_func(sess, self.model, _batch_x_adv, tar_label, batch_size=50)
                        print("\tGD-KDE attack: Mini batch at {}/{}, accuracy is {:.5} on classifying samples as targeted labels".format(
                              idx + 1, input_data.mini_batches, curr_accuracy))

                    _batch_x_adv = np.clip(_batch_x_adv,
                                           a_min=_scaled_min_extended,
                                           a_max=_scaled_max_extended
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
                accuracy = utils.test_func(sess, self.model, x_adv, np.concatenate([ground_truth_labels] * self.target_labels.shape[1]), batch_size=50)
                print("The classification accuracy is {:.5} on adversarial feature vectors.".format(accuracy))

                perturbations_amount_l0 = np.mean(np.sum(np.abs(x_adv_normalized - dataX) > 1e-6, axis=1))
                perturbations_amount_l1 = np.mean(np.sum(np.abs(x_adv_normalized - dataX), axis=1))
                perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(x_adv_normalized - dataX), axis=1)))
                print("\t The average l0 norm of perturbations is {:5}".format(perturbations_amount_l0))
                print("\t The average l1 norm of perturbations is {:5}".format(perturbations_amount_l1))
                print("\t The average l2 norm of perturbations is {:5}".format(perturbations_amount_l2))

            if sess_close_flag:
                sess.close()
        rtn_dataX = np.concatenate([dataX] * self.target_labels.shape[1])
        rtn_label = np.concatenate([ground_truth_labels] * self.target_labels.shape[1])
        return rtn_dataX, x_adv_normalized, rtn_label
