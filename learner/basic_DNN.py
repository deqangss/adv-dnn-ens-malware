"""
DNN for android malware detection, using the variant of drebin features.
"""
import os
import sys
from datetime import datetime
from timeit import default_timer
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from learner.classification import *
from tools import utils
from learner.feature_extractor import get_droid_feature, FeatureMapping, DREBIN_FEAT_INFO, feature_type_scope_dict
from config import config, logging

logger = logging.getLogger("learning.basic_dnn")

INFO = {
    'dataset_dir': config.get('dataset', 'dataset_root'),
    'feature_type': list(feature_type_scope_dict.keys())[0],  # 'drebin',
    'feature_mapping_type': config.get('feature.drebin', 'feature_mp'),
    'use_interdependent_features': False,
    'learning_algorithm': 'DNN'
}

DNN_HP = {
    'random_seed': 23456,
    'hidden_units': [160, 160],  # DNN has two hidden layers with each having 160 neurons
    'output_dim': 2,  # malicious vs. benign
    'n_epochs': 150,
    'batch_size': 128,
    'learning_rate': 0.001,
    'optimizer': 'adam'  # others are not supported
}


def graph(x_input,
          hidden_neurons=[160, 160],
          output_dim=2,
          is_training=True,
          name="BASIC_DNN",
          reuse=False):
    '''
    define the architecture of nerual network
    :param x_input: data 2D tensor
    :param hidden_neurons: neurons for hidden layers
    :param output_dim: number of classes
    :param is_training: option for training
    :param name: graph name
    :param reuse: option for reusing variables
    :return: graph
    '''
    with tf.variable_scope("{}".format(name), reuse=reuse):
        # dense layer #1 ~ #len(hidden_layers)
        # the neuron unit is layer_neurons[0]
        # input Tensor shape: [batch_size, input_dim]
        # output Tensor shape:[batch_size, hidden_neurons[0]]
        dense1 = tf.layers.dense(inputs=x_input, units=hidden_neurons[0], activation=tf.nn.relu, name="DENSE1")

        # dense layer #2
        # the neuron unit is layer_neurons[1]
        # input Tensor shape: [batch_size, hidden_neurons[0]]
        # output Tensor shape:[batch_size, hidden_neurons[1]]
        dense2 = tf.layers.dense(inputs=dense1, units=hidden_neurons[1], activation=tf.nn.relu, name="DENSE2")

        # bottlenect
        # dense layer #3
        # the neron unit output_dim
        # input Tensor shape: [batch_size, hidden_neurons[1]]
        # output Tensor shape: [batch_size, output_dim]
        dense3 = tf.layers.dense(inputs=dense2, units=output_dim, activation=None, name="DENSE3")

        return dense1, dense2, dense3


def tester(sess, testX, testy, model, required_info='label'):
    """
    model testing on test dataset
    :param sess: tf.Session
    :param testX: data for testing, type: 2-D float np.ndarry
    :param testy: corresponding ground truth labels, type: 1-D int np.ndarray
    :param model: trained model
    :params required_info: 'label' or 'prob'
    :return: predicted label or probability
    """
    test_input = utils.DataProducer(testX, testy, batch_size=20, name='test')
    if isinstance(testy, np.ndarray):
        test_num = testy.shape[0]
    else:
        test_num = len(testy)

    # check
    if required_info in 'label':
        info = model.y_pred
    elif required_info in 'proba':
        info = model.y_proba
    else:
        raise ValueError("'label' or 'proba' is supported.")

    with sess.as_default():
        pred = []
        for _, x, y in test_input.next_batch():
            test_dict = {
                model.x_input: x,
                model.y_input: y,
                model.is_training: False
            }
            _y_pred = sess.run(info, feed_dict=test_dict)
            pred.append(_y_pred)

        return np.concatenate(pred)[:test_num]


def macro_f1_socre(tpr, pos_count, tnr, neg_count):
    """
    macro f1 score for multiple classification
    """
    fnr = 1. - tpr
    fpr = 1. - tnr
    tp = np.rint(pos_count * tpr)
    tn = np.rint(neg_count * tnr)
    fp = np.rint(neg_count * fpr)
    fn = np.rint(pos_count * fnr)

    def binary_f1(tp, fp, fn):
        prec_pos = tp / (tp + fp)
        recall_pos = tp / (tp + fn)
        return 2 * (prec_pos * recall_pos) / (prec_pos + recall_pos)

    f1_pos = binary_f1(tp, fp, fn)
    f1_neg = binary_f1(tn, fn, fp)
    macro_f1 = (f1_pos + f1_neg) / 2.
    return macro_f1


class BasicDNNModel(Classifier):
    def __init__(self,
                 info_dict=None,
                 hyper_params=None,
                 reuse=False,
                 is_saving=True,
                 init_graph=True,
                 mode='train',
                 name='BASIC_DNN'):
        """
        build basic dnn model
        @param info_dict: None,
        @param hyper_params: hyper parameters,
        @param reuse: reuse the variables or not
        @param is_saving: option for saving weights
        @param init_graph: initialize graph
        @param mode: enable a mode for run the model, 'train' or 'test'
        @param name: model name
        """
        super(BasicDNNModel, self).__init__()
        # model setup
        self.is_saving = is_saving
        self.init_graph = init_graph
        try:
            assert mode == 'train' or mode == 'test'
        except:
            raise AssertionError("'train' or 'test' mode, not others.")

        self.mode = mode
        if info_dict is not None:
            self.info_dict = info_dict
        else:
            self.info_dict = INFO
        self.info = utils.ParamWrapper(self.info_dict)
        if hyper_params is not None:
            self.hp_params_dict = hyper_params
        else:
            self.hp_params_dict = DNN_HP
        self.hp_params = utils.ParamWrapper(self.hp_params_dict)
        self.model_name = name

        if self.is_saving:
            self.save_dir = config.get('experiments', name.lower())

        # feature extraction
        self.feature_tp = self.info.feature_type # drebin
        self.feature_mp = self.info.feature_mapping_type # binary
        self.dataset_dir = self.info.dataset_dir

        self.mal_dir = os.path.join(self.dataset_dir, config.get('dataset', 'malware_dir_name'))
        self.ben_dir = os.path.join(self.dataset_dir, config.get('dataset', 'benware_dir_name'))

        if not (os.path.exists(config.get('feature.' + self.feature_tp, 'dataX')) and
                os.path.exists(config.get('feature.' + self.feature_tp, 'datay')) and
                os.path.exists(config.get('feature.' + self.feature_tp, 'vocabulary')) and
                os.path.exists(config.get('feature.' + self.feature_tp, 'normalizer')) and
                os.path.exists(config.get('dataset', 'name_list'))):
            self._data_preprocess()

        # obtain some hyper-parameters
        self.input_dim = len(
            utils.read_pickle(config.get('feature.' + self.feature_tp, 'vocabulary')))
        self.hidden_layers = self.hp_params.hidden_units
        self.output_dim = self.hp_params.output_dim
        tf.set_random_seed(self.hp_params.random_seed)
        if self.init_graph:
            self.model_graph(reuse=reuse)

    def _data_preprocess(self):
        """
        feature extraction
        """
        if (not os.path.exists(self.ben_dir)) and (not os.path.exists(self.mal_dir)):
            logger.error("directory '{}' or '{}' has no APK data.".format(self.ben_dir, self.mal_dir))
            return
        try:
            label_dict = self.get_label_dict()

            data_root_dir = config.get("dataset", "dataset_root")
            feat_save_dir = os.path.join(data_root_dir, "apk_data")
            get_droid_feature(self.ben_dir, feat_save_dir, feature_type=self.feature_tp)
            get_droid_feature(self.mal_dir, feat_save_dir, feature_type=self.feature_tp)

            feature_mapping = FeatureMapping(feat_save_dir, feature_type=self.feature_tp)
            naive_features, name_list = feature_mapping.load_features()

            if len(naive_features) == 0:
                logger.error("No features extracted.")
                return

            # remove S6: used permissions, this type of features depend on feature 'S7' APIs and feature 'S2' permission
            if not self.info.use_interdependent_features:
                naive_features = feature_mapping.remove_interdependent_featrues(naive_features)

            gt_label = np.array([label_dict[os.path.splitext(name.strip())[0]] \
                                 for name in name_list])

            vocab, vocab_info_dict, features = feature_mapping.generate_vocab(naive_features)

            # feature splitting as training dataset, validation dataset, testing dataset
            train_features, test_features, train_y, test_y, train_name_list, test_name_list = \
                train_test_split(features, gt_label, name_list, test_size=0.2, random_state=0)
            train_features, val_features, train_y, val_y, train_name_list, val_name_list = \
                train_test_split(train_features, train_y, train_name_list, test_size=0.25, random_state=0)

            # select frequent features
            vocab_selected, vocab_info_dict_selcted = \
                feature_mapping.select_feature(train_features, train_y, vocab, vocab_info_dict, dim=10000)
            MSG = "After feature selection, the feature number is {} vs. {}".format(len(vocab_selected), len(vocab))
            logger.info(msg=MSG)

            if self.feature_mp == 'count':
                training_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, train_features, status='train')
                val_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, val_features)
                test_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, test_features)
            elif self.feature_mp == 'binary':
                training_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, train_features, status='train')
                val_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, val_features)
                test_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, test_features)
            else:
                raise ValueError("Not supported")

            # save features and feature representations
            utils.dump_pickle(vocab_selected, config.get('feature.' + self.feature_tp, 'vocabulary'))
            utils.dump_pickle(vocab_info_dict_selcted, config.get('feature.' + self.feature_tp, 'vocab_info'))
            utils.dump_joblib([training_feature_vectors, val_feature_vectors, test_feature_vectors],
                              config.get('feature.' + self.feature_tp, 'dataX'))
            utils.dump_joblib([train_y, val_y, test_y],
                              config.get('feature.' + self.feature_tp, 'datay'))

            utils.write_whole_file('\n'.join(train_name_list + val_name_list + test_name_list),
                                   config.get('dataset', 'name_list'))
        except Exception as ex:
            logger.error(str(ex))
            sys.exit(1)

    def feature_extraction(self, apk_paths, is_ordering=True):
        """
        feature extraction
        @param apk_paths: the list of applications
        @param is_ordering: return the list of features corresponds to the apk_paths
        """
        feature_save_dir = os.path.join("/tmp", "apk_data")

        if os.path.exists(feature_save_dir):
            # delete the files related to features
            shutil.rmtree(feature_save_dir, ignore_errors=True)
            # a loosely checking
            # file_number = len(os.listdir(feature_save_dir))
            # assert file_number == len(apk_paths), "Feature extraction halts: there are feature files in directory '{}', and please remove it if it is not necessary anymore".format(feature_save_dir)

        get_droid_feature(apk_paths, feature_save_dir, feature_type=self.feature_tp)
        feature_mapping = FeatureMapping(feature_save_dir, feature_type=self.feature_tp)
        if is_ordering:
            feature = feature_mapping.preprocess_feature(is_ordering, apk_paths)
        else:
            feature = feature_mapping.preprocess_feature()
        if not os.path.exists(config.get('feature.' + self.feature_tp, 'vocabulary')):
            logger.warning("No vocabulary.")
            return np.array([])
        vocab = utils.read_pickle(config.get('feature.' + self.feature_tp, 'vocabulary'))

        if self.feature_mp == 'count':
            return feature_mapping.count_feature_mapping_normalized(vocab, feature)
        else:
            return feature_mapping.binary_feature_mapping_normalized(vocab, feature)

    def model_graph(self, reuse=False):
        """
        build the graph
        """
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='X')
        self.y_input = tf.placeholder(dtype=tf.int64, shape=[None, ], name='Y')
        self.is_training = tf.placeholder(tf.bool, name="TRAIN")

        tf.set_random_seed(self.hp_params.random_seed)
        self.logits, self.y_tensor = self.forward(self.x_input, self.y_input, reuse=reuse)
        self.model_inference()

    def forward(self, x_tensor, y_tensor, reuse=False):
        """
        let data pass through the neural network
        :param x_tensor: input data
        :type: Tensor.float32
        :param y_tensor: label
        :type: Tensor.int64
        :param reuse: Boolean
        :return: Null
        """
        self.nn = graph
        _1, _2, logits = graph(
            x_tensor, self.hidden_layers, self.output_dim,
            self.is_training, name=self.model_name, reuse=reuse
        )
        return logits, y_tensor

    def model_inference(self):
        """
        model inference
        """
        # loss definition
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=self.y_tensor,
            logits=self.logits
        )

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

    def get_logits(self, dataX):
        """
        return logits for given data
        @param dataX 2D tensor
        """
        return self.nn(dataX,
                       self.hidden_layers,
                       self.output_dim,
                       is_training=False,
                       name=self.model_name,
                       reuse=True)[-1]

    def get_label_dict(self):
        """
        return label information, '1' means malicious examples and '0' means benign
        """
        def name_label_mapping(apk_dir, malicious=True):
            name_label_dict = {}
            for apk_path in list(utils.retrive_files_set(apk_dir, "", '|.apk')):
                apk_name = os.path.splitext(os.path.basename(apk_path))[0]
                if malicious:
                    name_label_dict[apk_name] = 1  # malicious label
                else:
                    name_label_dict[apk_name] = 0  # benign label
            return name_label_dict

        malware_label_mapping = name_label_mapping(self.mal_dir, malicious=True)
        benware_label_mapping = name_label_mapping(self.ben_dir, malicious=False)
        return dict(malware_label_mapping, **benware_label_mapping)

    def train(self, trainX=None, trainy=None, valX=None, valy=None):
        """train dnn"""
        if trainX is None or trainy is None or valX is None or valy is None:
            # load training dataset and validation dataset
            trainX, valX, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'dataX'))
            trainy, valy, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'datay'))

        train_input = utils.DataProducer(trainX, trainy, self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)
        val_input = utils.DataProducer(valX, valy, self.hp_params.batch_size, name='val')

        global_train_step = tf.train.get_or_create_global_step()
        saver = tf.train.Saver()
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.cross_entropy)
        merged_summaries = tf.summary.merge_all()

        # optimizer
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(self.hp_params.learning_rate).minimize(self.cross_entropy,
                                                                                      global_step=global_train_step)
        tf_cfg = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        tf_cfg.gpu_options.allow_growth = True
        tf_cfg.gpu_options.per_process_gpu_memory_fraction = 1.
        sess = tf.Session(config=tf_cfg)

        with sess.as_default():
            summary_writer = tf.summary.FileWriter(self.save_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            training_time = 0.0
            train_input.reset_cursor()
            output_steps = 50
            best_f1_val = 0.
            for step_idx, X_batch, y_batch in train_input.next_batch():
                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx + 1) % output_steps == 0:
                    print('Step {}/{}:{}'.format(step_idx + 1, train_input.steps, datetime.now()))
                    val_input.reset_cursor()
                    val_res_list = [sess.run([self.accuracy, self.y_pred], feed_dict={self.x_input: valX_batch,
                                                                                      self.y_input: valy_batch,
                                                                                      self.is_training: False}) \
                                    for [_, valX_batch, valy_batch] in val_input.next_batch()
                                    ]
                    val_res = np.array(val_res_list, dtype=object)
                    _acc = np.mean(val_res[:, 0])
                    _pred_y = np.concatenate(val_res[:, 1])
                    from sklearn.metrics import f1_score
                    _f1_score = f1_score(valy, _pred_y[:valy.shape[0]])

                    print('    validation accuracy {:.5}%'.format(_acc * 100))
                    print('    validation f1 score {:.5}%'.format(_f1_score * 100))

                    if step_idx != 0:
                        print('    {} samples per second'.format(
                            output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))

                    if best_f1_val < _f1_score:
                        best_f1_val = _f1_score
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        saver.save(sess,
                                   os.path.join(self.save_dir, 'checkpoint'),
                                   global_step=global_train_step)

                start = default_timer()
                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()

    def test_rpst(self, testX=None, testy=None, is_single_class=False):
        self.mode = 'test'
        if testX is None and testy is None:
            _, _, testX = utils.read_joblib(config.get('feature.' + self.feature_tp, 'dataX'))
            _, _, testy = utils.read_joblib(config.get('feature.' + self.feature_tp, 'datay'))

        if len(testX) == 0:
            print("No test data.")
            return

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()

        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        if cur_checkpoint is None:
            print("No saved parameters")
            return
        # load parameters
        saver = tf.train.Saver()
        eval_dir = os.path.join(self.save_dir, 'eval')
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            y_pred = tester(sess, testX, testy, self)

            from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
            accuracy = accuracy_score(testy, y_pred)
            b_accuracy = balanced_accuracy_score(testy, y_pred)

            MSG = "The accuracy on the test dataset is {:.5f}%"
            print(MSG.format(accuracy * 100))
            MSG = "The balanced accuracy on the test dataset is {:.5f}%"
            print(MSG.format(b_accuracy * 100))
            if not is_single_class:
                tn, fp, fn, tp = confusion_matrix(testy, y_pred).ravel()

                fpr = fp / float(tn + fp)
                fnr = fn / float(tp + fn)
                f1 = f1_score(testy, y_pred, average='binary')
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                    tf.Summary.Value(tag='f1 score', simple_value=f1)
                ])

                if not os.path.exists(eval_dir):
                    os.mkdir(eval_dir)
                summary_writer = tf.summary.FileWriter(eval_dir)
                summary_writer.add_summary(summary)

                print("Other evaluation metrics we may need:")
                MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
                print(MSG.format(fnr * 100, fpr * 100, f1 * 100))
            sess.close()
        return accuracy

    def test(self, apks, gt_labels, is_single_class=False, is_save_feat=True):
        try:
            testX = self.feature_extraction(apks)
            assert len(testX) == len(gt_labels), 'inconsistent first dimension'
            if is_save_feat:
                save_path = config.get('attack', 'advX')
                utils.dumpdata_np(testX, save_path)

        except Exception as ex:
            logger.error(str(ex))
            print("Feature extraction failed.")
            return 1

        return self.test_rpst(testX, gt_labels, is_single_class)


def _main():
    basic_dnn = BasicDNNModel()
    basic_dnn.train()
    basic_dnn.test_rpst()


if __name__ == "__main__":
    _main()
