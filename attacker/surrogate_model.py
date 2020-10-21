"""
DNN for android malware detection, using the variant of drebin features.
"""
import os
import sys
import shutil
from datetime import datetime
from timeit import default_timer

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from config import config, logging

from tools import utils
from learner.feature_extractor import get_droid_feature, FeatureMapping, drebin
from learner.basic_DNN import BasicDNNModel, tester
from learner import model_scope_dict
from defender import defense_model_scope_dict

MODLE_TEMP = BasicDNNModel

models = list(model_scope_dict.values()) + list(defense_model_scope_dict.values())

logger = logging.getLogger("learning.surrogate")

SUR_INFO = {
    'dataset_dir': os.path.join(config.get('DEFAULT', 'database_dir'),
                                config.get('DEFAULT', 'surrogate_dataset')),
    'feature_type': 'drebin',
    'feature_utility_rate': 1.,
    'feature_mapping_type': config.get('feature.drebin', 'feature_mp'),
    'use_interdependent_features': False,
    'learning_algorithm': 'DNN'
}

SUR_DNN_HP = {
    'random_seed': 3456,
    'hidden_units': [200, 200], #number of layers is fixed
    'output_dim': 2,
    'n_epochs': 30,
    'batch_size': 128,
    'learning_rate':0.001,
    'optimizer': 'adam' # others are not supported
}

def model_template_setting(model):
    global MODLE_TEMP
    if model in models:
        MODLE_TEMP = model
    else:
        raise ValueError("No such algorithm.")
def model_template_reset():
    global MODLE_TEMP
    MODLE_TEMP = BasicDNNModel

class SurrogateModel(MODLE_TEMP):
    def __init__(self, info_dict=None, hyper_params_dict = None, reuse = False, name = 'SURROGATE'):
        if info_dict is None:
            print("Information of model should be provided.")
            return
        if hyper_params_dict is None:
            print("Hyper-parameters are needed.")
            return

        self.info_dict = info_dict
        self.info = utils.ParamWrapper(self.info_dict)

        self.hp_dict = hyper_params_dict
        self.hp_params = utils.ParamWrapper(self.hp_dict)

        self.feature_tp = self.info.feature_type
        self.feature_mp = self.info.feature_mapping_type
        self.feature_utility_rate = self.info.feature_utility_rate
        self.dataset_dir = self.info.dataset_dir

        self.name = name

        self.mal_dir = os.path.join(self.dataset_dir, config.get('dataset', 'malware_dir_name'))
        self.ben_dir = os.path.join(self.dataset_dir, config.get('dataset', 'benware_dir_name'))


        tmp_save_dir = config.get('experiments', 'surrogate_save_dir')
        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)

        self.save_dir = tmp_save_dir

        # self._data_preprocess()

        # model necessaries
        self.input_dim = len(
            utils.read_pickle(os.path.join(self.save_dir, 'vocabulary')))  # update in the future
        self.hidden_layers = self.hp_params.hidden_units
        self.output_dim = self.hp_params.output_dim

        # self.model_graph()

        super(SurrogateModel, self).__init__(self.info_dict, self.hp_dict, reuse, is_saving=False, name = self.name)

    def graph_reset(self):
        tf.reset_default_graph()
        self.model_graph(reuse = False)

    def _data_preprocess(self):
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
            gt_label = np.array([label_dict[os.path.splitext(name.strip())[0]] \
                                 for name in name_list])

            if len(naive_features) == 0:
                logger.error("No features extracted.")
                return

            if not self.info.use_interdependent_features:
                naive_features = feature_mapping.remove_interdependent_featrues(naive_features)

            vocab, vocab_info_dict, feat_purified = feature_mapping.generate_vocab(naive_features)

            # feature splitting as training dataset, validation dataset,  testing dataset
            train_features, test_features, train_y, test_y, train_name_list, test_name_list = \
                train_test_split(feat_purified, gt_label, name_list, test_size=0.2, random_state=0)
            train_features, val_features, train_y, val_y, train_name_list, val_name_list = \
                train_test_split(train_features, train_y, train_name_list, test_size=0.25, random_state=0)

            # select features
            vocab_selected, vocab_info_dict_selcted = \
                feature_mapping.select_feature(train_features, train_y, vocab, vocab_info_dict, dim=10000)

            # feature preprocessing based on the feature utility rate
            if abs(self.feature_utility_rate - 1.) < 1e-10:
                naive_features = naive_features
            elif self.feature_utility_rate > 0. and self.feature_utility_rate < 1.:
                # todo
                pass
            else:
                raise ValueError

            if self.feature_mp == 'count':
                training_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, train_features, status='train')
                val_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, val_features)
                test_feature_vectors = \
                    feature_mapping.count_feature_mapping_normalized(vocab_selected, test_features)
            else:
                training_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, train_features, status='train')
                val_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, val_features)
                test_feature_vectors = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, test_features)

            utils.dump_pickle(vocab_selected, os.path.join(self.save_dir, 'vocabulary'))
            utils.dump_pickle(vocab_info_dict_selcted, os.path.join(self.save_dir, 'vocab_info'))
            utils.dump_joblib([training_feature_vectors, val_feature_vectors, test_feature_vectors],
                              os.path.join(self.save_dir, 'dataX'))
            utils.dump_joblib([train_y, val_y, test_y],
                              os.path.join(self.save_dir, 'datay'))
            utils.write_whole_file('\n'.join(name_list), os.path.join(self.save_dir, 'name_list'))
        except KeyError as ex:
            logger.error(str(ex))
            sys.exit(1)

        except Exception as ex:
            logger.error(str(ex))
            sys.exit(1)

    def feature_extraction(self, apk_paths, inorder = True):
        feat_save_dir = os.path.join("/tmp", "apk_data")
        if os.path.exists(feat_save_dir):
            shutil.rmtree(feat_save_dir)
        get_droid_feature(apk_paths, feat_save_dir, feature_type=self.feature_tp)
        feature_mapping = FeatureMapping(feat_save_dir, feature_type=self.feature_tp)
        if inorder:
            feature = feature_mapping.preprocess_feature(inorder, apk_paths)
        else:
            feature = feature_mapping.preprocess_feature()
        if not os.path.exists(os.path.join(self.save_dir, 'vocabulary')):
            logger.info("No vocabulary.")
            return np.array([])
        vocab = utils.read_pickle(os.path.join(self.save_dir, 'vocabulary'))

        if self.feature_mp == 'count':
            return feature_mapping.count_feature_mapping_normalized(vocab, feature)
        else:
            return feature_mapping.binary_feature_mapping_normalized(vocab, feature)

    def train(self, trainX = None, trainy = None, valX = None, valy = None):
        """train dnn based malware detector"""
        if trainX is None and trainy is None:
            trainX, valX, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'dataX'))
            trainy, valy, _ = utils.read_joblib(config.get('feature.' + self.feature_tp, 'datay'))

        train_input_supervised = utils.DataProducer(trainX, trainy,
                                                    self.hp_params.batch_size, n_epochs=self.hp_params.n_epochs)
        val_input = utils.DataProducer(valX, valy, self.hp_params.batch_size, name='test')

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
            train_input_supervised.reset_cursor()
            output_steps = 50
            best_val_acc = 0.
            for step_idx, X_batch, y_batch in train_input_supervised.next_batch():
                train_dict = {
                    self.x_input: X_batch,
                    self.y_input: y_batch,
                    self.is_training: True
                }

                if (step_idx + 1) % output_steps == 0:
                    print('Step {}/{}:{}'.format(step_idx + 1, train_input_supervised.steps, datetime.now()))
                    val_input.reset_cursor()
                    val_accs = [sess.run(self.accuracy, feed_dict={self.x_input: valX_batch,
                                                                    self.y_input: valy_batch,
                                                                    self.is_training: False}) \
                                 for [_, valX_batch, valy_batch] in val_input.next_batch()
                                 ]
                    _acc = np.mean(val_accs)
                    print('    validation accuracy {:.5}%'.format(_acc * 100))
                    if step_idx != 0:
                        print('    {} samples per second'.format(
                            output_steps * self.hp_params.batch_size / training_time))
                        training_time = 0.

                    summary = sess.run(merged_summaries, feed_dict=train_dict)
                    summary_writer.add_summary(summary, global_train_step.eval(sess))

                    if best_val_acc < _acc:
                        if not os.path.exists(self.save_dir):
                            os.makedirs(self.save_dir)
                        saver.save(sess, os.path.join(self.save_dir, 'checkpoint'),
                                   global_step=global_train_step)

                start = default_timer()
                sess.run(optimizer, feed_dict=train_dict)
                end = default_timer()
                training_time = training_time + end - start
        sess.close()

    def test_athand(self):
        dataX = utils.readdata_np(os.path.join(self.save_dir, 'dataX'))
        datay = utils.readdata_np(os.path.join(self.save_dir, 'datay'))
        train_idx, val_idx, test_idx = utils.train_validation_test_split(dataX.shape[0])

        if len(test_idx) == 0:
            print("No test data.")
            return

        # rebuild the graph
        tf.reset_default_graph()
        self.model_graph()

        cur_checkpoint = tf.train.latest_checkpoint(self.save_dir)
        if cur_checkpoint is None:
            print("No saved parameters")
            return

        saver = tf.train.Saver()
        eval_dir = os.path.join(self.save_dir, 'eval')
        sess = tf.Session()
        with sess:
            saver.restore(sess, cur_checkpoint)
            tester(sess, dataX[test_idx], datay[test_idx], self, eval_dir)
            sess.close()

    def test(self, apks, gt_labels):
        raise NotImplementedError

def _main():
    surrogate_model = SurrogateModel(SUR_INFO, SUR_DNN_HP, False)
    surrogate_model.train()
    surrogate_model.test_athand()

if __name__ == "__main__":
    _main()