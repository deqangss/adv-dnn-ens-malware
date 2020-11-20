"""Extract various types of features"""
import os
import collections
import warnings

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from tools import utils
from config import config, COMP, logging

from learner import drebin
from learner.drebin import DREBIN_FEAT_INFO

logger = logging.getLogger('learner.feature')

def normalize_data(X, is_fitting = False, feature_type = 'drebin'):
    if is_fitting:
        minmax_norm = MinMaxScaler()
        normalizer = minmax_norm.fit(X)
        utils.dump_pickle(normalizer, config.get('feature.' + feature_type, 'normalizer'))
    elif os.path.exists(config.get('feature.' + feature_type, 'normalizer')) and not is_fitting:
        normalizer = utils.read_pickle(config.get('feature.' + feature_type, 'normalizer'))
    else:
        raise ValueError("Unable to find the normalizer")
    feat_normlized = normalizer.transform(X)
    return feat_normlized


class FeatureMapping(object):
    def __init__(self, feature_save_dir, feature_type='drebin'):
        """
        process the feature data and get the numarial feature array
        :param feature_save_dir: save directory of feature documents
        :param feature_type: e.g., drebin
        """
        self.save_dir = feature_save_dir
        self.feature_tp = feature_type

    def load_features(self):
        if self.feature_tp in feature_type_scope_dict.keys():
            raw_feature_list, name_list = feature_type_scope_dict[self.feature_tp].load_features(self.save_dir)
        else:
            raise ValueError("No this type of feature '{}' and the avaiable types are '{}' ".format(self.feature_tp,
                                                                                                    ','.join(
                                                                                                        feature_type_scope_dict.keys())))
        return raw_feature_list, name_list

    def remove_interdependent_featrues(self, raw_features):
        """Remove the interdependent features"""
        if self.feature_tp in feature_type_scope_dict.keys():
            raw_feature_list = feature_type_scope_dict[self.feature_tp].remove_interdependent_features(raw_features)
        else:
            raise ValueError("No this type of feature '{}' and the avaiable types are '{}' ".format(self.feature_tp,
                                                                                                    ','.join(
                                                                                                        feature_type_scope_dict.keys())))
        return raw_feature_list

    def select_feature(self, features, gt_label, vocab, vocab_info_dict, dim = 100000):
        """
        select features based on the given dimension, or remove the zero value features.
        """
        if not isinstance(features, list):
            raise TypeError("A list of features are needed, but here {}.".format(type(features)))

        pos_loc = (gt_label == 1)
        feature_list_pos = np.array(features)[pos_loc]
        if len(feature_list_pos) <= 0:
            raise ValueError("No positives.")
        feature_vec_pos = self.binary_feature_mapping(vocab, feature_list_pos, short_type=True)
        feature_frq_pos = np.sum(feature_vec_pos, axis=0) / float(len(feature_vec_pos))

        neg_loc = ~pos_loc
        feature_list_neg = np.array(features)[neg_loc]
        if len(feature_list_neg) <= 0:
            raise ValueError("No negatives.")
        feature_vec_neg = self.binary_feature_mapping(vocab, feature_list_neg, short_type=True)
        feature_frq_neg = np.sum(feature_vec_neg, axis=0) / float(len(feature_vec_neg))

        zero_indicator = np.all(feature_vec_pos == 0, axis=0) & np.all(feature_vec_neg == 0, axis=0)
        vocab_reduced = list(np.array(vocab)[~zero_indicator])
        vocab_info_reduced = defaultdict(set,
                                         {k: v for k, v in vocab_info_dict.items() if k in vocab_reduced})

        if len(vocab_reduced) <= dim:
            return vocab_reduced, vocab_info_reduced
        else:
            feature_frq_diff = np.abs(feature_frq_pos[~zero_indicator] - feature_frq_neg[~zero_indicator])
            pos_selected = np.argsort(feature_frq_diff)[::-1][:dim]

            vocab_selected = []
            vocab_info_dict_selected = defaultdict(set)
            for p in pos_selected:
                w = vocab_reduced[p]
                vocab_selected.append(w)
                vocab_info_dict_selected[w] = vocab_info_reduced[w]

            return vocab_selected, vocab_info_dict_selected

    def generate_vocab(self, raw_featureset):
        try:
            vocabulary, vocab_info, clean_featureset = \
                feature_type_scope_dict[self.feature_tp].get_vocab(raw_featureset)
            return vocabulary, vocab_info, clean_featureset
        except ValueError as ex:
            logger.error("Failed to get feature information, " + str(ex))
        except Exception as ex:
            logger.error("Failed to get feature information, " + str(ex))

    def preprocess_feature(self, inorder = False, order_sequence = []):
        if not os.path.isdir(self.save_dir):
            print("No features '.data' file.")
            return []
        if self.feature_tp in feature_type_scope_dict.keys():
            extractor = feature_type_scope_dict[self.feature_tp]
            if not inorder:
                feature_data, _ = extractor.load_features(self.save_dir)
            else:
                feature_data, apk_name_list = extractor.load_features(self.save_dir, order_sequence)
                assert len(apk_name_list) == len(
                    order_sequence), 'Cannot extract features for these files \n{}\n, please remove them!'.format(
                    extractor.get_incap_instances(apk_name_list, order_sequence)
                )
                # print(feature_data)
            if len(feature_data) == 0:
                warnings.warn("Got no features.", stacklevel=4)
                return []
            else:
                return extractor.preprocess_feature(feature_data)
        else:
            raise ValueError("No this type of feature '{}' and the avaiable types are '{}' ".format(self.feature_tp,
                                                                                                    ','.join(
                                                                                                        feature_type_scope_dict.keys())))
    def binary_feature_mapping(self, vocabulary, feature_list, short_type = False):
        if len(vocabulary) == 0:
            print("Return no features")
            return
        if len(feature_list) == 0:
            print("No features")
            return
        dictionary = dict(zip(vocabulary, range(len(vocabulary))))
        # feature_vectors = []
        # for v in feature_list:
        #     feature_vec = np.zeros((len(dictionary)), dtype=np.float32)
        #     if len(v) > 0:
        #         filled_pos = [idx for idx in list(map(dictionary.get, v)) if idx is not None]
        #         if len(filled_pos) == 0:
        #             feature_vec[:] = 0.
        #         else:
        #             feature_vec[np.array(filled_pos)] = 1.
        #     feature_vectors.append(feature_vec)
        # return np.array(feature_vectors)
        if not short_type:
            feature_vectors = np.zeros((len(feature_list), len(vocabulary)), dtype = np.float32)
        else:
            feature_vectors = np.zeros((len(feature_list), len(vocabulary)), dtype=np.float16)
        for i, v in enumerate(feature_list):
            if len(v) > 0:
                filled_pos = [idx for idx in list(map(dictionary.get, v)) if idx is not None]
                if len(filled_pos) != 0:
                    feature_vectors[i, filled_pos] = 1.
                else:
                    logger.warning("Zero feature vector exsits.")
                    warnings.warn("Zero feature vector exsits.")
        return feature_vectors

    def binary_feature_mapping_normalized(self, vocabulary, features, status = 'test'):

        feature_vectors = self.binary_feature_mapping(vocabulary, features)
        if status == 'train':
            return normalize_data(feature_vectors, True)
        else:
            return normalize_data(feature_vectors)

    def count_feature_mapping(self, vocabulary, features):
        feature_vectors = []
        for f in features:
            feature_dim = len(f)
            if feature_dim == 0:
                raise ValueError("No features")

            feature_counter = collections.Counter(f)
            feature_value = [feature_counter.get(v) if feature_counter.get(v) is not None else 0 for v in vocabulary]
            feature_vectors.append(feature_value)
        return np.array(feature_vectors).astype(np.float32)

    def count_feature_mapping_normalized(self, vocabulary, features, status = 'test'):
        feature_vectors = self.count_feature_mapping(vocabulary, features)

        if status == 'train':
            return normalize_data(feature_vectors, True)
        else:
            return normalize_data(feature_vectors)


feature_type_scope_dict= {
    'drebin': drebin
}

feature_type_scope_dict = defaultdict(**feature_type_scope_dict)

def get_droid_feature(data_container, save_dir, feature_type = 'drebin'):
    """
    extract android features for apks in the denoted directory or an apk
    :param data_container: a directory contains apk files or a list of apk paths
    :param feature_type: feature types
    :return:output dir contains document, contents of which are extracted features.
    """

    if isinstance(data_container, str):
        if os.path.isfile(data_container):
            apk_paths = [data_container]
        elif os.path.isdir(data_container):
            apk_paths = list(utils.retrive_files_set(data_container, "", ".apk|"))
        else:
            raise ValueError("Input error : {}".format(data_container))
    elif isinstance(data_container, list):
        for z in data_container:
            if not os.path.isfile(z):
                raise ValueError("Input error: The '{}' does not like as a file path.".format(z))
        apk_paths = data_container
    else:
        raise TypeError("Input error: Incorrect type {}".format(data_container))


    if feature_type in feature_type_scope_dict.keys():
        feature_type_scope_dict[feature_type].GetApkData(apk_paths, save_dir)
    else:
        raise ValueError("No this type of feature '{}' and the avaiable types are '{}' ".format(feature_type,
                                                                                                ','.join(feature_type_scope_dict.keys())))
'''
=========================================================================================
====================================data processing======================================
=========================================================================================
'''
def random_over_sampling(X, y, ratio = None):
    """
    over sampling
    :param X: data
    :type 2D numpy array
    :param y: label
    :type 1D numpy.ndarray
    :param ratio: proportion
    :type float
    :return: X, y
    """
    if ratio is None:
        return X, y
    if not isinstance(ratio, float):
        raise TypeError("{}".format(type(ratio)))
    if ratio > 1.:
        ratio = 1.
    if ratio < 0.:
        ratio = 0.

    if not isinstance(X, np.ndarray) and not isinstance(y, np.ndarray):
        raise TypeError

    count_array = np.bincount(y)
    max_count_num = np.max(count_array)
    curr_count = np.rint(max_count_num * ratio).astype(np.int64)
    X_amended_list = [X]
    y_amended_list = [y]
    for l in range(len(count_array)):
        if count_array[l] < curr_count:
            # extend the corresponding data
            random_indices = np.random.choice(
                np.where(y == l)[0], curr_count - count_array[l]
            )
            X_amended_list.append(X[random_indices])
            y_amended_list.append(y[random_indices])
        else:
            warnings.warn("The data labelled by {} is not conducted by over sampling ({} vs {}).".format(
                l, count_array[l], curr_count
            ), stacklevel= 4)

    def random_shuffle(x, random_seed = 0):
        np.random.seed(random_seed)
        np.random.shuffle(x)
    X_amended = np.concatenate(X_amended_list)
    random_shuffle(X_amended)
    y_amended = np.concatenate(y_amended_list)
    random_shuffle(y_amended)

    return X_amended, y_amended





