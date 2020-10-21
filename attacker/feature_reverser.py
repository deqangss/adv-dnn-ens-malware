import os
import sys

import numpy as np
from collections import defaultdict

from abc import ABCMeta, abstractmethod

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config as cfg
from attacker.modifier import INSTR_ALLOWED, MetaInstTemplate, APIInstrSpecTmpl, OPERATOR, MetaDelimiter
from learner.feature_extractor import feature_type_scope_dict
from learner.drebin import get_api_ingredient
from tools import utils

logger = cfg.logging.getLogger("learning.basic_dnn")

class FeatureReverse(object):
    """Abstract base class for inverse feature classes."""
    __metaclass__ = ABCMeta

    def __init__(self, feature_type, feature_mp, use_default_feature = True):
        """
        feature reverse engineering
        :param feature_type: feature type, e.g., drebin
        :param feature_mp: binary bag of words, or counting the occurrence of words
        :param use_default_feature: use the default meta feature information or not, if False the surrogate feature will be leveraged
        """
        self.feature_type = feature_type
        self.feature_mp = feature_mp
        self.use_default_feature = use_default_feature

        self.insertion_array = None
        self.removal_array = None

        self.normalizer = None

    @abstractmethod
    def get_mod_array(self):
        raise NotImplementedError

    @abstractmethod
    def generate_mod_instruction(self, sample_paths, perturbations):
        raise NotImplementedError

def _check_instr(instr):
    elements = instr.strip().split(MetaDelimiter)
    if str.lower(elements[0]) in OPERATOR.values() and \
            str.lower(elements[1]) in cfg.COMP.values():
        return True
    else:
        return False


def _check_instructions(instruction_list):
    if not isinstance(instruction_list, list):
        instruction_list = [instruction_list]
    for instr in instruction_list:
        if not _check_instr(instr):
            return False
    return True

class DrebinFeatureReverse(FeatureReverse):
    def __init__(self, feature_mp='binary', use_default_feature = True):
        super(DrebinFeatureReverse, self).__init__(list(feature_type_scope_dict.keys())[0],
                                                   feature_mp,
                                                   use_default_feature)
        #load feature infomation
        try:
            if self.use_default_feature:
                self.normalizer = utils.read_pickle(cfg.config.get('feature.' + self.feature_type, 'normalizer'))
                self.vocab = utils.read_pickle(cfg.config.get('feature.' + self.feature_type, 'vocabulary'))
                self.vocab_info = utils.read_pickle(cfg.config.get('feature.' + self.feature_type, 'vocab_info'))
            else: # use surrogate feature meta-information
                self.normalizer = utils.read_pickle(os.path.join(
                    cfg.config.get('experiments', 'surrogate_save_dir'), 'normalizer'))
                self.vocab = utils.read_pickle(os.path.join(
                        cfg.config.get('experiments', 'surrogate_save_dir'), 'vocabulary'))
                self.vocab_info = utils.read_pickle(os.path.join(
                        cfg.config.get('experiments', 'surrogate_save_dir'), 'vocab_info'))
        except Exception as ex:
            logger.error(str(ex))
            raise IOError("Unable to load meta-information of feature.")

    def get_mod_array(self):
        """
        get binary indicator of showing the feature can be either modified or not
        '1' means modifiable and '0' means not
        """
        insertion_array = []
        removal_array = []

        if not os.path.exists(cfg.config.get('feature.' + self.feature_type, 'vocabulary')):
            print("No feature key words at {}.".format(cfg.config.get('feature.' + self.feature_type, 'vocabulary')))
            return insertion_array, removal_array
        if not os.path.exists(cfg.config.get('feature.' + self.feature_type, 'vocab_info')):
            print(
                "No feaure key words description at {}.".format(cfg.config.get('feature.' + self.feature_type, 'vocab_info')))

        word_catagory_dict = feature_type_scope_dict[self.feature_type].get_word_category(self.vocab,
                                                                                          self.vocab_info,
                                                                                          cfg.COMP)

        insertion_array = np.zeros(len(self.vocab), )
        removal_array = np.zeros(len(self.vocab), )

        for i, word in enumerate(self.vocab):
            cat = word_catagory_dict.get(word)
            if cat is not None:
                if cat in INSTR_ALLOWED[OPERATOR[0]]:
                    insertion_array[i] = 1
                else:
                    insertion_array[i] = 0
                if cat in INSTR_ALLOWED[OPERATOR[1]]:
                    removal_array[i] = 1
                else:
                    removal_array[i] = 0
            else:
                raise ValueError("Incompatible value.")

        return insertion_array, removal_array

    def generate_mod_instruction(self, sample_paths, perturbations):
        '''
        generate the instructions for samples in the attack list
        :param sample_paths: the list of file path
        :param perturbations: numerical perturbations on the un-normalized feature space, type: np.ndarray
        :return: {sample_path1: [meta_instruction1, ...], sample_path2: [meta_instruction1, ...],...}
        '''
        assert len(sample_paths) == len(perturbations)

        word_catagory_dict = feature_type_scope_dict[self.feature_type].get_word_category(self.vocab,
                                                                                          self.vocab_info,
                                                                                          cfg.COMP)
        instrs = defaultdict(list)
        for idx, path in enumerate(sample_paths):
            perturb_values = perturbations[idx][perturbations[idx].astype(np.int32) != 0]
            perturb_entities = np.array(self.vocab)[perturbations[idx].astype(np.int32) != 0].tolist()
            meta_instrs = []
            for e_idx, e in enumerate(perturb_entities):
                if perturb_values[e_idx] > 0:
                    _operator = OPERATOR[0]  # 'insert'
                else:
                    _operator = OPERATOR[1]  # 'remove'

                # template: "{operator}##{component}##{specName}##{Count}"
                _cat = word_catagory_dict[e]
                _info = list(self.vocab_info[e])[0]
                if _cat is cfg.COMP['Notdefined']:
                    continue

                if _cat in [cfg.COMP['Android_API'], cfg.COMP['Java_API']]:
                    class_name, method_name, params = get_api_ingredient(_info)
                    _info = APIInstrSpecTmpl.format(ClassName=class_name,ApiName=method_name, ApiParameter=params)
                if self.feature_mp == 'count':
                    _count = abs(int(round(perturb_values[e_idx])))
                elif self.feature_mp == 'binary':
                    if _operator == OPERATOR[0]:
                        _count = abs(int(round(perturb_values[e_idx])))
                    else:
                        _count = int(round(perturb_values[e_idx]))
                else:
                    raise ValueError("Allowing feature mapping type: 'count' or 'binary'")

                meta_instr = MetaInstTemplate.format(Operator = _operator, Component=_cat, SpecName=_info, Count=_count)
                if perturb_values[e_idx] > 0:
                    meta_instrs.append(meta_instr)
                else:
                    meta_instrs.insert(0, meta_instr)
            if _check_instructions(meta_instrs):
                instrs[path] = meta_instrs
            else:
                raise AssertionError(" Generate the incorrent intructions.")
        return instrs







