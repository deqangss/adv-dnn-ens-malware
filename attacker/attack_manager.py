import os
import sys
import time
import shutil

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config as cfg
from tools import utils

from attacker.methods import SaltAndPepper, \
    PointWise, JSMA, FGSM, GrosseAttack, \
    PGD, PGDl1, GDKDE, Mimicry, \
    PGDAdam, BCA_K, BGA_K, MAX
from attacker.feature_reverser import DrebinFeatureReverse
from learner.feature_extractor import feature_type_scope_dict
from learner import model_scope_dict
from defender import defense_model_scope_dict
from attacker.modifier import modify_sample, get_original_name, name_adv_file
from attacker.surrogate_model import SUR_DNN_HP, SUR_INFO, SurrogateModel

logger = cfg.logging.getLogger("attack_manager")
logger.addHandler(cfg.ErrorHandler)

# default hyper-parameters for waging attacks
# note: other hyper-parameters may produce better attack results
method_params_dict = {
    'saltandpepper': {'max_eta': 1., 'repetition': 100, 'random_seed': 0},
    'pointwise': {'repetition': 30, 'random_seed': 0},
    'fgsm': {'epsilon': 1., 'ord': 'l-infinity', 'batch_size': 50},
    'jsma': {'theta': 1, 'iterations': 100, 'batch_size': 50, 'force_iteration': False},
    'grosse': {'iterations': 100, 'batch_size': 50, 'force_iteration': False},
    'bca_k': {'iterations': 100, 'batch_size': 50, 'force_iteration': False},
    'bga_k': {'max_iteration': 100, 'batch_size': 50},
    'pgdlinf': {'step_size': 0.01, 'ord': 'l-infinity', 'rand_round': False, 'max_iteration': 1000, 'batch_size': 50},
    'pgdl2': {'step_size': 1., 'ord': 'l2', 'rand_round': False, 'max_iteration': 1000, 'batch_size': 50},
    'pgdl1': {'k': 1, 'step_size': 1., 'max_iteration': 100, 'batch_size': 50, 'force_iteration': False},
    'pgd_adam': {'learning_rate': 0.01, 'max_iteration': 1000, 'batch_size': 50},
    'gdkde': {'step_size': 1., 'max_iteration': 1000, 'negative_data_num': 5000, 'kernel_width': 20.,
              'lambda_factor': 1., 'distance_max': 20., 'xi': 0., 'batch_size': 50},
    'mimicry': {'trial': 30, 'random_seed': 0, 'is_reducing_pert': True},
    'max': {'random_seed': 0, 'iteration': 5, 'call_saltandpepper': False, 'use_fast_version': False, 'varepsilon': 1e-9,
            'attack_names': ['pgdl1', 'pgdl2', 'pgdlinf', 'pgd_adam', 'gdkde']}  #
}

attack_scope_dict = {
    'saltandpepper': SaltAndPepper,
    'pointwise': PointWise,
    'fgsm': FGSM,
    'jsma': JSMA,
    'grosse': GrosseAttack,
    'bca_k': BCA_K,
    'bga_k': BGA_K,
    'pgdlinf': PGD,
    'pgdl2': PGD,
    'pgdl1': PGDl1,
    'pgd_adam': PGDAdam,
    'gdkde': GDKDE,
    'mimicry': Mimicry,
    'max': MAX
}

attack_method_dict = {
    # wage grey-box, white-box attacks
    0: ['saltandpepper',
        'pointwise',
        'fgsm',
        'jsma',
        'grosse',
        'bca_k',
        'bga_k',
        'pgdlinf',
        'pgdl2',
        'pgdl1',
        'pgd_adam',
        'gdkde',
        'mimicry',
        'max'],
    1: []  # able to wage black-box based attacks, todo
}

# ['dataset', 'feature', 'algorithm', 'free_access', 'free_response']
GREYBOX = 'grey-box'
WHITEBOX = 'white-box'
BLACKBOX = 'black-box'
attack_scenario_dict = {
    WHITEBOX: {},
    GREYBOX: {'algo_knowledge': 0000, 'feature_knowledge': 1111, 'dataset_knowledge': 1111, 'free_feedback': 1,
              'free_access': 1},
    BLACKBOX: {'free_feedback': 1, 'free_access': 1}
}

targeted_model_names_dict = model_scope_dict.copy()
targeted_model_names_dict.update(defense_model_scope_dict)


class AttackManager(object):
    def __init__(self,
                 attack_method_name,
                 attack_scenario='white-box',
                 targeted_model_name='basic_dnn',
                 is_sample_level=True,
                 adv_file_checking=False,
                 **kwargs
                 ):
        """
        attack management
        @param attack_method_name: attack method such as mimicy
        @param attack_scenario: white-box or grey-box
        @param targeted_model_name: model name
        @param is_sample_level: whether generate the executable mawlare examples or not
        @param adv_file_checking: checking the feature representation of executable mawlare examples
        """
        self.attack_method_name = attack_method_name
        self.attack_scenario = attack_scenario
        self.targeted_model_name = targeted_model_name
        self.is_smaple_level = is_sample_level
        self.check = adv_file_checking & self.is_smaple_level
        self.attack_path_list = [os.path.join(
            os.path.join(cfg.config.get('dataset', 'dataset_root'),
                         cfg.config.get('dataset', 'malware_dir_name')
                         ),
            att_name) for att_name in utils.readtxt(cfg.config.get('dataset', 'attack_list'))
        ]
        self.gt_labels = [1.] * len(self.attack_path_list)  # only perturb the mawlare examples

        self.attack_mode = -1
        self.other_args = kwargs
        self.targeted_model_info = None
        self.feature_reverser = None
        self.feature_vectors_of_attacker = None
        self.targeted_model_of_attacker = None

        # variables may be used
        self.feature_rate_of_attacker = None
        self.algorithm_hp_of_attacker = None
        self.dataset_rate_of_attacker = None
        self.defense_klg_of_attacker = None

        self._initilize()

    def _initilize(self):
        """initialization"""
        all_method_names = []
        for mode, method_list in attack_method_dict.items():
            if self.attack_method_name in method_list:
                self.attack_mode = mode
            all_method_names.extend(method_list)

        if not self.attack_method_name in all_method_names:
            raise ValueError("\n\t Attack method '{}' are supported".format(all_method_names))

        if not self.attack_scenario in attack_scenario_dict.keys():
            raise ValueError("\n\t Attack scenario '{}' are supported".format(attack_scenario_dict.keys()))

        if not self.targeted_model_name in targeted_model_names_dict.keys():
            raise ValueError("\n\t targed model '{}' are supported".format(targeted_model_names_dict.keys()))

        # get the information of targeted model
        self.targeted_model = targeted_model_names_dict[self.targeted_model_name](mode='test')
        self.targeted_model_info = self.targeted_model.info
        self.targeted_model_hp = self.targeted_model.hp_params

        if self.attack_scenario == WHITEBOX:
            self.targeted_model_of_attacker = self.targeted_model
            if self.targeted_model_of_attacker.feature_tp == list(feature_type_scope_dict.keys())[0]:  # 'drebin'
                self.feature_reverser = DrebinFeatureReverse(feature_mp=self.targeted_model_of_attacker.feature_mp)
            else:
                raise ValueError("Only " + ' '.join(feature_type_scope_dict.keys()) + " are supported.")

        if self.attack_scenario == GREYBOX:
            """
            Training a dnn model as the surrogate model here.
            In the paper, we use the hardened model as the surrogate models 
            """
            sur_info_dict = {}
            sur_hp_dict = {}
            for k, v in self.other_args:
                if k in attack_scenario_dict[GREYBOX].keys():
                    attack_scenario_dict[GREYBOX][k] = v
                else:
                    raise ValueError("No '{}' key, please check it based on '{}'".format(k, ','.join(
                        attack_scenario_dict[GREYBOX].keys())))

            if attack_scenario_dict[GREYBOX]['algo_knowledge'] == 0000:  # zero knowledge about algorithm
                sur_info_dict['learning_algorithm'] = 'DNN'
                sur_hp_dict = SUR_DNN_HP.copy()
            else:
                raise NotImplementedError

            if attack_scenario_dict[GREYBOX]['feature_knowledge'] == 1111:
                sur_info_dict['feature_type'] = self.targeted_model_info.feature_type
                sur_info_dict['feature_mapping_type'] = self.targeted_model_info.feature_mapping_type
                sur_info_dict['feature_utility_rate'] = 1.
            else:
                raise NotImplementedError

            if attack_scenario_dict[GREYBOX]['dataset_knowledge'] == 1111:
                sur_info_dict['dataset_dir'] = self.targeted_model_info.dataset_dir
            else:
                raise NotImplementedError

            surrogate_model = SurrogateModel(sur_info_dict, sur_hp_dict, False)
            # surrogate_model.train()
            # surrogate_model.graph_reset() # reset the graph, avoiding the loading of adam parameters
            self.targeted_model_of_attacker = surrogate_model

            self.feature_reverser = DrebinFeatureReverse(feature_mp=surrogate_model.feature_mp,
                                                         use_default_feature=True)  # may trigger issue, surrogate model will use default features

        if self.attack_scenario == BLACKBOX:
            for k, v in self.other_args:
                if k in attack_scenario_dict[BLACKBOX].keys():
                    attack_scenario_dict[BLACKBOX][k] = v
                else:
                    raise ValueError("No '{}' key, please check it based on '{}'".format(k, ','.join(
                        attack_scenario_dict[BLACKBOX].keys())))

    def generate_perturbations(self, pert_ratio=100.):
        def resample_manip_set(insert_map, removal_map):
            """
            sample certain manipulations from total set randomly
            """
            if isinstance(insert_map, list):
                insert_map = np.array(insert_map)
            if isinstance(removal_map, list):
                insert_map = np.array(removal_map)
            assert len(insert_map) == len(removal_map)
            s = insert_map.shape

            if pert_ratio < 0 and pert_ratio > 100.:
                raise ValueError("Ratio should be in the range of [0,100]")

            if pert_ratio == 0:
                return np.zeros(s), np.zeros(s)
            elif pert_ratio == 100:
                return insert_map, removal_map
            else:
                p = pert_ratio / 100.
                np.random.seed(0)
                permmit_region = (np.random.uniform(0, 1, size=s) > 1. - p).astype(insert_map.dtype)
                insert_map_ = np.bitwise_and(insert_map.astype(np.int32),
                                             permmit_region.astype(np.int32))
                removal_map_ = np.bitwise_and(removal_map.astype(np.int32),
                                              permmit_region.astype(np.int32))
                return insert_map_.astype(insert_map.dtype), removal_map_.astype(removal_map.dtype)

        # load feature vectors for attack data
        if self.attack_scenario == WHITEBOX:
            if self.attack_mode == 0:
                if not os.path.exists(cfg.config.get('attack', 'attackX')):
                    self.feature_vectors_of_attacker = self.targeted_model_of_attacker.feature_extraction(
                        self.attack_path_list, is_ordering=True)
                    utils.dumpdata_np(self.feature_vectors_of_attacker, cfg.config.get('attack', 'attackX'))
                else:
                    self.feature_vectors_of_attacker = utils.readdata_np(cfg.config.get('attack', 'attackX'))

                # initialize attack
                insertion_perm_array, removal_perm_array = self.feature_reverser.get_mod_array()
                insertion_perm_array, removal_perm_array = resample_manip_set(insertion_perm_array, removal_perm_array)
                kwparams = method_params_dict[self.attack_method_name]

                attack = attack_scope_dict[self.attack_method_name](self.targeted_model_of_attacker,
                                                                    self.feature_vectors_of_attacker.shape[1],
                                                                    insertion_perm_array=insertion_perm_array,
                                                                    removal_perm_array=removal_perm_array,
                                                                    normalizer=self.feature_reverser.normalizer,
                                                                    verbose=True,
                                                                    **kwparams
                                                                    )
                logger.info(msg=kwparams)
                prist_feat_vec, adv_feat_vec, labels = \
                    attack.perturb(self.feature_vectors_of_attacker,
                                   np.ones(self.feature_vectors_of_attacker.shape[0]))
                return prist_feat_vec, adv_feat_vec, labels

            elif self.attack_mode == 1:
                raise NotImplementedError
            else:
                raise ValueError("Attack modes {} are allowed.".format(attack_method_dict.keys()))

        elif self.attack_scenario == GREYBOX:
            if self.attack_mode == 0:
                feature_saved_path = os.path.join(cfg.config.get('experiments', 'surrogate_save_dir'),
                                                  'attack_feature.data')
                if not os.path.exists(feature_saved_path):
                    self.feature_vectors_of_attacker = self.targeted_model_of_attacker.feature_extraction(
                        self.attack_path_list, is_ordering=True)
                    utils.dumpdata_np(self.feature_vectors_of_attacker, feature_saved_path)
                else:
                    self.feature_vectors_of_attacker = utils.readdata_np(feature_saved_path)

                insertion_perm_array, removal_perm_array = self.feature_reverser.get_mod_array()
                insertion_perm_array, removal_perm_array = resample_manip_set(insertion_perm_array, removal_perm_array)
                kwparams = method_params_dict[self.attack_method_name]

                attack = attack_scope_dict[self.attack_method_name](self.targeted_model_of_attacker,
                                                                    self.feature_vectors_of_attacker.shape[1],
                                                                    insertion_perm_array=insertion_perm_array,
                                                                    removal_perm_array=removal_perm_array,
                                                                    normalizer=self.feature_reverser.normalizer,
                                                                    verbose=True,
                                                                    **kwparams
                                                                    )

                prist_feat_vec, adv_feat_vec, labels = \
                    attack.perturb(self.feature_vectors_of_attacker,
                                   np.ones(self.feature_vectors_of_attacker.shape[0]))
                return prist_feat_vec, adv_feat_vec, labels

            elif self.attack_mode == 1:
                raise NotImplementedError
            else:
                raise ValueError("Attack modes {} are allowed.".format(attack_method_dict.keys()))

        elif self.attack_scenario == BLACKBOX:
            raise NotImplementedError
        else:
            raise ValueError("'{}' attack scenario is not support.".format(self.attack_scenario))

    def generate_exc_malware_sample(self, perturbations=None, adv_save_dir=None):
        """Modify the apk based on the numeral perturbations"""
        assert isinstance(perturbations, np.ndarray)
        assert perturbations.shape[0] % len(self.attack_path_list) == 0

        # Sample might have several perturbation vectors
        apk_paths = self.attack_path_list * (perturbations.shape[0] // len(self.attack_path_list))
        mod_instr = self.feature_reverser.generate_mod_instruction(apk_paths, perturbations)

        modify_sample(mod_instr, adv_save_dir, proc_number=4, vb=False)

        if self.check:
            """
            We check the perturbed APKs by comparing their feature representation to the perturbed representation
            """
            adv_save_paths = []
            for apk in self.attack_path_list:
                adv_save_paths.append(
                    os.path.join(adv_save_dir, name_adv_file(apk) + '.apk')
                )

            adv_features = self.targeted_model.feature_extraction(adv_save_paths)
            pris_data_path = os.path.join(cfg.config.get('attack', self.attack_method_name), "pristine_{}.data".format(
                method_params_dict[self.attack_method_name].get('ord', '')))
            if os.path.exists(pris_data_path):
                pris_feature_vectors = utils.readdata_np(pris_data_path)
            else:
                raise ValueError("No pristine data.")

            if len(adv_features) != len(pris_feature_vectors):
                logger.warning("Expect the same number of adversarial and pristine feature vectors ({} vs. {})".format(
                    len(adv_features),
                    len(pris_feature_vectors)
                ))
                return None, perturbations

            if self.feature_reverser.normalizer is not None:
                _perturbations = np.rint(utils.normalize_inverse(adv_features, self.feature_reverser.normalizer)) - \
                                 np.rint(utils.normalize_inverse(pris_feature_vectors, self.feature_reverser.normalizer))
            else:
                _perturbations = adv_features - pris_feature_vectors

            if not np.all(np.abs(_perturbations - perturbations) <= 5e-1):
                logger.warning("Unable to perturb some components exactly as generated perturbations.")
                unequal_pos = (abs(_perturbations - perturbations) > 1e-6)
                vocab = utils.read_pickle(cfg.config.get('feature.' + self.targeted_model.feature_tp, 'vocabulary'))
                for i in range(len(unequal_pos)):
                    if np.any(unequal_pos[i]):
                        MSG_INFO = "Failed to perturb some features:"
                        MSG_FILE = 'File name: {} with index {}'.format(apk_paths[i], i)
                        MSG_res = 'Required perturbations {} vs. Resulting perturbations {} corresponds to elements:{}'
                        MSG = MSG_INFO + '\n' + MSG_FILE + '\n' + \
                              MSG_res.format(perturbations[i, unequal_pos[i]],
                                             _perturbations[i, unequal_pos[i]],
                                             np.array(vocab)[unequal_pos[i]])
                        logger.warning(MSG)
            else:
                logger.info("Perturbed APKs follow the generated perturbations exactly.")
            return adv_features, perturbations
        else:
            return None, perturbations

    def attack(self):
        save_dir = cfg.config.get('attack', self.attack_method_name)
        if not os.path.exists(save_dir):
            utils.mkdir(save_dir)

        perturbations = None
        pristine_feature_vec = None
        adv_feature_vec = None
        labels = self.gt_labels
        try:
            pristine_feature_vec, adv_feature_vec, labels = self.generate_perturbations()
            save_path = os.path.join(save_dir, "pristine_{}.data".format(
                method_params_dict[self.attack_method_name].get('ord', '')))
            utils.dumpdata_np(pristine_feature_vec, save_path)
            save_path = cfg.config.get('attack', 'advX')
            utils.dumpdata_np(adv_feature_vec, save_path)

            # backup
            save_path = os.path.join(save_dir, "{}_{}.data".format(
                self.attack_method_name,
                method_params_dict[self.attack_method_name].get('ord', '')))
            utils.dumpdata_np(adv_feature_vec, save_path)
            save_path = os.path.join(save_dir, "{}_{}.label".format(
                self.attack_method_name,
                method_params_dict[self.attack_method_name].get('ord', '')))
            utils.dumpdata_np(labels, save_path)

            if self.feature_reverser.normalizer is not None:
                perturbations = utils.normalize_inverse(adv_feature_vec, self.feature_reverser.normalizer) - \
                                utils.normalize_inverse(pristine_feature_vec, self.feature_reverser.normalizer)
            else:
                perturbations = adv_feature_vec - pristine_feature_vec
        except Exception as ex:
            logger.exception(ex)
            logger.error(str(ex))
            logger.error("Failed to generate perturbations.")
            return 1

        if perturbations is None:
            adv_feat_save_dir = cfg.config.get('attack', self.attack_method_name)
            adv_data_path = os.path.join(
                adv_feat_save_dir,
                '{}_{}.data'.format(self.attack_method_name,
                                    method_params_dict[self.attack_method_name].get('ord', '')))
            pris_data_path = os.path.join(adv_feat_save_dir, "pristine_{}.data".format(
                method_params_dict[self.attack_method_name].get('ord', '')))

            if os.path.exists(adv_data_path) and os.path.exists(pris_data_path):
                adv_feature_vec = utils.readdata_np(adv_data_path)
                pristine_feature_vec = utils.readdata_np(pris_data_path)
            else:
                raise ValueError("No perturbations.")

            if self.feature_reverser.normalizer is not None:
                perturbations = utils.normalize_inverse(adv_feature_vec, self.feature_reverser.normalizer) - \
                                utils.normalize_inverse(pristine_feature_vec, self.feature_reverser.normalizer)
            else:
                perturbations = adv_feature_vec - pristine_feature_vec
            logger.warn("Perturbations generated from snapshot with degree {:.5f}".format(
                np.mean(np.sum(np.abs(perturbations), axis=1))
            ))

        if not self.is_smaple_level:
            # collect info.
            # (1) scale of perturbations
            perturbations_amount_l0 = np.mean(np.sum(np.abs(perturbations) > 1e-6, axis=1))
            perturbations_amount_l1 = np.mean(np.sum(np.abs(perturbations), axis=1))
            perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(perturbations), axis=1)))
            msg = "Average scale of perturbations on adversarial feature vector measured by l0 norm {:.5f}, l1 norm {:.5f}, l2 norm {:.5f}"
            print(msg.format(perturbations_amount_l0, perturbations_amount_l1, perturbations_amount_l2))
            logger.info(msg.format(perturbations_amount_l0, perturbations_amount_l1, perturbations_amount_l2))

            # (2) accuracy on pristine feature vector and perturbed feature vector
            acc_prist = self.targeted_model.test_rpst(pristine_feature_vec, self.gt_labels, is_single_class=True)
            print("Accuracy on pristine features:", acc_prist)
            logger.info("Accuracy on pristine features:{:.5f}".format(acc_prist))
            acc_pert = self.targeted_model.test_rpst(adv_feature_vec, labels, is_single_class=True)
            print("Accuracy on perturbed features:", acc_pert)
            logger.info("Accuracy on perturbed features:{:.5f}".format(acc_pert))
        else:
            try:
                save_dir = os.path.join(save_dir, 'adv_apks')
                adv_features, perturbations = \
                    self.generate_exc_malware_sample(perturbations, save_dir)
                test_adv_dir = cfg.config.get('attack', 'adv_sample_dir')
                if os.path.exists(test_adv_dir):
                    shutil.rmtree(test_adv_dir, ignore_errors=True)
                shutil.copytree(save_dir, cfg.config.get('attack', 'adv_sample_dir'))
            except Exception as ex:
                logger.error(str(ex))
                logger.exception(ex)
                logger.error("Failed to modify the APKs.")
                return 2

            # we dump the apk information here.
            # If the malicious functionality should be checked, please run ./oracle/run_oracle.py
            # self.estimate_functionality(save_dir) # todo

            # collect info.
            # (1) scale of perturbations
            perturbations_amount_l0 = np.mean(np.sum(np.abs(perturbations) > 1e-6, axis=1))
            perturbations_amount_l1 = np.mean(np.sum(np.abs(perturbations), axis=1))
            perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(perturbations), axis=1)))
            msg = "Average scale of perturbations on adversarial feature vector measured by l0 norm {:.5f}, l1 norm {:.5f}, l2 norm {:.5f}"
            print(msg.format(perturbations_amount_l0, perturbations_amount_l1, perturbations_amount_l2))
            logger.info(msg.format(perturbations_amount_l0, perturbations_amount_l1, perturbations_amount_l2))

            # (2) accuracy on pristine feature vector and perturbed feature vector
            acc_prinst = self.targeted_model.test_rpst(pristine_feature_vec, self.gt_labels, is_single_class=True)
            print("Accuracy on pristine features:", acc_prinst)
            logger.info("Accuracy on pristine features:{:.5f}".format(acc_prinst))
            acc_pert = self.targeted_model.test_rpst(adv_feature_vec, labels, is_single_class=True)
            print("Accuracy on perturbed features:", acc_pert)
            logger.info("Accuracy on perturbed features:{:.5f}".format(acc_pert))

            # (3) perturbations and accuracy on adversarial malware samples
            if adv_features is None:
                adv_apk_names = os.listdir(save_dir)
                adv_apk_paths = [os.path.join(save_dir, name) for name in adv_apk_names]
                adv_features = self.targeted_model.feature_extraction(adv_apk_paths)
            utils.dumpdata_np(adv_features, cfg.config.get('attack', 'radvX'))
            perturbations = adv_features - pristine_feature_vec
            perturbations_amount_l0 = np.mean(np.sum(np.abs(perturbations) > 1e-6, axis=1))
            perturbations_amount_l1 = np.mean(np.sum(np.abs(perturbations), axis=1))
            perturbations_amount_l2 = np.mean(np.sqrt(np.sum(np.square(perturbations), axis=1)))
            msg = "Average scale of perturbations on adversarial malware measured by l0 norm {:.5f}, l1 norm {:.5f}, l2 norm {:.5f}"
            print(msg.format(perturbations_amount_l0, perturbations_amount_l1, perturbations_amount_l2))
            logger.info(msg.format(perturbations_amount_l0, perturbations_amount_l1, perturbations_amount_l2))

            acc_adv_mal = self.targeted_model.test_rpst(adv_features, self.gt_labels, is_single_class=True)
            print("Accuracy on adversarial malware samples:", acc_adv_mal)
            logger.info("Accuracy on adversarial malware samples:{:.5f}".format(acc_adv_mal))

        return 0

def _main():
    attack_mgr = AttackManager(attack_method_name='pgdlinf',
                               attack_scenario='white-box',
                               targeted_model_name='basic_dnn',
                               is_sample_level=False)
    return attack_mgr.attack()


if __name__ == "__main__":
    sys.exit(_main())
