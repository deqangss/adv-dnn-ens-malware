
import os
import sys
import time

from learner.basic_DNN import BasicDNNModel, INFO, DNN_HP
from attacker.attack_manager import AttackManager
from learner import DNN_HP
from defender import AdversarialTrainingDNN, ADV_TRAIN_HP
from defender import AdversarialTrainingDNNMax, AdversarialDeepEnsembleMax, DAdversarialDeepEnsembleMax

class Learner(object):
    def __init__(self):
        self.model = BasicDNNModel()

    def train(self):
        MSG = "Train basic neural networks."
        print(MSG)
        MSG = "The model information is:\n {}".format(self.model.info_dict)
        print(MSG)
        MSG = "The hyper-parameters are defined as:\n {}".format(self.model.hp_params_dict)
        print(MSG)
        time.sleep(10)
        self.model.mode = 'train'
        self.model.train()

    def predict(self, apks = None, gt_lables = None):
        self.model.mode = 'test'
        if apks is None and gt_lables is None:
            self.model.test_rpst()
        else:
            self.model.test(apks, gt_lables)
    def pred_adv(self):
        from config import config
        from tools.utils import readdata_np, retrive_files_set
        import numpy as np
        self.model.mode = 'test'

        adv_apks_dir = config.get('attack', 'adv_sample_dir')
        if os.path.exists(adv_apks_dir):
            adv_apks_path = list(retrive_files_set(adv_apks_dir, '', '.apk'))
        else:
            print("No adversarial malware samples. Exit!")
            sys.exit(0)
        if len(adv_apks_path) <= 0:
            print("No adversarial malware samples. Exit!")
            sys.exit(0)
        adv_apks_label = np.array([1.] * len(adv_apks_path))
        # assert len(adv_apks_path) <= len(advX) # some apks may fail to be perturbed into adversarial versions
        print("Test on adversarial malware samples:")
        self.model.test(adv_apks_path, adv_apks_label, is_single_class=True)

        adv_feature_path = config.get('attack', 'advX')
        if os.path.exists(adv_feature_path):
            advX = readdata_np(adv_feature_path)
        else:
            print("No adversarial instances. Exit!")
            sys.exit(0)
        if len(advX) <= 0:
            print("No adversarial instances. Exit!")
            sys.exit(0)
        advy = np.array([1.] * len(advX))
        print("\n\nTest on adversarial feature representation:")
        self.model.test_rpst(advX, advy, is_single_class=True)

        pristine_dataX = readdata_np(config.get('attack', 'attackX'))
        y = np.array([1.] * len(pristine_dataX))
        print("\n\nTest on pristine malware sample:")
        self.model.test_rpst(pristine_dataX, y, is_single_class=True)


class Attacker(object):
    def __init__(self, attack_scenario = 'white-box', victim_name = 'basic_dnn', attack_method='fgsm',
                 is_real_sample = True):
        self.method = attack_method
        self.targeted_model_name = victim_name
        self.scenario = attack_scenario
        self.is_real_sample = is_real_sample
        self.attack_mgr = AttackManager(self.method, self.scenario, self.targeted_model_name,self.is_real_sample)

    def attack(self):
        return self.attack_mgr.attack()

class Defender(object):
    def __init__(self, defense_method_name = 'adv_training'):
        self.defense_method_name = defense_method_name
        self.defense = None

        if self.defense_method_name == 'atrfgsm':
            adv_train_hp = ADV_TRAIN_HP.copy()
            adv_train_hp.update(DNN_HP)
            adv_train_hp['maximizer_name'] = 'pgd_inf'
            self.defense = AdversarialTrainingDNN(hyper_params = adv_train_hp)
        elif self.defense_method_name == 'atadam':
            adv_train_hp = ADV_TRAIN_HP.copy()
            adv_train_hp.update(DNN_HP)
            adv_train_hp['maximizer_name'] = 'pgd_adam'
            self.defense = AdversarialTrainingDNN(hyper_params=adv_train_hp)
        elif self.defense_method_name ==  'atma':
            self.defense = AdversarialTrainingDNNMax()
        elif self.defense_method_name == 'adema':
            self.defense = AdversarialDeepEnsembleMax()
        elif self.defense_method_name == 'dadema':
            self.defense = DAdversarialDeepEnsembleMax()
        else:
            raise ValueError(
                "Please choose method from 'atdfgsm', 'atadam', 'atma', 'adema', and 'dadema.")

    def train(self):
        MSG = "Train defense {}.".format(self.defense_method_name)
        print(MSG)
        MSG = "The model information is:\n {}".format(self.defense.info_dict)
        print(MSG)
        MSG = "The hyper-parameters are defined as:\n {}".format(self.defense.hp_params_dict)
        print(MSG)
        time.sleep(10)
        self.defense.mode = 'train'
        self.defense.train()

    def predict(self, apks=None, gt_lables=None):
        self.defense.mode = 'test'
        if apks is None and gt_lables is None:
            self.defense.test_rpst()
        else:
            self.defense.test(apks, gt_lables)

    def pred_adv(self):
        from config import config
        from tools.utils import readdata_np, retrive_files_set
        import numpy as np
        self.defense.mode = 'test'
        adv_apks_dir = config.get('attack', 'adv_sample_dir')
        if os.path.exists(adv_apks_dir):
            adv_apks_path = list(retrive_files_set(adv_apks_dir, '', '.apk'))
        else:
            print("No adversarial malware samples. Exit!")
            sys.exit(0)
        if len(adv_apks_path) <= 0:
            print("No adversarial malware samples. Exit!")
            sys.exit(0)
        adv_apks_label = np.array([1.] * len(adv_apks_path))
        # assert len(adv_apks_path) <= len(advX)  # some apks may fail to be perturbed into its adversarial versions
        print("\n\nTest on adversarial malware samples:")
        self.defense.test(adv_apks_path, adv_apks_label, is_single_class=True)

        adv_feature_path = config.get('attack', 'advX')
        if os.path.exists(adv_feature_path):
            advX = readdata_np(adv_feature_path)
        else:
            print("No adversarial feature representation. Exit!")
            sys.exit(0)
        if len(advX) <= 0:
            print("No adversarial feature representation. Exit!")
            sys.exit(0)
        advy = np.array([1.] * len(advX))
        print("\n\nTest on adversarial feature representation:")
        self.defense.test_rpst(advX, advy, is_single_class=True)

        pristine_dataX = readdata_np(config.get('attack', 'attackX'))
        y = np.array([1.] * len(pristine_dataX))
        print("\n\nTest on pristine malware sample:")
        self.defense.test_rpst(pristine_dataX, y, is_single_class=True)