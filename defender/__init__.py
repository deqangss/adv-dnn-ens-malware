"""Some classifiers will inherit the learner"""
from collections import defaultdict

from defender.at import AdversarialTrainingDNN, ADV_TRAIN_HP
from defender.at_ma import AdversarialTrainingDNNMax
from defender.ade_ma import AdversarialDeepEnsembleMax
from defender.d_ade_ma import DAdversarialDeepEnsembleMax


defense_model_scope_dict = {
    'atrfgsm' : AdversarialTrainingDNN,
    'atadam' : AdversarialTrainingDNN,
    'atma' : AdversarialTrainingDNNMax,
    'adema' : AdversarialDeepEnsembleMax,
    'dadema' : DAdversarialDeepEnsembleMax
}

defense_model_scope_dict = defaultdict(**defense_model_scope_dict)