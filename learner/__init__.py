"""
classifier implementation: data preprocessing and model learning
"""
from collections import defaultdict
from learner import feature_extractor

from learner.basic_DNN import BasicDNNModel, DNN_HP

_model_scope_dict = {
    'basic_dnn': BasicDNNModel
}

model_scope_dict = defaultdict(**_model_scope_dict)


