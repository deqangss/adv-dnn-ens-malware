"""attack method abstract class"""

import os
import sys

from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)

from tools import utils
from tools.utils import get_min_max_bound, normalize_transform, normalize_inverse
from config import config

class Attack(object):
    """Abstract base class for all attack classes."""
    __metaclass__ = ABCMeta
    def __init__(self, targeted_model,
                 input_dim,
                 insertion_perm_array,
                 removal_perm_array,
                 normalizer = None,
                 verbose = False):
        """
        Initialize the attack method class
        :param targeted_model: victim model
        :param input_dim: feature vector dimension
        :param insertion_perm_array: binary 1D array servers as indicator, showing which features can be inserted
        :param removal_perm_array: binary  1D array, showing which features can be removed
        :param normalizer: feature normalizer, sklearn normalizer
        :param verbose: print information
        """
        self.model = targeted_model
        self.input_dim = input_dim
        self.verbose = verbose

        self.insertion_perm_array = np.array(insertion_perm_array)
        self.removal_perm_array = np.array(removal_perm_array)
        self.normalizer = normalizer

        self.clip_min = None
        self.clip_max = None
        self.scaled_clip_min = None
        self.scaled_clip_max = None

        if normalizer is not None:
            self.clip_min, self.clip_max = get_min_max_bound(normalizer = normalizer)
            self.scaled_clip_min = normalize_transform(np.reshape(self.clip_min, (1, -1)), normalizer = normalizer)
            self.scaled_clip_max = normalize_transform(np.reshape(self.clip_max, (1, -1)), normalizer = normalizer)

    @abstractmethod
    def perturb(self, dataX, ground_truth_labels):
        raise NotImplementedError