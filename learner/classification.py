"""classifier abstract class"""
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class Classifier(object):
    """Abstract base class for all classifier classes."""
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def _data_preprocess(self):
        """
        feature extraction
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x_tensor, y_tensor, reuse = False):
        """
        let data pass through the neural network
        :param x_tensor: input data
        :type: Tensor.float32
        :param y_tensor: label
        :type: Tensor.int64
        :param reuse: Boolean
        :return: Null
        """
        raise  NotImplementedError

    @abstractmethod
    def model_inference(self):
        """
        model inference, such as prediction, loss, ......
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, trainX = None, trainy = None, valX = None, valy = None):
        """
        train a model upon (trainX, trainy), if value is none, default data will be leveraged
        :param trainX: np.2Darray
        :param trainy: np.1Darray
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, apks, gt_labels):
        """
        Conducting test on a list of apks
        @param apks: a list of application paths
        @param gt_labels: corresponding ground truth labels
        """
        raise NotImplementedError