from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import warnings
import sys
import math

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
flags = tf.flags
FLAGS = flags.FLAGS

from tools.base_util import *


def str2bool(v):
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (e.g., True or Fasle)!")


class ParamWrapper(object):
    def __init__(self, params):
        if not isinstance(params, dict):
            params = vars(params)
        self.params = params

    def __getattr__(self, name):
        val = self.params.get(name)
        if val is None:
            MSG = "Setting params ({}) is deprecated"
            warnings.warn(MSG.format(name))
            val = FLAGS.__getattr__(name)
        return val


def weight_init(shape, name):
    return tf.Variable(
        tf.random_uniform(shape, -tf.sqrt(6. / (shape[0] + shape[-1])), tf.sqrt(6. / (shape[0] + shape[-1]))),
        name=name)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    return tf.train.Feature(floatlist=tf.train.FloatList(value=[value]))


def initialize_uninitilized_global_variables(sess):
    # from https://github.com/tensorflow/cleverhans/tree/master/cleverhans
    # List all global variables
    global_vars = tf.global_variables()
    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def zero_masking(input_dim, keeping_ratio = 0.5, random_seed = 0):
    np.random.seed(random_seed)
    return (np.random.uniform(0., 1., size = (1, input_dim)) <= keeping_ratio).astype(np.float32)

def optimize_linear(grad, eps, ord=np.inf):
  """
  code is from: https://github.com/tensorflow/cleverhans
  Solves for the optimal input to a linear function under a norm constraint.

  Optimal_perturbation = argmax_{eta, ||eta||_{ord} < eps} dot(eta, grad)

  :param grad: tf tensor containing a batch of gradients
  :param eps: float scalar specifying size of constraint region
  :param ord: int specifying order of norm
  :returns:
    tf tensor containing optimal perturbation
  """

  # In Python 2, the `list` call in the following line is redundant / harmless.
  # In Python 3, the `list` call is needed to convert the iterator returned by `range` into a list.
  red_ind = list(range(1, len(grad.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    # Take sign of gradient
    optimal_perturbation = tf.sign(grad)
    # The following line should not change the numerical results.
    # It applies only because `optimal_perturbation` is the output of
    # a `sign` op, which has zero derivative anyway.
    # It should not be applied for the other norms, where the
    # perturbation has a non-zero derivative.
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
  elif ord == 1:
    abs_grad = tf.abs(grad)
    sign = tf.sign(grad)
    max_abs_grad = tf.reduce_max(abs_grad, red_ind, keepdims=True)
    tied_for_max = tf.to_float(tf.equal(abs_grad, max_abs_grad))
    num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
    optimal_perturbation = sign * tied_for_max / num_ties
  elif ord == 2:
    square = tf.maximum(avoid_zero_div,
                        tf.reduce_sum(tf.square(grad),
                                   reduction_indices=red_ind,
                                   keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)
  else:
    raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

  # Scale perturbation to be the solution for the norm=eps rather than
  # norm=1 problem
  scaled_perturbation = tf.multiply(eps, optimal_perturbation)
  return scaled_perturbation

def test_func(sess, model, dataX, datay, batch_size=50):
    """Get accuracy from a trained model with fixed vaiable names"""
    try:
        model.mode = 'test'
        mini_batchs = dataX.shape[0] // batch_size + 1
        _accuracy = []
        for i in range(mini_batchs):
            start_i = i * batch_size
            end_i = start_i + batch_size
            if end_i > dataX.shape[0]:
                end_i = dataX.shape[0]
            if start_i == end_i:
                continue

            test_dict = {
                model.x_input: dataX[start_i: end_i],
                model.y_input: datay[start_i: end_i],
                model.is_training: False
            }
            _acc = sess.run(model.accuracy, feed_dict=test_dict)
            _accuracy.append(_acc)
        model.mode = 'train'
        return np.mean(_accuracy)
    except Exception as ex:
        model.mode = 'train'
        raise ValueError(str(ex) + "\n\t Model stated variable names are different from template.")

def get_min_max_bound(normalizer=None):
    '''
    get the min and max contraints for data,
    :param normalizer: the normalizer, if None, load it from default location
    :return: minimum value and maximum value for each dimension
    '''
    if normalizer is not None:
        return normalizer.data_min_, normalizer.data_max_
    else:
        raise ValueError("No normalizer exists!")

def normalize_transform(X, normalizer=None):
    """Normalize feature into [0,1]"""
    if normalizer is not None:
        scale_data = normalizer.transform(X)
        return scale_data
    else:
        raise ValueError("No normalizer exists!")

def normalize_inverse(X, normalizer=None):
    """Remap the normalized feature into original feature space"""
    if normalizer is not None:
        if np.min(X) < 0 and np.max(X) > 1.:
            warnings.warn("The data is not within the range [0, 1]")
        return normalizer.inverse_transform(X)
    else:
        raise IOError("No normalizer exists!")

def get_other_classes_batch(nb_classes, current_classes):
    '''
    get the class labels except for the given class
    :param nb_classes: the number of categories, integer
    :param current_class: given the class label, integer
    :return: a list contains the other class labels
    '''
    if not isinstance(nb_classes, int) and not isinstance(current_classes, np.ndarray):
        msg = "Current class label must be 1D array.\n"
        raise TypeError(msg)
    if np.min(current_classes) < 0 and np.max(current_classes) >= nb_classes:
        msg = "The given class should be within the range [0, nb_classes).\n"
        raise  ValueError(msg)

    other_classes = []
    for idx in range(current_classes.shape[0]):
        classes = list(range(nb_classes))
        classes.remove(current_classes[idx])
        other_classes.append(classes)
    return np.array(other_classes)

def lab2onehot(indices, depth=None, on_value = 1.,off_value=0.):
    indices = np.array(indices)
    if len(indices) == 0:
        return
    if len(indices.shape) > 1:
        return
    if indices.dtype != np.int32:
        indices = indices.astype(np.int32)

    if depth == None:
        depth = np.max(indices) + 1

    onehot_enc = np.zeros((indices.shape[0], depth), dtype = np.float32)
    onehot_enc[:] = off_value
    onehot_enc[range(indices.shape[0]), indices] = on_value
    return onehot_enc

# ============================IO==========================================

def readdata_np(data_path):
    try:
        with open(data_path, 'rb') as f_r:
            data = np.load(f_r)
        return data
    except IOError as e:
        sys.stderr.write("Unable to open {0}.\n".format(data_path))
        sys.exit(1)



def dumpdata_np(data, data_path):
    if not isinstance(data, np.ndarray):
        warnings.warn("The array is not the numpy.ndarray type.")
    data_dir = os.path.dirname(data_path)
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(data_path, 'wb') as f_s:
            np.save(f_s, data)
    except OSError as e:
        sys.stderr.write(e)

def readtxt(path, mode = 'r'):
    if os.path.isfile(path):
        with open(path, mode) as f_r:
            lines = f_r.read().strip().splitlines()
    else:
        sys.stderr.write("Only file path is supported.\n")
    return lines

def file_missing_warnings(attack_files):
    sys.stderr.write("Unable to locate the file {}. ".format(attack_files))
    sys.stderr.write("Please show the correct file path for {}. ".format(attack_files))
    sys.exit()

def get_files(src, file_ext=""):
    files = []
    if os.path.isdir(src):
        files = [os.path.join(src, name) for name in os.listdir(src) if
                 os.path.isfile(os.path.join(src, name)) and os.path.splitext(name)[1]==ext]
    return files

def dump_pickle(data, path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with open(path, 'wb') as wr:
        pkl.dump(data, wr)
    return True


def read_pickle(path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))

def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        from sklearn.externals import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
    except IOError:
        raise IOError("Dump data failed.")

def read_joblib(path):
    from sklearn.externals import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        sys.stderr.write(e)
        sys.exit(1)


class DataProducer(object):
    def __init__(self, dataX, datay, batch_size, n_epochs = None, n_steps=None, name='train'):
        '''
        The data factory yield data at designated batch size and steps
        :param dataX: 2-D array numpy type supported. shape: [num, feat_dims]
        :param datay: 2-D or 1-D array.
        :param batch_size: setting batch size for training or testing. Only integer supported.
        :param n_epochs: setting epoch for training. The default value is None
        :param n_steps: setting global steps for training. The default value is None. If provided, param n_epochs will be neglected.
        :param name: 'train' or 'test'. if the value is 'test', the n_epochs will be set to 1.
        '''
        try:
            assert(name=='train' or name == 'test' or name == 'val')
        except Exception as e:
            raise AssertionError("Only support selections: 'train', 'val' or 'test'.\n")

        self.dataX = dataX
        self.datay = datay
        self.batch_size = batch_size
        self.mini_batches = self.dataX.shape[0] // self.batch_size
        if self.dataX.shape[0] % self.batch_size > 0:
            self.mini_batches = self.mini_batches + 1
            if (self.dataX.shape[0] > self.batch_size) and \
                    (name == 'train' or name == 'val'):
                np.random.seed(0)
                rdm_idx = np.random.choice(self.dataX.shape[0], self.batch_size - self.dataX.shape[0] % self.batch_size, replace=False)
                self.dataX = np.vstack([dataX, dataX[rdm_idx]])
                self.datay = np.concatenate([datay, datay[rdm_idx]])

        if name == 'train':
            if n_epochs is not None:
                self.steps = n_epochs * self.mini_batches
            elif n_steps is not None:
                self.steps = n_steps
            else:
                self.steps = None
        if name == 'test' or name == 'val':
            self.steps = None

        self.name = name
        self.cursor = 0
        if self.steps is None:
            self.max_iterations = self.mini_batches
        else:
            self.max_iterations = self.steps

    def next_batch(self):
        while self.cursor < self.max_iterations:
            pos_cursor = self.cursor % self.mini_batches
            start_i = pos_cursor * self.batch_size

            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            if start_i == end_i:
                break

            yield self.cursor, self.dataX[start_i:end_i], self.datay[start_i: end_i]
            self.cursor = self.cursor + 1

    def next_batch2(self):
        while self.cursor < self.max_iterations:
            pos_cursor = self.cursor % self.mini_batches
            start_i = pos_cursor * self.batch_size
            if start_i == self.dataX.shape[0]:
                start_i = 0
            end_i = start_i + self.batch_size
            if end_i > self.dataX.shape[0]:
                end_i = self.dataX.shape[0]
            self.cursor = self.cursor + 1
            yield self.cursor, start_i, end_i, self.dataX[start_i:end_i], self.datay[start_i: end_i]

    def reset_cursor(self):
        self.cursor = 0

    def get_current_cursor(self):
        return self.cursor

def train_validation_test_split(vec_num, seed = 2345):
    if isinstance(vec_num, int) and vec_num > 0:
        train_idx, test_idx = train_test_split(range(vec_num), test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)
        return train_idx, val_idx, test_idx
    else:
        return None

#data operands
def array2onehot(indices, depth=None, on_value = 1.,off_value=0.):
    '''
    convert the indices (1D numpy array) into one-hot encodings.
    :param indices: 1D array
    :param depth: the output is [number_samples, depth]
    :param on_value: the value will be assigned at the positions on indices
    :param off_value: the value will be assigned at the positions off indices
    :return: one-hot encoding 2-D array
    '''
    indices = np.array(indices)
    if len(indices) == 0:
        return
    if len(indices.shape) > 1:
        return
    if indices.dtype != np.int32:
        indices = indices.astype(np.int32)

    if depth == None:
        depth = np.max(indices) + 1

    onehot_enc = np.zeros((indices.shape[0], depth), dtype = np.float32)
    onehot_enc[:] = off_value
    onehot_enc[range(indices.shape[0]), indices] = on_value
    return onehot_enc

def one_hot2labels(one_hot):
    return np.argmax(one_hot, axis = -1)

#helper functions for attack methods
def round_x(x, alpha = 0.5):
    '''
    rounds x by thresholding it according to alpha
    :param x: input 2D array
    :param alpha: threshold, a scalar
    :return: float tensor of 0s and 1s
    '''
    if isinstance(x, tf.Tensor):
        return tf.to_float(x >= alpha)
    elif isinstance(x, np.ndarray):
        return (x >= alpha).astype(np.float32)
    else:
        warnings.warn("Doesn't support.\n")
        raise NotImplementedError

def or_float_tensors(x1, x2):
    '''
    operand OR of tensor x1 and tensor x2
    :param x1: Tensor
    :param x2: Tensor, shape is same as x1
    :return: float tensor of 0s and 1s
    '''
    if isinstance(x1, tf.Tensor) and isinstance(x2, tf.Tensor):
        return tf.to_float(tf.bitwise.bitwise_or(tf.cast(x1, tf.int8),
                                                 tf.cast(x2, tf.int8)))
    else:
        return np.bitwise_or(x1.astype(np.int8), x2.astype(np.int8)).astype(np.float32)



def get_initial_starting(x, sampling = False):
    '''
    randomly initialize the start point for data x.
    functionality is preserved
    :param x: training data 2D tf.tensor
    :param sampling:flag to sample randomly from feasible area or just use x
    :return:randomly sampled starting point of x
    '''
    if sampling:
        rand_x = round_x(np.random.uniform(size = x.shape))
        return or_float_tensors(x, rand_x)
    else:
        return x

def get_batch_size(integer, max_number_accepted = 50):
    def divisorGenerator(n):
        large_divisors = []
        for i in range(1, int(math.sqrt(n) + 1)):
            if n % i == 0:
                yield i
                if i*i != n:
                    large_divisors.append(n / i)
        for divisor in reversed(large_divisors):
            yield divisor
    if isinstance(integer, int):
        divisors = list(divisorGenerator(integer))
        for d in divisors:
            if d >= max_number_accepted:
                return int(d)
            else:
                pass
        return int(divisors[-1])
    else:
        raise TypeError("Incorrent input type, Int is accepted.")