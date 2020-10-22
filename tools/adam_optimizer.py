"""
The adam optimizer. codes are from Cleverhans:https://github.com/tensorflow/cleverhans.
The reason is the TF adam optimizer may have some issues. Please look the 'UnrolledOptimizer'
"""

import tensorflow as tf

class TensorOptimizer(object):
    """Functional-stype optimizer which does not use TF Variables.
    UnrolledOptimizers implement optimizers where the values being optimized
    are ordinary Tensors, rather than Variables. TF Variables can have strange
    behaviors when being assigned multiple times within a single sess.run()
    call, particularly in Distributed TF, so this avoids thinking about those
    issues. In cleverhans, these are helper classes for the `pgd_attack`
    method.
    """

    def _compute_gradients(self, loss_fn, x, unused_optim_state):
        """Compute a new value of `x` to minimize `loss_fn`.

        Args:
            loss_fn: a callable that takes `x`, a batch of images, and returns
                a batch of loss values. `x` will be optimized to minimize
                `loss_fn(x)`.
            x: A list of Tensors, the values to be updated. This is analogous
                to the `var_list` argument in standard TF Optimizer.
            unused_optim_state: A (possibly nested) dict, containing any state
                info needed for the optimizer.

        Returns:
            new_x: A list of Tensors, the same length as `x`, which are updated
            new_optim_state: A dict, with the same structure as `optim_state`,
                which have been updated.
        """

        # Assumes `x` is a list,
        # and contains a tensor representing a batch of images
        assert len(x) == 1 and isinstance(x, list), \
            'x should be a list and contain only one image tensor'
        x = x[0]
        loss = tf.reduce_mean(loss_fn(x), axis=0)
        return tf.gradients(loss, x)

    def _apply_gradients(self, grads, x, optim_state):
        raise NotImplementedError(
            "_apply_gradients should be defined in each subclass")

    def minimize(self, loss_fn, x, optim_state):
        grads = self._compute_gradients(loss_fn, x, optim_state)
        return self._apply_gradients(grads, x, optim_state)

    def init_optim_state(self, x):
        """Returns the initial state of the optimizer.

        Args:
            x: A list of Tensors, which will be optimized.

        Returns:
            A dictionary, representing the initial state of the optimizer.
        """
        raise NotImplementedError(
            "init_optim_state should be defined in each subclass")

class TensorAdam(TensorOptimizer):
    """
    The Adam optimizer defined in https://arxiv.org/abs/1412.6980.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-9):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self.optim_state = {}

    def init_state(self, x):
        optim_state = {}
        optim_state["t"] = 0.
        optim_state["m"] = [tf.zeros_like(v) for v in x]
        optim_state["u"] = [tf.zeros_like(v) for v in x]
        self.optim_state = optim_state
        return optim_state

    def _apply_gradients(self, grads, x, optim_state):
        """Refer to parent class documentation."""
        new_x = [None] * len(x)
        new_optim_state = {
            "t": optim_state["t"] + 1.,
            "m": [None] * len(x),
            "u": [None] * len(x)
        }
        t = new_optim_state["t"]
        for i in range(len(x)):
            g = grads[i]
            m_old = optim_state["m"][i]
            u_old = optim_state["u"][i]
            new_optim_state["m"][i] = (
                self._beta1 * m_old + (1. - self._beta1) * g)
            new_optim_state["u"][i] = (
                self._beta2 * u_old + (1. - self._beta2) * g * g)
            m_hat = new_optim_state["m"][i] / (1. - tf.pow(self._beta1, t))
            u_hat = new_optim_state["u"][i] / (1. - tf.pow(self._beta2, t))
            new_x[i] = (
                x[i] - self._lr * m_hat / (tf.sqrt(u_hat) + self._epsilon))
        return new_x, new_optim_state

class NadamOptimizer(TensorAdam):
    def __init__(self, lr=0.001, mu=0.9, ups=0.999, epsilon=1e-9):
        super(NadamOptimizer, self).__init__(lr = lr, beta1= mu, beta2= ups, epsilon= epsilon)

    def _apply_gradients(self, grads, x, optim_state):
        new_x = [None] * len(x)
        new_optim_state = {
            "t": optim_state["t"] + 1.,
            "m": [None] * len(x),
            "u": [None] * len(x)
        }
        t = new_optim_state["t"]
        for i in xrange(len(x)):
            g = grads[i]
            m_old = optim_state["m"][i]
            u_old = optim_state["u"][i]
            new_optim_state["m"][i] = (
                self._beta1 * m_old + (1. - self._beta1) * g)
            new_optim_state["u"][i] = (
                self._beta2 * u_old + (1. - self._beta2) * g * g)

            # m_hat = new_optim_state["m"][i] / (1. - tf.pow(self._beta1, t))
            # u_hat = new_optim_state["u"][i] / (1. - tf.pow(self._beta2, t))

            m_hat = self._beta1 * new_optim_state["m"][i] / (1. - tf.pow(self._beta1, t + 1)) + (
                        1. - self._beta1) * g / (1. - tf.pow(self._beta1, t))
            u_hat = self._beta2 * new_optim_state["u"][i] / (1. - tf.pow(self._beta2, t))

            new_x[i] = (
                x[i] - self._lr * m_hat / (tf.sqrt(u_hat) + self._epsilon))
        return new_x, new_optim_state
