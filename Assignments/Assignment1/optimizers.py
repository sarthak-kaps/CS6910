import numpy as np


class Optimizer:
    learning_rate = None

    # the appropriate child class should set the learning rate as suited to the optimization
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # child class must implement this function
    def apply_gradients(self, gradients, weights):
        pass


class SGD(Optimizer):
    nesterov = False
    momentum = None
    weight_velocity = None
    bias_velocity = None

    def __init__(self, learning_rate=0.1, momentum=0, nesterov=False):
        super(SGD, self).__init__(learning_rate)

        self.nesterov = nesterov
        self.momentum = momentum

        if momentum is not None and (momentum > 1 or momentum < 0):
            raise ValueError(
                "Error : Momentum must lie between 0 and 1 both inclusive")

    def compile(self, weight_shape, bias_shape):
        self.weight_velocity = np.zeros(weight_shape)
        self.bias_velocity = np.zeros(bias_shape)

    def apply_gradients(self, weight_gradient, bias_gradient, weight, bias):
        new_weight = weight
        new_bias = bias
        assert(weight_gradients.shape == self.weight_velocity.shape)
        assert(bias_gradients.shape == self.bias_velocity.shape)

        if self.nesterov:
            self.weight_velocity = self.momentum * self.weight_velocity - \
                self.learning_rate * weight_gradient
            new_weight = weight + self.momentum * \
                self.weight_velocity - self.learning_rate * weight_gradients
            self.bias_velocity = self.momentum * self.bias_velocity - \
                self.learning_rate * bias_gradient
            new_bias = bias + self.momentum * self.bias_velocity - \
                self.learning_rate * bias_gradient
        else:
            self.weight_velocity = self.momentum * self.weight_velocity - \
                self.learning_rate * weight_gradient
            new_weights = weight + self.weight_velocity
            self.bias_velocity = self.momentum * self.bias_velocity - \
                self.learning_rate * bias_gradient
            new_bias = bias + self.bias_velocity

        return (new_weights, new_bias)
