import numpy as np


class Optimizer:
    learning_rate = None

    # the appropriate child class should set the learning rate as suited to the optimization
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    # child class must implement this function
    def apply_gradients(self, gradients, weights):
        pass

# Implements Simple Gradient Descent, Momentum Based Gradient Descent, Nesterov Accelerated Gradient Descent
class SGD(Optimizer):
    nesterov = False
    momentum = None
    weight_velocity = None
    bias_velocity = None

    def __init__(self, learning_rate=0.01, momentum=0, nesterov=False):
        super(SGD, self).__init__(learning_rate)

        self.nesterov = nesterov
        self.momentum = momentum

        if momentum is not None and (momentum > 1 or momentum < 0):
            raise ValueError(
                "Error : Momentum must lie between 0 and 1 both inclusive")

    def compile(self, weight_shape, bias_shape):
        self.weight_velocity = np.zeros(weight_shape)
        self.bias_velocity = np.zeros(bias_shape)

    def apply_gradients(self, weight_gradient, bias_gradient, weight, bias, epoch):
        new_weight = weight
        new_bias = bias
        assert(weight_gradient.shape == self.weight_velocity.shape)
        assert(bias_gradient.shape == self.bias_velocity.shape)

        if self.nesterov:
            self.weight_velocity = self.momentum * self.weight_velocity - \
                self.learning_rate * weight_gradient
            new_weight = weight + self.momentum * \
                self.weight_velocity - self.learning_rate * weight_gradient
            self.bias_velocity = self.momentum * self.bias_velocity - \
                self.learning_rate * bias_gradient
            new_bias = bias + self.momentum * self.bias_velocity - \
                self.learning_rate * bias_gradient
        else:
            self.weight_velocity = self.momentum * self.weight_velocity - \
                self.learning_rate * weight_gradient
            new_weight = weight + self.weight_velocity
            self.bias_velocity = self.momentum * self.bias_velocity - \
                self.learning_rate * bias_gradient
            new_bias = bias + self.bias_velocity

        return (new_weight, new_bias)

# Implements RMS propogation Gradient Descent
class RMSprop(Optimizer):
    def __init__(self, learning_rate = 0.001, beta = 0.9, eps = 1e-7):
        super(RMSprop, self).__init__(learning_rate)
        self.beta = beta 
        self.eps = eps


    def compile(self, weight_shape, bias_shape):
        self.weight_velocity = np.zeros(weight_shape)
        self.bias_velocity = np.zeros(bias_shape)


    def apply_gradients(self, weight_gradient, bias_gradient, weight, bias, epoch):
        new_weight = weight
        new_bias = bias

        assert(weight_gradient.shape == self.weight_velocity.shape)
        assert(bias_gradient.shape == self.bias_velocity.shape)

        self.weight_velocity = self.beta * self.weight_velocity + (1 - self.beta) * np.square(weight_gradient)
        self.bias_velocity = self.beta * self.bias_velocity + (1 - self.beta) * np.square(bias_gradient)

        new_weight = weight - self.learning_rate * np.reciprocal(np.sqrt(self.weight_velocity + self.eps)) * weight_gradient
        new_bias = bias - self.learning_rate * np.reciprocal(np.sqrt(self.bias_velocity + self.eps)) * bias_gradient

        return (new_weight, new_bias)

class Adam(Optimizer):
    def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-7):
        super(Adam, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def compile(self, weight_shape, bias_shape): 
        self.weight_momentum = np.zeros(weight_shape)
        self.bias_momentum = np.zeros(bias_shape)
        self.weight_velocity = np.zeros(weight_shape)
        self.bias_velocity = np.zeros(bias_shape)

    def apply_gradients(self, weight_gradient, bias_gradient, weight, bias, epoch):
        new_weight = weight
        new_bias = bias
 
        assert(weight_gradient.shape == self.weight_velocity.shape)
        assert(bias_gradient.shape == self.bias_velocity.shape)
        
        assert(weight_gradient.shape == self.weight_momentum.shape)
        assert(bias_gradient.shape == self.bias_momentum.shape)

        beta_1 = self.beta_1
        beta_2 = self.beta_2

        self.weight_momentum = beta_1 * self.weight_momentum + (1 - beta_1) * weight_gradient
        self.bias_momentum = beta_1 * self.bias_momentum + (1 - beta_1) * bias_gradient

        self.weight_velocity = beta_2 * self.weight_velocity + (1 - beta_2) * np.square(weight_gradient)
        self.bias_velocity = beta_2 * self.bias_velocity + (1 - beta_2) * np.square(bias_gradient)

        weight_momentum_hat = self.weight_momentum * (1 / (1 - np.power(beta_1, epoch)))
        bias_momentum_hat = self.bias_momentum * (1 / (1 - np.power(beta_1, epoch)))

        weight_velocity_hat = self.weight_velocity * (1 / (1 - np.power(beta_2, epoch)))
        bias_velocity_hat = self.bias_velocity * (1 / (1 - np.power(beta_2, epoch)))

        new_weight = weight - self.learning_rate * np.reciprocal(np.sqrt(weight_velocity_hat + self.eps)) * weight_momentum_hat
        new_bias = bias - self.learning_rate * np.reciprocal(np.sqrt(bias_velocity_hat + self.eps)) * bias_momentum_hat

        return (new_weight, new_bias)

class Nadam(Optimizer) :
    
    def __init__(self, learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-7):
        super(Nadam, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def compile(self, weight_shape, bias_shape): 
        self.weight_momentum = np.zeros(weight_shape)
        self.bias_momentum = np.zeros(bias_shape)
        self.weight_velocity = np.zeros(weight_shape)
        self.bias_velocity = np.zeros(bias_shape)

    def apply_gradients(self, weight_gradient, bias_gradient, weight, bias, epoch):
        new_weight = weight
        new_bias = bias
 
        assert(weight_gradient.shape == self.weight_velocity.shape)
        assert(bias_gradient.shape == self.bias_velocity.shape)
        
        assert(weight_gradient.shape == self.weight_momentum.shape)
        assert(bias_gradient.shape == self.bias_momentum.shape)

        beta_1 = self.beta_1
        beta_2 = self.beta_2

        self.weight_momentum = beta_1 * self.weight_momentum + (1 - beta_1) * weight_gradient
        self.bias_momentum = beta_1 * self.bias_momentum + (1 - beta_1) * bias_gradient

        self.weight_velocity = beta_2 * self.weight_velocity + (1 - beta_2) * np.square(weight_gradient)
        self.bias_velocity = beta_2 * self.bias_velocity + (1 - beta_2) * np.square(bias_gradient)

        weight_momentum_hat = self.weight_momentum * (1 / (1 - np.power(beta_1, epoch)))
        bias_momentum_hat = self.bias_momentum * (1 / (1 - np.power(beta_1, epoch)))

        weight_velocity_hat = self.weight_velocity * (1 / (1 - np.power(beta_2, epoch)))
        bias_velocity_hat = self.bias_velocity * (1 / (1 - np.power(beta_2, epoch)))
        
        weight_momentum_hat_nu = beta_1 * weight_momentum_hat
        bias_momentum_hat_nu = beta_1 * bias_momentum_hat

        weight_gradient_nu = ((1 - beta_1) / (1 - np.power(beta_1, epoch))) * weight_gradient
        bias_gradient_nu = ((1 - beta_1) / (1 - np.power(beta_1, epoch))) * bias_gradient

        new_weight = weight - self.learning_rate * np.reciprocal(np.sqrt(weight_velocity_hat + self.eps)) * (weight_momentum_hat_nu + weight_gradient_nu)
        new_bias = bias - self.learning_rate * np.reciprocal(np.sqrt(bias_velocity_hat + self.eps)) * (bias_momentum_hat_nu + bias_gradient_nu)

        
        return (new_weight, new_bias)

    

