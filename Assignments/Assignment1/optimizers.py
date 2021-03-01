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
    velocity = None

    def __init__(self, parameters, learning_rate = None, momentum = 0, nesterov = False):
        if learning_rate is None:
            super(SGD, Optimizer).__init__(0.1)
        else: 
            self.learning_rate = learning_rate
        
        self.nesterov = nesterov
        self.momentum = momentum
        
        if momentum is not None and (momentum > 1 or momentum < 0) :
            raise ValueError("Error : Momentum must lie between 0 and 1 both inclusive")
        
        self.velocity = np.zeros((parameters.shape))

    def apply_gradients(self, gradients, parameters):
        new_parameters = parameters

        assert(gradients.shape == parameters.shape and parameters.shape == self.velocity.shape)

        for i in range(0, len(parameters)):
            for j in range(0, len(parameters[i])):
                if nesterov:
                    self.velocity[i][j] = self.momentum * self.velocity[i][j] - self.learning_rate * gradients[i][j]
                    new_parameters[i][j] = parameters[i][j] + self.momentum * self.velocity[i][j] - self.learning_rate * gradients[i][j]
                else:
                    self.velocity[i][j] = self.momentum * self.velocity[i][j] - self.learning_rate * gradients[i][j]
                    new_parameters[i][j] = parameters[i][j] + self.velocity[i][j]

        return new_parameters

