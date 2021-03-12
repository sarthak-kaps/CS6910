import numpy as np
import math


'''
Metric and Loss functions takes 3 parameters,
1) y_true => the true classes (a column vector with values from 0 to max_classes)
2) y_classes => the predicted classes (a column vector with values from 0 to max_classes)
3) y_probab => the predicted probabilites of all class (a matrix with shape (len,max_classes))
'''
# Metrics and Loss Functions


def mse(y_true, y_classes, y_probab):
    return np.mean((y_true - y_classes)**2)


def accuracy(y_true, y_classes, y_probab):
    return np.mean(y_true == y_classes)


def cross_entropy(y_true, y_classes, y_probab):
    c_e = 0
    for i in range(0, len(y_true)):
        c_e += np.log(y_probab[i][y_true[i]])
    return -(c_e / len(y_true))


'''
After adding new metric or loss function , it should be added in this dictionary.
This will take care of adding them to the main code automatically.
'''
metrics = {"mse": mse, "accuracy": accuracy, "cross_entropy": cross_entropy}


'''
Activation functions takes input vector as input.
Note: Take care of the nan values in case of np.exp function
'''
# Activation Functions and their derivatives


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -50, 50)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    t = np.clip(x, -50, 50)
    return (np.exp(t) - np.exp(-t))/(np.exp(t) + np.exp(-t))


def tanh_derivative(x):
    return (1 - tanh(x) ** 2)


def ReLU(x):
    return np.array(list(map(lambda a: np.maximum(0, a), x)))


def ReLU_derivative(x):
    return np.array(list(map(lambda a: 1 if a > 0 else 0, x)))


def linear(x): return x


def linear_derivative(x): return 1


'''
Add all the activation function with their proper names in this dictionary.
This dictionary is used by models.py for getting the functions from strings.
Also, the same name should be used for adding derivatives.
'''
activations = {"sigmoid": sigmoid, "tanh": tanh,
               "ReLU": ReLU, "linear": linear}
activations_derivatives = {
    "sigmoid": sigmoid_derivative, "tanh": tanh_derivative, "ReLU": ReLU_derivative, "linear": linear_derivative}


# Output function

def softmax(x):
    t = np.exp(np.clip(x, -50, 50))
    return t/(np.sum(t, axis=0))


'''
Functions used for initializing weights and biases.
'''
# Initializers
def random(shape): return np.random.random(shape)


def zeros(shape): return np.zeros(shape)


def xavier(shape): return np.random.normal(
    scale=math.sqrt(2/np.sum(shape)), size=shape)


initializers = {"random": random, "zeros": zeros, "xavier": xavier}
