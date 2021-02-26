import numpy as np


# Metrics and Loss Functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


metrics = {"mse": mse, "accuracy": accuracy}

# Activation Functions


def sigmoid(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def ReLU(x):
    return np.array(map(lambda a: max(0, a), x))


activations = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": ReLU}


# Optimizers

optimizers = {}
