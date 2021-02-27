import numpy as np


# Metrics and Loss Functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

'''
Assume classes are 0 to k - 1
y_true -> list of classes in the training data
y_pred -> list that contains the class probability prediction for each of the k class for the input data 
class probabilities are zero indexed 
'''
def cross_entropy(y_true, y_pred):
    c_e = 0
    for i in range(0, len(y_true)):
        c_e += y_pred[i][y_true[i]]
    return c_e / len(y_true)


metrics = {"mse": mse, "accuracy": accuracy, "cross_entropy" : cross_entropy}

# Activation Functions


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return (1 - tanh(x) ** 2)

def ReLU(x):
    return np.array(map(lambda a: max(0, a), x))


activations = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": ReLU}
activations_derivatives = {"sigmoid" : sigmoid_derivative, "tanh" : tanh_derivative}


# Optimizers

optimizers = {}


y_true = []
y_pred = []

n = int(input())
k = int(input())

for i in range(0, n):
    x = int(input())
    y_true.append(x)
for i in range(0, n):
    pred = []
    for j in range(0, k):
        x = float(input())
        pred.append(x)
    y_pred.append(pred)

print(cross_entropy(y_true, y_pred))
