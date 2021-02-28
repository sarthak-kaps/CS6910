import numpy as np
import pandas as pd
import tensorflow.keras as keras
import math
import extras



''' 
Implementation of class Softmax needed for output layer
Need to know when we are at the output layer
For each layer following requiremnts -
      1) For hidden layer - value output by each neuron (h_{i}) 1 <= i <= n, the value that goes into each neuron (a_{i}) and the weight matrix W and the bias vector b
      2) For output layer - the class probability computed y_{i} 1 <= i <= k, the input into each neuron (a_{i}), the weight matrix W and the bias vector b  
      ''
'''

class Dense:
    def __init__(self, units: int, activation="sigmoid", use_bias=True, weight_initializer='random', bias_initializer="random", **kwargs):
        """Creates a dense layer

        Args:
            units (int): Number of neurons
            activation (str, optional): Choice of activation fuction. Choices are ["sigmoid","tanh","ReLU"]. Defaults to "sigmoid".
            use_bias (bool, optional): Use bias?. Defaults to True.
            kernel_initializer (str, optional): Choice of kernel initializer. Choices are ["random","xavier"]. Defaults to "random".
            bias_initializer (str, optional): Choice of bias initializer. Choices are ["random","zero"]. Defaults to "random".
        """
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.input_dim = kwargs.get("input_dim")
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.built = False
        return self

    def build(self):
        self.init_weights()
        self.init_biases()
        self.built = True

    def init_weights(self):
        if self.weight_initializer == "random":
            self.weights = np.random.random((self.units, self.input_dim))
        elif self.weight_initializer == "xavier":
            self.weights = np.random.normal(
                scale=math.sqrt(2/(self.units + self.input_dim)))
        else:
            raise ValueError("Invalid weight initializer %s" %
                             (self.weight_initializer))

    def init_biases(self):
        if self.bias_initializer == "random":
            self.bias_initializer = np.random.random((self.units,))
        elif self.bias_initializer == "zero":
            self.bias_initializer = np.zeros((self.units,))
        else:
            raise ValueError("Invalid bias initializer %s" %
                             (self.bias_initializer))

    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    def get_weights(self): return self.weights

    def set_bias(self, bias): self.bias = bias


    def get_bias(self): return self.bias


class Sequential:
    def __init__(self, layers: list = None):
        self.layers = []
        if layers:
            for layer in layers:
                self.add(layer)
        return self

    def add(self, layer: Dense):
        if layer.input_dim and (len(self.layers) == 0):
            self.input_dim = layer.input_dim
        elif layer.input_dim and (len(self.layers) != 0):
            if(self.layers[-1].units != layer.input_dim):
                raise ValueError("Layer shape mismatch. %d != %d" %
                                 (self.layers[-1].units, layer.input_dim))
        elif len(self.layers) == 0:
            raise ValueError("Input shape required for the first layer")
        else:
            layer.input_dim = self.layers[-1].units

        self.layers.append(layer)

    def compile(self, optimizer='rmsprop', loss="mse", metrics: list = ["accuracy"]):
        self.optimizer = extras.optimizers.get(optimizer)
        if self.optimizer is None:
            raise ValueError(
                f"Optimizer {optimizer} not defined. Choices are {extras.optimizers.keys()}.")

        self.loss = extras.metrics.get(loss)
        if self.loss is None:
            raise ValueError(
                f"Loss {loss} not defined. Choices are {extras.metrics.keys()}")

        self.metrics = list(map(extras.metrics.get, metrics))
        for met in metrics:
            if met not in extras.metrics.keys():
                raise ValueError(
                    f"Metrics {met} not defined. Choices are {extras.metrics.keys()}")

        for layer in self.layers:
            layer.build()

    def fit(self, X, y, epochs=10):
        pass

    def evaluate(self, X, y):
        pass

    def predict(self, X):
        pass


    # Assumption all vectors are numpy arrays and by default column vectors
    def back_propagation(parameters):    # parameters contains the pair of W, b for each layer at which the gradient is to be computed
        num_layers = len(self.layers)
        output_layer = self.layers[num_layers - 1]
        grad_a = output_layer.get_gradient()
        gradients = [] # the ith index of the gradients vector will contain the gradient with respect to weight, bias for the ith layer, stored as (grad(weight), grad(bias))
        for i in range(num_layers - 1, 0, -1):
            weight_gradient = grad_a @ self.layers[i - 1].h.T   # .h is the hidden layer output
            bias_gradient = grad_a
            grad_h = parameters[i][0].T @ grad_a
            grad_a = grad_h * np.array([extras.activations_derivatives[self.activation](e) for e in self.layers[i - 1].a]).T  # .a is the hidden layer input
            gradients.append((weight_gradient, bias_gradient))
        return gradients

    # function to find gradient with respect to the parameters, please pass all the parameters, you can later use the ones that are important
    def find_gradient(parameters):
        # some preprocessing maybe needed
        return back_propagation(parameters)
    
    # I am writing the pseude code for simple gradient_descent here
    # Arguments for gradient_descent can be the train data passed to fit
    def gradient_descent():
        '''
        Until convergence or max_iterations
        Run forward propagation  ---> appropriate function call implemented by Rudra
        Expected result -> at each layer the required values stated at line 10 should be set

        Run backpropagation ---> appropriate function call implemented by Sarthak
        Expected result -> for each layer the parameters will be set
        '''
        pass



