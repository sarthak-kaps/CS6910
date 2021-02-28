import numpy as np
from typing import List
import extras


'''
Implementation of class Softmax needed for output layer
Need to know when we are at the output layer
For each layer following requiremnts -
      1) For hidden layer - value output by each neuron (h_{i}) 1 <= i <= n, the value that goes into each neuron (a_{i}) and the weight matrix W and the bias vector b
      2) For output layer - the class probability computed y_{i} 1 <= i <= k, the input into each neuron (a_{i}), the weight matrix W and the bias vector b
      ''
'''


class Layer:
    """
    Base class for layers.
    Child class must override build(), call(inputs) methods
    """
    built = False

    # Used for Backpropogation
    h = None  # Input of this layer
    a = None  # Output of this layer

    input_dim = None
    output_dim = None

    def __init__(self, input_dim=None):
        self.built = False
        self.fit = False
        self.input_dim = input_dim

    def init_w_and_b(self):
        pass

    def build(self):
        assert(self.built != True)
        assert(self.input_dim != None)
        assert(self.output_dim != None)
        self.init_w_and_b()
        self.built = True

    def call(self, inputs):
        self.h = inputs
        self.a = inputs
        return inputs

    def flush_io(self):
        self.h = None
        self.a = None


class Softmax(Layer):
    def __init__(self, **kwargs):
        """Creates a Softmax layer.  After initialization the objects acts as a callable.
        Extra params are input_dim.
        """
        super(Softmax, self).__init__(kwargs)
        self.output_dim = self.input_dim

    def call(self, inputs):
        self.h = inputs
        self.a = extras.softmax(inputs)
        return self.a


class Dense(Layer):
    activation = None
    weight_initializer = None
    bias_initializer = None
    use_bias = False
    weights = None
    biases = None
    activation_fn = None

    def __init__(self, units: int, use_bias=True, activation="linear", weight_initializer='random', bias_initializer="random", **kwargs):
        """Creates a dense layer. After initialization the objects acts as a callable.

        Args:
            units (int): Number of neurons
            activation (str, optional): Choice of activation fuction. Choices are ["sigmoid","tanh","ReLU"]. Defaults to "sigmoid".
            use_bias (bool, optional): Use bias?. Defaults to True.
            kernel_initializer (str, optional): Choice of kernel initializer. Choices are ["random","xavier"]. Defaults to "random".
            bias_initializer (str, optional): Choice of bias initializer. Choices are ["random","zero"]. Defaults to "random".
        """
        super(Dense, self).__init__(kwargs)
        self.use_bias = use_bias
        self.output_dim = units
        self.activation_fn = extras.activations.get(activation)
        self.activation_deri_fn = extras.activations_derivatives.get(
            activation)
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        return self

    def init_w_and_b(self):
        self.init_biases()
        self.init_weights()

    def build(self):
        """Build initializes weights and biases.
        It requires that input_dim and output_dim are set.
        Also, build can be called just once.
        """
        assert(self.built != True)
        assert(self.input_dim != None)
        assert(self.output_dim != None)
        self.init_w_and_b()
        self.built = True

    def call(self, inputs):
        """It is the main computation step.
        Requires input_dim match with the shape of inputs.
        Also, output_dim must be set using the init method

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert(self.built == True)
        assert(self.input_dim != None)
        assert(self.input_dim == inputs.shape[1:])
        assert(self.output_dim != None)
        assert(self.activation_fn != None)
        assert(self.activation_deri_fn != None)
        self.h = np.matmul(self.weights, inputs) + self.biases
        self.a = self.activation_fn(self.h)
        return self.a

    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    def get_weights(self): return self.weights

    def set_bias(self, bias):
        if(self.use_bias):
            self.bias = bias

    def get_bias(self):
        if(self.use_bias):
            return self.bias
        else:
            return None

    def init_weights(self):
        weight_init_fn = extras.initializers.get(self.weight_initializer)
        if not weight_init_fn:
            raise ValueError("Invalid weight initializer %s" %
                             (self.weight_initializer))

        self.weights = weight_init_fn((self.output_dim, self.input_dim))

    def init_biases(self):
        bias_init_fn = extras.initializers.get(self.bias_initializer)
        if not bias_init_fn:
            raise ValueError("Invalid bias initializer %s" %
                             (self.bias_initializer))

        self.bias = bias_init_fn((self.output_dim,))


class Sequential:
    def __init__(self, layers: List[Layer] = None):
        self.layers = List[Layer]
        if layers:
            if(layers[0].input_dim == None):
                raise ValueError(
                    "Input dimension must be set for the first layer.")
            for layer in layers:
                self.add(layer)
        return self

    def add(self, layer: Layer):
        if layer.input_dim and (len(self.layers) == 0):
            self.input_dim = layer.input_dim
        elif layer.input_dim and (len(self.layers) != 0):
            if(self.layers[-1].output_dim != layer.input_dim):
                raise ValueError("Layer shape mismatch. %d != %d" %
                                 (self.layers[-1].output_dim, layer.input_dim))
        elif len(self.layers) == 0:
            raise ValueError("Input shape required for the first layer")
        else:
            layer.input_dim = self.layers[-1].output_dim

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

    def fit(self, X, y, epochs=10, verbose=1, cold_start=False):
        if not cold_start:
            for layer in self.layers:
                layer.init_w_and_b()

        for i in range(epochs):
            output = self.__forward(X)
            self.__back_propagation(y)
            if((i+1) % verbose == 0):
                y_pred = np.argmax(output, axis=-1)
                print(f"Epoch {i+1}/{epochs} Loss : {self.loss(y, y_pred)}",
                      ", ".join(list(map(lambda f: f(y, y_pred), self.metrics))))

        for layer in self.layers:
            layer.flush_io()

    def evaluate(self, X, y):
        output = self.__forward(X)
        y_pred = np.argmax(output, axis=-1)
        return (self.loss(y, y_pred),) + tuple(map(lambda f: f(y, y_pred), self.metrics))

    def predict(self, X):
        output = self.__forward(X)
        return np.argmax(output, axis=-1)

    def predict_proba(self, X):
        return self.__forward(X)

    # Where is bias being used in this function
    # Also, as I am storing weights, biases, inputs and outputs in the layer.
    # I removed parameters from this function
    def __back_propagation(self, y):
        num_layers = len(self.layers)
        output_layer = self.layers[num_layers - 1]
        grad_a = output_layer.get_gradient(y)
        # the ith index of the gradients vector will contain the gradient with respect to weight, bias for the ith layer, stored as (grad(weight), grad(bias))
        for i in range(num_layers - 1, 0, -1):
            # .h is the hidden layer output
            weight_gradient = grad_a @ self.layers[i - 1].h.T
            self.layers[i-1].weights -= weight_gradient
            if(self.layers[i-1].use_bias):
                bias_gradient = grad_a
                self.layers[i-1].bias -= bias_gradient
            grad_h = self.layers[i-1].weights.T @ grad_a
            grad_a = grad_h * \
                np.array(
                    map(self.layers[i-1].activation_deri_fn, self.layers[i-1].a)).T

    def __forward(self, X):
        n_layers = len(self.layers)
        inp = X
        for i in range(0, n_layers-1):
            inp = self.layers[i](inp)
        return inp

    # function to find gradient with respect to the parameters, please pass all the parameters, you can later use the ones that are important
    # def find_gradient(self, parameters, y):
    #     # some preprocessing maybe needed
    #     return self.__back_propagation(parameters, y)
