import numpy as np
from typing import List
import extras
import optimizers
import copy

'''
Implementation of class Softmax needed for output layer
Need to know when we are at the output layer
For each layer following requiremnts -
      1) For hidden layer - value output by each neuron (h_{i}) 1 <= i <= n, the value that goes into each neuron (a_{i}) and the weight matrix W and the bias vector b
      2) For output layer - the class probability computed y_{i} 1 <= i <= k, the input into each neuron (a_{i}), the weight matrix W and the bias vector b
      ''
'''

layer_count = 0


class Layer:
    """
    Base class for layers.
    Child class must override build(), call(inputs) methods
    """
    built = False
    name = None

    # Used for Backpropogation
    h = None  # Output of this layer
    a = None  # Input of this layer

    # NOTE : I think h should be the output and a the input, try if you could change this

    input_dim = None
    output_dim = None

    def __init__(self, **kwargs):
        self.built = False
        self.fit = False
        self.input_dim = kwargs.get("input_dim")

    def init_w_and_b(self):
        pass

    def build(self):
        assert(self.built != True)
        assert(self.input_dim != None)
        assert(self.output_dim != None)
        self.init_w_and_b()
        self.built = True

    def __call__(self, inputs):
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
        super(Softmax, self).__init__(**kwargs)
        self.output_dim = self.input_dim
        self.name = "Softmax "
        # layer_count += 1

    def build(self):
        self.output_dim = self.input_dim
        self.built = True

    def __call__(self, inputs):
        self.a = inputs
        self.h = extras.softmax(inputs)
        return self.h

    def get_gradient(self, y):
        grad = np.zeros(self.a.shape)
        grad[y] = 1   # assuming y is the true label
        return -(grad - self.h)


class Dense(Layer):
    activation = None
    weight_initializer = None
    bias_initializer = None
    use_bias = False
    weights = None
    biases = None
    activation_fn = None
    l2 = None   

    def __init__(self, units: int, activation="linear", weight_initializer='random', bias_initializer="random", l2 = 0, **kwargs):
        """Creates a dense layer. After initialization the objects acts as a callable.

        Args:
            units (int): Number of neurons
            activation (str, optional): Choice of activation fuction. Choices are ["sigmoid","tanh","ReLU"]. Defaults to "sigmoid".
            # use_bias (bool, optional): Use bias?. Defaults to True.
            kernel_initializer (str, optional): Choice of kernel initializer. Choices are ["random","xavier"]. Defaults to "random".
            bias_initializer (str, optional): Choice of bias initializer. Choices are ["random","zero"]. Defaults to "random".
            l2 = L2 regularization parameter, default 0
        """
        super(Dense, self).__init__(**kwargs)
        self.use_bias = True
        self.output_dim = units
        self.activation_fn = extras.activations.get(activation)
        self.activation_deri_fn = extras.activations_derivatives.get(
            activation)
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = "Dense "
        self.l2 = l2
        # layer_count += 1

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

    def __call__(self, inputs):
        """It is the main computation step.
        Requires input_dim match with the shape of inputs.
        Also, output_dim must be set using the init method.
        Here, inputs is expected to be of size (d,1)

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert(self.built == True)
        assert(self.input_dim != None)
        # assert((self.input_dim,) == tuple(inputs.reshape(1, -1).shape[1:]))
        assert(self.output_dim != None)
        assert(self.activation_fn != None)
        assert(self.activation_deri_fn != None)
        # weights = (out, inp)
        # inputs = (inp, n)
        # bias = (out, 1)
        # (out,n)
        self.a = np.matmul(self.weights, inputs) + self.bias
        self.h = self.activation_fn(self.a)
        return self.h

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

        self.bias = bias_init_fn((self.output_dim, 1))

    def get_gradient(self):
        pass  # not being implemented for now


class Sequential:
    def __init__(self, layers=None):
        self.layers = []
        if layers:
            if(layers[0].input_dim == None):
                raise ValueError(
                    "Input dimension must be set for the first layer.")
            for layer in layers:
                self.add(layer)

    def add(self, layer):
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

    def compile(self, optimizer='rmsprop', loss="mse", metrics: list = ["accuracy"], **kwargs):
        optimizer_fn = extras.optimizers.get(optimizer)
        if optimizer == "SGD":
            optimizer_fn = optimizer.SGD(kwargs)

        self.compile(self, optimizer_fn, loss, metrics)

    def compile(self, optimizer, loss="mse", metrics: list = ["accuracy"]):
        # self.optimizer = extras.optimizers.get(optimizer)
        # if optimizer == "SGD":
        #     # pass appropriate parameters
        #     self.optimizer = optimizers.SGD()
        if optimizer is None:
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

        self.optimizers = []
        # self.optimizer.compile((len(self.layers), ))
        for i in range(0, len(self.layers) - 1) :
            self.optimizers.append(copy.copy(optimizer))
            self.optimizers[i].compile(self.layers[i].get_weights().shape, self.layers[i].get_bias().shape)
    
    def fit(self, X, Y, epochs=100, verbose=1, batch_size = -1,  cold_start=False):
        if not cold_start:
            for layer in self.layers:
                layer.init_w_and_b()
    

        eta = 0.1  # temporary
        for ep in range(epochs):
            outputs = []
            w, b = self.__get_all_parameters()

            weight_grads, bias_grads = [], []
            for i in range(len(self.layers)-1):
                #print(w[i].shape, end = " ")
                weight_grads.append(np.zeros(w[i].shape))
                bias_grads.append(np.zeros(b[i].shape))
    
            w, b = None, None
            for x, y in zip(X, Y):
                x = np.reshape(x, (-1, 1))
                outputs.append(self.__forward(x))
                # Later on we will change X and y to support mini batch and stochastic gradient descent
                w, b = self.__back_propagation(x, y)
                for i in range(0, len(w)):
                    weight_grads[i] += w[i]
                    bias_grads[i] += b[i]

            # Here a call will be made to the optimizer to update the parameters
            # new_weights, new_bias = self.optimizer.apply_gradients(
                # weight_grads, bias_grads, self.__get_all_parameters())

            for i in range(0, len(self.layers)-1):
                old_weights = self.layers[i].get_weights()
                old_bias = self.layers[i].get_bias()
                new_weights, new_bias = self.optimizers[i].apply_gradients(weight_grads[i], bias_grads[i], old_weights, old_bias, ep + 1)
                self.layers[i].set_weights(new_weights)
                self.layers[i].set_bias(new_bias)
            
            if((ep+1) % verbose == 0):
                y_pred = np.argmax(np.array(outputs), axis=-1)
                print(f"Epoch {ep+1}/{epochs} Loss : {self.loss(y, y_pred)}",
                      ", ".join(list(map(lambda f: str(f(y, y_pred)), self.metrics))))
            
        for layer in self.layers:
            layer.flush_io()

    def evaluate(self, X, y):
        output = []
        for x in X:
            x = np.reshape(x, (-1, 1))
            output.append(self.__forward(x))
        y_pred = np.argmax(output, axis=1)
        print(output, y_pred)
        # return (self.loss(y, y_pred),) + tuple(map(lambda f: f(y, y_pred), self.metrics))
        #print(self.loss(y, y_pred),) + tuple(map(lambda f: f(y, y_pred), self.metrics))
        cnt, nan_cnt = 0, 0
        for i in range(0, len(y)) :
            if y[i] != y_pred[i] :
                cnt += 1
            if y_pred[i] == np.nan :
                nan_cnt += 1
        print(cnt / len(X), nan_cnt / len(X))
    def predict(self, X):
        output = self.__forward(X.T)
        return np.argmax(output, axis=-1)

    def predict_proba(self, X):
        return self.__forward(X.T)

    # Mostly done (a bit confused about the matrix multiplications !)
    def __back_propagation(self, X, y):
        n_layers = len(self.layers)
        output_layer = self.layers[n_layers - 1]
        grad_a = output_layer.get_gradient(y)
        weight_grads, bias_grads = [], []
        # the ith index of the gradients vector will contain the gradient with respect to weight, bias for the ith layer, stored as (grad(weight), grad(bias))
        for i in range(n_layers - 2, -1, -1):
            # print(self.layers[i].name)
            # .h is the hidden layer output
            # ak = bk + Wk * hk -1
            # a = (d,1)
            # h = (k,1)
            if i > 0:
                weight_gradient = np.matmul(grad_a, self.layers[i - 1].h.T)
            else:
                #weight_gradient = np.matmul(
                 #   grad_a, np.ones((1, self.layers[i].input_dim)))
                weight_gradient = np.matmul(
                    grad_a, X.T)
            if(self.layers[i].use_bias):
                bias_gradient = grad_a

            # grad_h = (k,1)
            # grad_a = (k,1)
            grad_h = self.layers[i].weights.T @ grad_a
            if i > 0:
                t = np.array(list(
                    map(self.layers[i - 1].activation_deri_fn, self.layers[i - 1].a)))
                grad_a = grad_h * t

            #print(weight_gradient.shape, bias_gradient.shape) 
            
            weight_gradient = weight_gradient + self.l2 * self.layers[i].weights
            
            if self.layers[i].use_bias:
                bias_gradient = bias_gradient + self.l2 * self.layers[i].biases

            weight_grads.append(weight_gradient)
            if(self.layers[i].use_bias):
                bias_grads.append(bias_gradient)
            else:
                bias_grads.append(None)

        weight_grads.reverse()
        bias_grads.reverse()

        return (weight_grads, bias_grads)

    def __forward(self, X):
        n_layers = len(self.layers)
        inp = X
        for i in range(0, n_layers):
            inp = self.layers[i](inp)
        return inp

    def __get_all_parameters(self):
        weights = []
        biases = []
        for i in range(0, len(self.layers)-1):
            weights.append(self.layers[i].get_weights())
            biases.append(self.layers[i].get_bias())
        return (weights, biases)
