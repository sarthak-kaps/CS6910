import numpy as np
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

    def build(self):
        self.output_dim = self.input_dim
        self.built = True

    def __call__(self, inputs):
        self.a = inputs
        self.h = extras.softmax(inputs)
        return self.h

    # Done
    def get_gradient(self, y, loss_metric = "cross_entropy"):
        if loss_metric == "cross_entropy" :
            grad = np.zeros(self.a.shape)
            for i in range(len(y)):
                grad[y[i], i] = 1
            return -(grad - self.h)
        elif loss_metric == "mse" :
            grad = np.zeros(self.a.shape)
            for i in range(len(y)):
                y_true_prob = np.zeros(self.output_dim)
                y_true_prob[y[i]] = 1
                y_pred_prob = self.h[:, i]
                grad[:, i] = (y_true_prob - y_pred_prob) * y_pred_prob - ((y_true_prob - y_pred_prob).T @ y_pred_prob) * y_pred_prob
            return grad
        else :
            raise ValueError("Unexpected Loss Metric, got " + loss_metric)


class Dense(Layer):
    activation = None
    weight_initializer = None
    bias_initializer = None
    use_bias = False
    weights = None
    biases = None
    activation_fn = None
    l2 = None

    def __init__(self, units: int, activation="linear", weight_initializer='random', bias_initializer="random", l2=0, **kwargs):
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
        if self.activation_fn is None:
            print("No such activation function - %s, choices are %s" %
                  (activation, ", ".join(extras.activations.keys())))
            return None
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
            raise ValueError("Invalid weight initializer %s. Choices are %s" %
                             (self.weight_initializer, ", ".join(extras.initializers.keys())))

        self.weights = weight_init_fn((self.output_dim, self.input_dim))

    def init_biases(self):
        bias_init_fn = extras.initializers.get(self.bias_initializer)
        if not bias_init_fn:
            raise ValueError("Invalid bias initializer %s. Choices are %s" %
                             (self.bias_initializer, ", ".join(extras.initializers.keys())))

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

    def compile(self, optimizer, loss="cross_entropy", metrics: list = ["accuracy"]):
        # self.optimizer = extras.optimizers.get(optimizer)
        # if optimizer == "SGD":
        #     # pass appropriate parameters
        #     self.optimizer = optimizers.SGD()
        if optimizer is None:
            raise ValueError(
                f"Optimizer {optimizer} not defined. Choices are {extras.optimizers.keys()}.")

        self.loss = extras.metrics.get(loss)
        self.loss_name = loss
        if self.loss is None:
            raise ValueError(
                f"Loss {loss} not defined. Choices are {extras.metrics.keys()}")

        self.metrics = list(map(extras.metrics.get, metrics))
        for met in metrics:
            if met not in extras.metrics.keys():
                raise ValueError(
                    f"Metrics {met} not defined. Choices are {extras.metrics.keys()}")

        self.print_str = "Epoch %d/%d : Loss = %s, " + \
            ", ".join(["%s = %%s" % i for i in metrics])

        for layer in self.layers:
            layer.build()

        self.optimizers = []

        for i in range(0, len(self.layers) - 1):
            self.optimizers.append(copy.copy(optimizer))
            self.optimizers[i].compile(
                self.layers[i].get_weights().shape, self.layers[i].get_bias().shape)

    def fit(self, X, Y, Xval=None, Yval=None, epochs=100, verbose=1, batch_size=-1, cold_start=False, callback_fn=None):
        """Fits the neural network on dataset X and y

        Args:
            X (np.ndarray): Training inputs
            Y (np.ndarray): Training labels
            epochs (int, optional): Number of epochs to train on. Defaults to 100.
            verbose (int, optional): Verbosity for printing the results. Defaults to 1.
            batch_size (int, optional): Processes batch_size inputs together. Defaults to -1.
            cold_start (bool, optional): If set to true, it uses the previously trained weights. Defaults to False.
            callback_fn (function, optional): A callback function which must take (epochs, train_metrics,val_metrics) as inputs. Defaults to None.
                            train_metrics and val_metrics will be passed as tupple.
                            Ex: (loss, metrics1, metrics2, ...)
        """
        if not cold_start:
            for layer in self.layers:
                layer.init_w_and_b()

        if batch_size == -1:
            batch_size = len(X)

        ts = 1
        for ep in range(epochs):
            outputs = []
            #w, b = self.__get_all_parameters()

            
            range_start = 0
            range_end = batch_size

            """
            while range_start < batch_size :
                weight_grads, bias_grads = [], []
                for i in range(len(self.layers)-1):
                    weight_grads.append(np.zeros(w[i].shape))
                    bias_grads.append(np.zeros(b[i].shape))
                X_batch = X[range_start : range_end]
                Y_batch = Y[range_start : range_end]
                w, b = None, None
                for x, y in zip(X_batch, Y_batch):
                    x = np.reshape(x, (-1, 1))
                    outputs.append(self.__forward(x))
                    # Later on we will change X and y to support mini batch and stochastic gradient descent
                    w, b = self.__back_propagation(x, y)
                    for i in range(0, len(w)):
                        weight_grads[i] += w[i]
                        bias_grads[i] += b[i]
            range_start = 0
            range_end = batch_size
            """
            while range_start < batch_size:

                X_batch = X[range_start: range_end]
                Y_batch = Y[range_start: range_end]

                x = X_batch.T
                y = Y_batch.T
                _ = self.__forward(x)
                w, b = self.__back_propagation(x, y)

                for i in range(0, len(self.layers)-1):
                    self.layers[i].weights, self.layers[i].bias = self.optimizers[i].apply_gradients(
                        w[i], b[i], self.layers[i].weights, self.layers[i].bias, ts)

                ts += 1
                range_start += batch_size
                range_end += batch_size
                range_end = min(range_end, len(X))

            if((ep+1) % verbose == 0):
                train_results = self.evaluate(X, Y)
                if Xval is not None and Yval is not None:
                    val_results = self.evaluate(Xval, Yval)
                else:
                    val_results = None
                if(callback_fn != None):
                    callback_fn(ep+1, train_results, val_results)
                print(self.print_str % ((ep+1, epochs,) + train_results))

        if not cold_start:
            for layer in self.layers:
                layer.flush_io()

    def evaluate(self, X, y):
        output = self.__forward(X.T).T
        y_pred = np.argmax(output, axis=-1)
        return (self.loss(y, y_pred, output),) + tuple(map(lambda f: f(y, y_pred, output), self.metrics))

    def predict(self, X):
        output = self.__forward(X.T).T
        return np.argmax(output, axis=-1)

    def predict_proba(self, X):
        return self.__forward(X.T).T

    # Mostly done (a bit confused about the matrix multiplications !)
    # X = (10,n)
    # y = (1,n)
    def __back_propagation(self, X, y):
        n_layers = len(self.layers)
        output_layer = self.layers[n_layers - 1]
        grad_a = output_layer.get_gradient(y, self.loss_name)
        # grad_a = (2,n)
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
                weight_gradient = np.matmul(grad_a, X.T)

            if(self.layers[i].use_bias):
                bias_gradient = np.reshape(np.sum(grad_a, axis=-1), (-1, 1))

            grad_h = self.layers[i].weights.T @ grad_a

            if i > 0:
                t = np.array(list(
                    map(self.layers[i - 1].activation_deri_fn, self.layers[i - 1].a)))
                grad_a = grad_h * t

            weight_gradient = weight_gradient + \
                self.layers[i].l2 * self.layers[i].weights

            if self.layers[i].use_bias:
                bias_gradient = bias_gradient + \
                    self.layers[i].l2 * self.layers[i].bias

            #print(weight_gradient.shape, bias_gradient.shape)
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
