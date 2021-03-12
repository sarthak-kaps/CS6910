import numpy as np
from . import extras
from . import optimizers
import copy


class Layer:
    """
    Base class for layers.
    Child class must override build(self) and __call__(self,inputs) methods
    It does no computation, just provides an uniform interface.
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
    """Softmax layer implementation. After initialization, the objects acts as a callable.
    It just applies the softmax function on the input.
    No weights and/or biases are used.
    """

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

    def get_gradient(self, y, loss_metric="cross_entropy"):
        if loss_metric == "cross_entropy":
            grad = np.eye(self.output_dim)[y].T
            return -(grad - self.h)
        elif loss_metric == "mse":
            grad = np.zeros(self.a.shape)
            for i in range(len(y)):
                y_true_prob = np.zeros(self.output_dim)
                y_true_prob[y[i]] = 1
                y_pred_prob = self.h[:, i]
                grad[:, i] = (y_true_prob - y_pred_prob) * y_pred_prob - \
                    ((y_true_prob - y_pred_prob).T @ y_pred_prob) * y_pred_prob
            return -grad
        else:
            raise ValueError("Unexpected Loss Metric, got " + loss_metric)


class Dense(Layer):
    """Implementation of dense layer.
    computes output = activation(w*input + b)
    Once initialized, it acts as a callable.
    """
    activation = None
    weight_initializer = None
    bias_initializer = None
    weights = None
    bias = None
    activation_fn = None
    l2 = None

    def __init__(self, units: int, activation="linear", weight_initializer='xavier', bias_initializer="random", l2=0, **kwargs):
        """Creates a dense layer. After initialization the objects acts as a callable.

        Args:
            units (int): Number of neurons
            activation (str, optional): Choice of activation fuction. Choices are ["sigmoid","tanh","ReLU"]. Defaults to "sigmoid".
            kernel_initializer (str, optional): Choice of kernel initializer. Choices are ["random","xavier"]. Defaults to "xavier".
            bias_initializer (str, optional): Choice of bias initializer. Choices are ["random","zero"]. Defaults to "random".
            l2 = L2 regularization parameter, default 0
        """
        super(Dense, self).__init__(**kwargs)
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
        """
        assert(self.built == True)
        assert(self.input_dim != None)
        assert(self.output_dim != None)
        assert(self.activation_fn != None)
        assert(self.activation_deri_fn != None)
        self.a = np.matmul(self.weights, inputs) + self.bias
        self.h = self.activation_fn(self.a)
        return self.h

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
    """Provides implementation of Sequential model.
    Multiple layers can be added of varies sizes and activations.
    NOTE: For now, softmax layer must be used as an output layer.
    """

    def __init__(self, layers=None):
        """Initializes Sequential model for layers.

        Args:
            layers (list of (Dense or Softmax), optional): List of layers to be added. Layers can also be added using add function. Defaults to None.

        Raises:
            ValueError: Input dimension must be set for the first layer
        """
        self.layers = []
        if layers:
            if(layers[0].input_dim == None):
                raise ValueError(
                    "Input dimension must be set for the first layer.")
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """appends a layer to the sequential model.
        Note: it computes the input and output shape automatically.
        Except, for the first layer. Input dimension must be passed.

        Args:
            layer (Dense of Softmax): Layer to be added in the function

        """
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

    def compile(self, optimizer="adam", loss="cross_entropy", metrics: list = ["accuracy"]):
        """Compiles the sequential model. Build each layer by initializing weights.
        Also, initializes added optimizers and loss functions.

        Args:
            optimizer (Any child of Optimizer class or a string): [description]
            loss (str, optional): Type of loss function used. mse or cross_entropy. Defaults to "cross_entropy".
            metrics (list, optional): Type of metrices to be printed. Defaults to ["accuracy"].

        """
        if type(optimizer) == str:
            optimizer = optimizers.optimizers.get(optimizer)
            if optimizer is None:
                raise ValueError(
                    f"Optimizer {optimizer} not defined. Choices are {optimizers.optimizers.keys()}.")

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
                self.layers[i].weights.shape, self.layers[i].bias.shape)

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

        # Check for cold start
        if not cold_start:
            for layer in self.layers:
                layer.init_w_and_b()

        # Init batch size
        if batch_size == -1:
            batch_size = len(X)

        # main loop of training
        ts = 1
        for ep in range(epochs):

            # divide the input in batches
            range_start = 0
            range_end = batch_size

            # loop of batches.
            while range_start < len(X):
                # Create the batches
                X_batch = X[range_start: range_end]
                Y_batch = Y[range_start: range_end]

                # Set the proper shape
                x = X_batch.T
                y = Y_batch.T

                # Forward propogation
                _ = self.__forward(x)

                # Backward propogation
                w, b = self.__back_propagation(x, y)

                # Set the weights and biases
                for i in range(0, len(self.layers)-1):
                    self.layers[i].weights, self.layers[i].bias = self.optimizers[i].apply_gradients(
                        w[i], b[i], self.layers[i].weights, self.layers[i].bias, ts)

                # increment the batches
                ts += 1
                range_start += batch_size
                range_end += batch_size
                range_end = min(range_end, len(X))

            # Print progress and run callback
            if((ep+1) % verbose == 0):
                train_results = self.evaluate(X, Y)
                if Xval is not None and Yval is not None:
                    val_results = self.evaluate(Xval, Yval)
                else:
                    val_results = None
                if(callback_fn != None):
                    callback_fn(ep+1, train_results, val_results)
                print(self.print_str % ((ep+1, epochs,) + train_results))

        # check for cold start
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

    def __back_propagation(self, X, y):
        # Initialize variables
        n_layers = len(self.layers)
        output_layer = self.layers[n_layers - 1]
        weight_grads, bias_grads = [], []

        # Get the gradient of the output layer.
        grad_a = output_layer.get_gradient(y, self.loss_name)
        for i in range(n_layers - 2, -1, -1):
            if i > 0:
                weight_gradient = np.matmul(grad_a, self.layers[i - 1].h.T)
            else:
                weight_gradient = np.matmul(grad_a, X.T)

            bias_gradient = np.reshape(np.sum(grad_a, axis=-1), (-1, 1))

            grad_h = self.layers[i].weights.T @ grad_a

            if i > 0:
                t = np.array(list(
                    map(self.layers[i - 1].activation_deri_fn, self.layers[i - 1].a)))
                grad_a = grad_h * t

            weight_gradient = weight_gradient + \
                self.layers[i].l2 * self.layers[i].weights

            bias_gradient = bias_gradient + \
                self.layers[i].l2 * self.layers[i].bias

            weight_grads.append(weight_gradient)
            bias_grads.append(bias_gradient)

        weight_grads.reverse()
        bias_grads.reverse()

        return (weight_grads, bias_grads)

    def __forward(self, X):
        n_layers = len(self.layers)
        inp = X
        for i in range(0, n_layers):
            inp = self.layers[i](inp)
        return inp
