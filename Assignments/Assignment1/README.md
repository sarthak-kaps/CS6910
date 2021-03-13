# ASSIGNMENT 1

### _Neural Networks_

- Our interface is heavily inspired from keras interface for the same.

- Our constructed modules have been aimed to maintain the same API and are easy to add more layers and more models to it.

- You can find various examples in examples directory. Also, all the wandb scripts are located in runners directory.

- Currently, the source code consists of the following files -

#### models.py

- Contains all the classes and functions for the main neural network models and the layers.
- Following classes are supported currently -

  - `Sequential` (Neural network base)
    - Contains all the necessary functions with keras-like interface.
  - `Layer` (parent): (Base class provided minimal interface)
    - `Dense` (inherits Layer class)
    - `Softmax` (inherits Layer class)

- Sequential class supports the following functions -
  - `add(layer)` : adds a layer object to the neural network
  - `compile(optimizer="adam", loss="cross_entropy", metrics: list = ["accuracy"])` : compiles the model by building layers and initializing weights.
  - `fit(X, Y, Xval=None, Yval=None, epochs=100, verbose=1, batch_size=-1, cold_start=False, callback_fn=None)` : updates weights and biases of each layer by fitting on (X , Y) dataset.
  - `evaluate(X, y)` : Evaluates the model on (X, Y) dataset
  - `predict(X)` : Returns the predicted classes on X dataset
  - `predict_proba(X)` : Returns the predicted probability of each class on X dataset.

#### optimizers.py

- Contains the implementation of 6 different optimizers.
- Also, the interface is fixed by a generic class `Optimizer`, which is inherited by each individual optimizer classes.
- Currently supported optimizers inclue
  - `SGD`
  - `Nesterov`
  - `Momentum`
  - `RMSprop`
  - `Adam`
  - `Nadam`
- Each of the optimzers can be incorporated easily with the models created.
- New optimizers can be easily added by using the same interface defined by `Optimizer` class.

#### extras.py

- This file contains all the metrices, loss functions activation functions, and initialization functions implementations.
- For each of the above types, a dictionary is defined, which is integrated with models.py in such a way that whenever a new function is added, just a function pointer needs to be stored in this dictionary.
- Currently supported functions are
  - metrices - [```mse```,```accuracy```,```cross_entropy```]
  - loss functions - [```mse```,```accuracy```,```cross_entropy```]
  - activations - [```sigmoid```,```ReLU```,```tanh```]
  - initializers - [```random```,```xavier```,```zeros```]

#### Examples

**Note:** The commands to run words only from the root directory, i.e., Assignment1

###### example1.py

```python
import numpy as np
from keras.datasets import mnist
from Assignment1.src import models

# Loading dataset and names
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# normalization
train_x = np.reshape(train_x, (-1, 784))/255.0
test_x = np.reshape(test_x, (-1, 784))/255.0

# Main model configuration
model = models.Sequential()
model.add(models.Dense(128, "ReLU", input_dim=784))
model.add(models.Dense(128, "ReLU"))
model.add(models.Dense(10, "ReLU"))
model.add(models.Softmax())

# Model compilation
model.compile(optimizer="adam", loss="cross_entropy",
              metrics=["mse", "accuracy"])

# fit model on train and validation
model.fit(train_x, train_y, epochs=10)
```

- **To run** : `python -m Assignment1.examples.example1`

###### example2.py

```python
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from Assigment1.src import models

# List of metrices to be checked while fitting
metrics_list = ["mse", "accuracy", "cross_entropy"]

# Loading dataset and names
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# normalization
train_x = np.reshape(train_x, (-1, 784))/255.0
test_x = np.reshape(test_x, (-1, 784))/255.0

# Splitting train dataset for validation
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.1, random_state=0, shuffle=True)

# Main model configuration
model = models.Sequential()
model.add(models.Dense(128, "ReLU", input_dim=784))
model.add(models.Dense(128, "ReLU"))
model.add(models.Dense(10, "ReLU"))
model.add(models.Softmax())


# Model compilation
model.compile(optimizer="adam", loss="cross_entropy",
              metrics=["mse", "accuracy"])

# fit model on train and validation
model.fit(train_x, train_y, epochs=10, batch_size=512)

# Evaluate on test dataset
model.evaluate(test_x, test_y)

```

- **To run** : `python -m Assignment1.examples.example2`
