import numpy as np
from keras.datasets import mnist
from src import models
from src import optimizers

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

# using the Adam Optimizer
opt = optimizers.Adam(0.005)

# Model compilation
model.compile(optimizer=opt, loss="cross_entropy",
              metrics=["mse", "accuracy"])

# fit model on train and validation
model.fit(train_x, train_y, epochs=10)
