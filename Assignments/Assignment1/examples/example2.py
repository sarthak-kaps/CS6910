import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from src import models
from src import optimizers

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

# using the Adam Optimizer
opt = optimizers.Adam(0.005)

# Model compilation
model.compile(optimizer=opt, loss="cross_entropy",
              metrics=["mse", "accuracy"])

# fit model on train and validation
model.fit(train_x, train_y, epochs=10, batch_size=512)

# Evaluate on test dataset, will return metrics
model.evaluate(test_x, test_y)
