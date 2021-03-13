import numpy as np
import wandb
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

from ..src import models, optimizers

np.random.seed(0)


# Default dictionary for hyperparams
configs = [
    dict(
        hidden_layer_size=128,
        num_hidden_layers=4,
        learning_rate=1e-3,
        epochs=12,
        l2=1e-3,
        optimizer="nadam",
        batch_size=256,
        weight_init="xavier",
        activation="ReLU",
        loss="cross_entropy"
    ), dict(
        hidden_layer_size=128,
        num_hidden_layers=4,
        learning_rate=1e-3,
        epochs=12,
        l2=1e-3,
        optimizer="nadam",
        batch_size=512,
        weight_init="xavier",
        activation="ReLU",
        loss="mse"
    ), dict(
        hidden_layer_size=64,
        num_hidden_layers=4,
        learning_rate=5e-3,
        epochs=10,
        l2=1e-4,
        optimizer="nadam",
        batch_size=512,
        weight_init="xavier",
        activation="ReLU",
        loss="cross_entropy"
    )]

# List of metrices to be checked while fitting
metrics_list = ["mse", "accuracy", "cross_entropy"]

# Callback function to log metrices while fitting


def callback(eps, train_metrics, val_metrics):
    # print(metrics)
    wandb.log({"train_loss": train_metrics[0], "train_mse": train_metrics[1],
               "train_accuracy": train_metrics[2], "train_cross_entropy": train_metrics[3],
               "val_loss": val_metrics[0], "val_mse": val_metrics[1], "val_accuracy": val_metrics[2],
               "val_cross_entropy": val_metrics[3]
               })


# Wandb initialization
wandb.init(config=configs[2],
           project="assignment1", name="mnist_run_3")
config = wandb.config

# Loading dataset and names
(train_x, train_y), (test_x, test_y) = mnist.load_data()


# normalization
train_x = np.reshape(train_x, (-1, 784))/255.0
test_x = np.reshape(test_x, (-1, 784))/255.0


# Splitting train dataset for validation
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.1, random_state=0, shuffle=True)

# Main model configuration
model = models.Sequential()
for i in range(config.num_hidden_layers - 1):
    if(i == 0):
        model.add(models.Dense(config.hidden_layer_size,
                               activation=config.activation, weight_initializer=config.weight_init, l2=config.l2, input_dim=784))
    else:
        model.add(models.Dense(config.hidden_layer_size,
                               activation=config.activation, weight_initializer=config.weight_init, l2=config.l2))
model.add(models.Dense(10, activation=config.activation,
                       weight_initializer=config.weight_init, l2=config.l2))
model.add(models.Softmax())


# Model compilation
opt = optimizers.optimizers[config.optimizer](config.learning_rate)
model.compile(opt, loss=config.loss, metrics=metrics_list)

# fit model on train and validation
model.fit(train_x, train_y, val_x, val_y, epochs=config.epochs, verbose=1,
          batch_size=config.batch_size, callback_fn=callback)

metrics = model.evaluate(test_x, test_y)
test_mse = metrics[1]
test_accuracy = metrics[2]
test_cross_entropy = metrics[3]

wandb.log({"test_mse": test_mse, "test_accuracy": test_accuracy,
           "test_cross_entropy": test_cross_entropy})

y_pred = model.predict(test_x)

class_names = [f"{i}" for i in range(10)]
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=test_y,
    preds=y_pred,
    class_names=class_names)})
