import numpy as np
import wandb
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

from src import models, optimizers, extras

# For consistentency
np.random.seed(0)

# Default dictionary for hyperparams
hyperparameter_defaults = dict(
    hidden_layer_size=128,
    num_hidden_layers=3,
    learning_rate=0.001,
    epochs=10,
    l2=25e-6,
    optimizer="adam",
    batch_size=128,
    weight_init="xavier",
    activation="ReLU",
    loss="cross_entropy"
)

# List of metrices to be checked while fitting
metrics_list = ["mse", "accuracy", "cross_entropy"]

# Callback function for logging various metrices to wandb


def callback(eps, train_metrics, val_metrics):
    # print(metrics)
    wandb.log({"train_loss": train_metrics[0], "train_mse": train_metrics[1],
               "train_accuracy": train_metrics[2], "train_cross_entropy": train_metrics[3],
               "val_loss": val_metrics[0], "val_mse": val_metrics[1], "val_accuracy": val_metrics[2],
               "val_cross_entropy": val_metrics[3]
               })


# Wandb initialization
wandb.init(config=hyperparameter_defaults)
config = wandb.config
wandb.run.name = f"hl_{config.hidden_layer_size}_nhl_{config.num_hidden_layers}_bs_{config.batch_size}_opt_{config.optimizer}"
wandb.run.save()

# Loading dataset and names
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


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


# Check test dataset prediction
y_pred = model.predict(test_x)

# Adding test accuracy
wandb.log({"test_accuracy": extras.accuracy(test_y, y_pred, None)})

# Creating the Confusion matrix
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=test_y,
    preds=y_pred,
    class_names=class_names)})
