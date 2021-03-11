import wandb
import models
import optimizers
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep

hyperparameter_defaults = dict(
    hidden_layer_size=32,
    num_hidden_layers=3,
    learning_rate=0.001,
    epochs=5,
    l2=0,
    optimizer="adam",
    batch_size=64,
    weight_init="random",
    activation="ReLU",
)


metrics_list = ["mse", "accuracy"]


def callback(eps, train_metrics, val_metrics):
    # print(metrics)
    wandb.log({"train_loss": train_metrics[0], "train_mse": train_metrics[1],
               "train_accuracy": train_metrics[2],
               "val_loss": val_metrics[0], "val_mse": val_metrics[1], "val_accuracy": val_metrics[2]})

    # Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
config = wandb.config

# Your model here ...
np.random.seed(0)
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_x = np.reshape(train_x, (-1, 784))/255.0
test_x = np.reshape(test_x, (-1, 784))/255.0

train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.1, random_state=0, shuffle=True)

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

opt = optimizers.optimizers[config.optimizer](config.learning_rate)
model.compile(opt, metrics=metrics_list)

model.fit(train_x, train_y, val_x, val_y, epochs=config.epochs, verbose=10,
          batch_size=config.batch_size, callback_fn=callback)

# Log metrics inside your training loop