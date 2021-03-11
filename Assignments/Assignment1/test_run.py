#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix

import models
import optimizers
import wandb

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep

hyperparameter_defaults = dict(
    hidden_layer_size=128,
    num_hidden_layers=3,
    learning_rate=0.001,
    epochs=100,
    l2=25e-5,
    optimizer="adam",
    batch_size=2048,
    weight_init="xavier",
    activation="ReLU",
)


metrics_list = ["mse", "accuracy", "cross_entropy"]


def callback(eps, train_metrics, val_metrics):
    # print(metrics)
    wandb.log({"epoch": eps, "train_loss": train_metrics[0], "train_mse": train_metrics[1],
               "train_accuracy": train_metrics[2], "train_cross_entropy": train_metrics[3]})


    # Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults, project='assignment1')
config = wandb.config
print(config)

# Your model here ...
np.random.seed(0)
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_x = np.reshape(train_x, (-1, 784))/255.0
test_x = np.reshape(test_x, (-1, 784))/255.0

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

model.fit(train_x, train_y, epochs=config.epochs, verbose=10,
          batch_size=config.batch_size, callback_fn=callback)

y_pred = model.predict(test_x)


wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=test_y,
    preds=y_pred,
    class_names=class_names)})

# Log metrics inside your training loop
# ""

test_y_name = [class_names[e] for e in test_y]
y_pred_name = [class_names[e] for e in y_pred]


def plot_cm(y_true, y_pred, figsize=(15, 15)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n' % (p)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="Greens", annot=annot, fmt='', ax=ax)
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


img_arr = plot_cm(test_y_name, y_pred_name)
wandb.log({"Confusion Matrix custom": [
          wandb.Image(img_arr, caption="Confusion matrix 2")]})
