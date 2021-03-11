import models
import optimizers
import numpy as np
import wandb
from sklearn.model_selection import train_test_split

wandb.init(project="trial")

np.random.seed(0)
X = np.random.random((10000, 10)) - 0.5
t = np.sum(X, axis=-1)
y = (t > 0) * 1

train_x, val_x, train_y, val_y = train_test_split(
    X, y, test_size=0.1, random_state=0)


model = models.Sequential()
model.add(models.Dense(20, activation="sigmoid", input_dim=10))
model.add(models.Dense(20, activation="sigmoid"))
model.add(models.Dense(2, activation="sigmoid"))
model.add(models.Softmax())

<<<<<<< HEAD
opt = optimizers.Adam(learning_rate=0.01)
=======
wandb.config.learnning_rate = 0.1

opt = optimizers.Adam(learning_rate=0.1)
>>>>>>> e6421270455b757b9cc01d4590739fe156b15183
model.compile(opt)

for i in range(20):
    model.fit(train_x, train_y, epochs=10, verbose=10, cold_start=True)
    train_loss, train_acc = model.evaluate(train_x, train_y)
    val_loss, val_acc = model.evaluate(val_x, val_y)
    wandb.log({"epoch": i*10, "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc})

"""
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

model = models.Sequential()
model.add(models.Dense(4, activation = "sigmoid", input_dim = 2, weight_initializer = "random"))
#model.add(models.Dense(20, activation = "sigmoid"))
model.add(models.Dense(2, activation = "sigmoid", weight_initializer = "random"))
model.add(models.Softmax())

opt = optimizers.Adam(learning_rate = 0.01)

model.compile(opt)
model.fit(X, Y, epochs = 1000, verbose = 10)
model.evaluate(X, Y)
"""
