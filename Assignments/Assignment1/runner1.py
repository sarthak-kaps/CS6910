import models
import optimizers
import numpy as np
from sklearn.model_selection import train_test_split


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

opt = optimizers.Adam(learning_rate=0.1)
model.compile(opt)

model.fit(train_x, train_y, epochs=100, verbose=10, cold_start=True)
train_loss, train_acc = model.evaluate(train_x, train_y)
val_loss, val_acc = model.evaluate(val_x, val_y)
print("train : loss - %f acc - %f" % (train_loss, train_acc))
print("val   : loss - %f acc - %f" % (val_loss, val_acc))
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
