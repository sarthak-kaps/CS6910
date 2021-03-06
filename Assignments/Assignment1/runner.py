import models
import optimizers
import numpy as np

np.random.seed(0)
X = np.random.random((1000, 10)) - 0.5
t = np.sum(X, axis=-1)
y = (t > 0) * 1

print(((np.count_nonzero(y)/len(y)))*100)

model = models.Sequential()
model.add(models.Dense(20, activation="sigmoid", input_dim=10))
model.add(models.Dense(20, activation="sigmoid"))
model.add(models.Dense(2, activation="sigmoid"))
model.add(models.Softmax())

opt = optimizers.Adam(learning_rate=0.01)
model.compile(opt)

model.fit(X, y, epochs=200, verbose=10)
model.evaluate(X, y)


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
