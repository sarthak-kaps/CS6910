import models
import optimizers
import numpy as np

X = np.random.random((1000, 10))
t = np.sum(X ** 2, axis=1)
y = (t > np.mean(t)) * 1

model = models.Sequential()
model.add(models.Dense(20, input_dim=10))
model.add(models.Dense(2))
model.add(models.Softmax())

opt = optimizers.Nadam()
model.compile(opt)

model.fit(X, y, epochs = 1000, verbose = 100)
model.evaluate(X, y)
