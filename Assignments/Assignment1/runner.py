import models
import optimizers
import numpy as np

np.random.seed(0)
X = np.random.random((1000, 10)) - 0.5
t = np.sum(X, axis=-1)
y = (t > 0) * 1

print(((np.count_nonzero(y)/len(y)))*100)

model = models.Sequential()
model.add(models.Dense(20, input_dim=10))
model.add(models.Dense(2))
model.add(models.Softmax())

opt = optimizers.SGD(1e-6, 1)
model.compile(opt, metrics=["cross_entropy"])

model.fit(X, y, epochs=200, verbose=10)
model.evaluate(X, y)
