import models
import optimizers
import numpy as np
from keras.datasets import fashion_mnist

np.random.seed(0)
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_x = np.reshape(train_x, (-1, 784))/255.0
test_x = np.reshape(test_x, (-1, 784))/255.0

model = models.Sequential()
model.add(models.Dense(128, activation="ReLU", input_dim=784))
# model.add(models.Dense(32, activation="sigmoid"))
model.add(models.Dense(10, activation="ReLU"))
model.add(models.Softmax())

opt = optimizers.SGD()
model.compile(opt)

model.fit(train_x, train_y, epochs=100, verbose=1, batch_size=32)
train_loss, train_acc = model.evaluate(train_x, train_y)
val_loss, val_acc = model.evaluate(test_x, test_y)
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
