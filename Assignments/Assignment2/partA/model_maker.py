import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


def make_model(input_shape, num_classes) :
  model = keras.Sequential()
  model.add(layers.Input(input_shape))
  model.add(layers.Conv2D(16, (3, 3)))
  model.add(layers.Activation("relu"))
  model.add(layers.MaxPooling2D(pool_size = (2, 2)))
  
  model.add(layers.Conv2D(16, (3, 3)))
  model.add(layers.Activation("relu"))
  model.add(layers.MaxPooling2D(pool_size = (2, 2)))
  
  model.add(layers.Conv2D(16, (3, 3)))
  model.add(layers.Activation("relu"))
  model.add(layers.MaxPooling2D(pool_size = (2, 2)))
  
  model.add(layers.Conv2D(16, (3, 3)))
  model.add(layers.Activation("relu"))
  model.add(layers.MaxPooling2D(pool_size = (2, 2)))
  
  model.add(layers.Conv2D(16, (3, 3)))
  model.add(layers.Activation("relu"))
  model.add(layers.MaxPooling2D(pool_size = (2, 2)))
  
  model.add(layers.Flatten())
  
  model.add(layers.Dense(64, activation = "relu"))
  model.add(layers.Dense(10, activation = "softmax"))
  
  return model
