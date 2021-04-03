import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

num_convolution_layers = 5 # fixed
min_number_of_filters = 8 
max_number_of_filters = 512
num_filter_gap = 32

def make_model(input_shape, num_classes, config) :
  model = keras.Sequential()
  model.add(layers.Input(input_shape))
   
  num_filters = [config.number_of_filters_first_layer]

  if config.filter_organisation == "same" :
    for i in range(1, num_convolution_layers) :
      num_filters.append(num_filters[i - 1])
  
  elif config.filter_organisation == "half" :
    for i in range(1, num_convolution_layers) :
      num_filters.append(max(num_filters[i - 1] / 2, min_number_of_filters))
  
  elif config.filter_organisation == "double" :
    for i in range(1, num_convolution_layers) :
      num_filters.append(min(num_filters[i - 1] * 2, max_number_of_filters))

  elif config.filter_organisation == "linear_inc" :
    for i in range(1, num_convolution_layers) :
      num_filters.append(num_filters[i - 1] + num_filter_gap)

  else :
    raise ValueError("Invalid Filter Organisation {}", config.filter_organisation)

  filter_sizes = [config.filter_size_0, config.filter_size_1, \
      config.filter_size_2, config.filter_size_3, config.filter_size_4]
  
  for i in range(0, num_convolution_layers) :
    model.add(layers.Conv2D(num_filters[i], (filter_sizes[i], filter_sizes[i])))
    
    if config.batch_normalisation == "Yes" :
      model.add(BatchNormalization())
    
    model.add(layers.Activation(config.convolution_activation))
    model.add(layers.MaxPooling2D(pool_size = (config.pool_size, config.pool_size)))
    
  if config.dropout > 0 :
    model.add(layers.Dropout(config.dropout))
  
  model.add(layers.Flatten())
  
  model.add(layers.Dense(config.neurons_in_dense_layer, activation = config.dense_activation))
  
  if config.dropout > 0 :
    model.add(layers.Dropout(config.dropout))

  model.add(layers.Dense(10, activation = "softmax"))
  
  return model
