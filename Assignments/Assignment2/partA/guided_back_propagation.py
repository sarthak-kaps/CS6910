import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb
import data_gen


# Default dictionary for hyperparams
hyperparameter_defaults = dict(
    number_of_filters_first_layer = 32,
    filter_organisation = "same",
    filter_size_0 = 3,
    filter_size_1 = 3,
    filter_size_2 = 3,
    filter_size_3 = 3,
    filter_size_4 = 3,
    pool_size = 2,
    data_augmentation = "No",
    dropout = 0.2,
    batch_normalisation = "No",
    convolution_activation = "relu",
    dense_activation = "relu",
    neurons_in_dense_layer = 128,
    epochs = 10,
    optimizer = "adam",
    batch_size = 32,
)

class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']

# Wandb initialization
wandb.init(config=hyperparameter_defaults)
config = wandb.config
wandb.run.name = f"org_{config.filter_organisation}_nfLayer1_{config.number_of_filters_first_layer}_dataAug_{config.data_augmentation}_drpout_{config.dropout}_bs_{config.batch_size}_opt_{config.optimizer}"

wandb.run.save()


# get the training data
train_ds, val_ds, test_ds = data_gen.generate_dataset(config)

# load the best model
model = tf.keras.models.load_model("org_linear_inc_nfLayer1_32_dataAug_False_drpout_0_bs_32_opt_nadam")

@tf.custom_gradient
def guided_back_ReLU(x):
  # if dy i.e the value of gradient coming into this layer is < 0 then ignore relu and pass 0
  # otherwise apply Relu and multiply it with the gradient dy coming here
  def grad(dy) :
    return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
  return tf.nn.relu(x), grad

#print(model.summary())

final_layer_name = "conv2d_4"

# restrict model till only the last convolution layer
model = tf.keras.models.Model(
    inputs = [model.inputs],    
    outputs = [model.get_layer(final_layer_name).output]
)

# all layers except the input layer
model_layers = model.layers[1:]

fired_neurons = set() # stores any 10 neurons in the final convolution layer that got fired 
num_required_neurons = 10

os.system("mkdir Guided_Back_Prop_Images")

for layer in model_layers :
  if hasattr(layer, 'activation') :
    if layer.activation == tf.keras.activations.relu :
      layer.activation = guided_back_ReLU

for images, labels in test_ds : 
  if len(fired_neurons) >= num_required_neurons :
    break
  with tf.GradientTape() as tape :
    inputs = tf.cast(images, tf.float32)
    tape.watch(inputs)
    outputs = model(inputs)

  grads = tape.gradient(outputs, inputs)

  guided_back_prop_list = grads
  
  for (im, guided_back_prop) in zip(images, guided_back_prop_list) : 
    if len(fired_neurons) >= num_required_neurons :
      break
    gb_viz = np.dstack((
        guided_back_prop[:, :, 0], 
        guided_back_prop[:, :, 1],
        guided_back_prop[:, :, 2],
      ))
    gb_viz -= np.min(gb_viz)
    gb_viz /= (gb_viz.max() + 0.0001)
    
    fig, axes = plt.subplots(2, 1, figsize = (20, 20))
  
    imgplot = axes[0].imshow(gb_viz)
    axes[1].imshow(im)
    plt.axis("off")

    if(np.max(gb_viz) is not np.nan and np.max(gb_viz) > 0.9) :
      neuron = np.argmax(gb_viz)
      fired_neurons.add(neuron)
      plt.xlabel("Image at which " + str(neuron) + " in the final convolution layer gets fired")
      plt.savefig("Guided_Back_Prop_Images/Fired_Neuron_" + str(len(fired_neurons)) + ".png")

    #plt.show() 
    #print(np.argmax(gb_viz), np.max(gb_viz), gb_viz.shape, gb_viz.shape[0] * gb_viz.shape[1] * gb_viz.shape[2])
