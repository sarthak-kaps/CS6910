import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback
import data_gen
import model_maker


# Default dictionary for hyperparams
hyperparameter_defaults = dict(
    number_of_filters_first_layer = 32,
    filter_organisation = "double",
    filter_size_0 = 3,
    filter_size_1 = 3,
    filter_size_2 = 3,
    filter_size_3 = 3,
    filter_size_4 = 3,
    pool_size = 2,
    data_augmentation = "Yes",
    dropout = 0.2,
    batch_normalisation = "Yes",
    convolution_activation = "relu",
    dense_activation = "tanh",
    neurons_in_dense_layer = 128,
    epochs = 10, # 30
    optimizer = "adam",
    batch_size = 256,
)

class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']

# Wandb initialization
wandb.init(config=hyperparameter_defaults)
config = wandb.config

model = tf.keras.models.load_model("best_model")

# get prediction on the test images
predict = model.predict(test_ds)
pred_class_indices = np.argmax(predict, axis = 1)
true_class_indices = test_ds.labels


def get_accuracy() :
  acc = 0
  # find accuracy
  assert(len(pred_class_indices) = len(true_class_indices))
  
  for i in range(0, len(pred_class_indices)) :
    acc += int(pred_class_indices[i] == true_class_indices[i])

  wandb.log({"test_accuracy": acc})

# choose 30 images and plot them with their true class and predicted class
# For more points try adding the class our model max confused with
def plot_sample_images :
  img_cnt = 0
  image_and_labels = []
  for images, labels in test_ds :
    if img_cnt >= 30 : 
      break
    choose = 0
    for (im, lb) in zip(images, labels) :
      if choose >= 2 :
        break
      image_and_labels.append((im, lb))
      img_cnt += 1
      choose += 1
   
  fig, axes = plt.subplots(3, 10, figsize = (50, 50))
  axes = axes.flatten()
 
  for ((image, label), ax) in zip(image_and_labels, axes) :
    ax.imshow(image)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([]) 
    pred = test.predict(image)
    ax.set_xlabel("TRUE CLASS " + class_names[np.argmax(label)] + " PRED CLASS " + class_names[np.argmax(pred)])
  plt.tight_layout()
  wandb.log({"Classification on sample test images", plt})

def visualize_layer_1_filter() :
  filters, biases = model.layers[1].get_weights() 
  # normalize filter values to 0-1 so we can visualize them
  f_min, f_max = filters.min(), filters.max()
  filters = (filters - f_min) / (f_max - f_min)
  # plot first few filters
  n_filters, ix = config.number_of_filters_first_layer, 1
  for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately
    for j in range(3):
      ax = plt.subplot(n_filters, 3, ix)
      ax.set_xticks([])
      ax.set_yticks([])
      # plot filter channel in grayscale
      plt.imshow(f[:, :, j])
      ix += 1
  # show the figure
  wandb.log({"Layer 1 Filter Visualization", plt})

get_accuracy()
plot_sample_images()
visualize_layer_1_filter()
