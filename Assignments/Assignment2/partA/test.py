import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
import data_gen
import model_maker


# Default dictionary for hyperparams
hyperparameter_defaults = dict(
    number_of_filters_first_layer = 32,
    filter_organisation = "linear_inc",
    filter_size_0 = 3,
    filter_size_1 = 3,
    filter_size_2 = 3,
    filter_size_3 = 3,
    filter_size_4 = 3,
    pool_size = 2,
    data_augmentation = "No",
    dropout = 0,
    batch_normalisation = "No",
    convolution_activation = "relu",
    dense_activation = "tanh",
    neurons_in_dense_layer = 128,
    epochs = 20,
    optimizer = "nadam",
    batch_size = 32,
)

class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']

# Wandb initialization
wandb.init(config=hyperparameter_defaults)
config = wandb.config

train_ds, val_ds, test_ds = data_gen.generate_dataset(config)
model = tf.keras.models.load_model("org_linear_inc_nfLayer1_32_dataAug_False_drpout_0_bs_32_opt_nadam")

def get_accuracy() :
  loss, accuracy = model.evaluate(
        test_ds)
  print('Test_Accuracy: ', round((accuracy)*100, 2))

# choose 30 images and plot them with their true class and predicted class
# For more points try adding the class our model max confused with
def plot_sample_images() :
  image_list, true_class, pred_class, pred_proba = [], [], [], []
  for images, labels in test_ds :
    pred_proba = model.predict(images)
    pred_class = np.argmax(pred_proba, axis = 1)
    true_class = np.argmax(labels, axis = 1)
    image_list = images
    break
  fig, axes = plt.subplots(10, 3, figsize = (50, 50))
  axes = axes.flatten()
  cnt = 0
  for (image, true_label, pred_label, ax) in zip(image_list, true_class, pred_class, axes) :
    if cnt >= 30 :
      break
    ax.imshow(image)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([]) 
    ax.set_xlabel("TRUE CLASS : " + class_names[true_label] + "\n" + "PREDICTED CLASS : " + class_names[pred_label] + "\n" + \
        "CONFIDENCE FOR ACTUAL CLASS : " + str(pred_proba[cnt][true_label] * 100) + "%" + "\n" + "CONFIDENCE FOR PREDICTED CLASS : " + str(np.max(pred_proba[cnt] * 100)) + "%")
    cnt += 1
  plt.tight_layout()
  plt.savefig("Classification_on_sample_test_images.png")

def visualize_layer_1_filter() :
  filters = model.layers[0].get_weights()[0]
  # print(filters)
  # normalize filter values to 0-1 so we can visualize them
  f_min, f_max = filters.min(), filters.max()
  filters = (filters - f_min) / (f_max - f_min)
  # plot first few filters

  n_filters = config.number_of_filters_first_layer
  print(n_filters)
  fig, ax = plt.subplots(n_filters // 4, 4, figsize = (10, 25))
  for i in range(0, n_filters, 4):
    # get the filter
    for j in range(i, i + 4) :
      ax[i // 4, j % 4].imshow(filters[:, :, :, j])
      ax[i // 4, j % 4].xaxis.set_ticks([])
      ax[i // 4, j % 4].yaxis.set_ticks([])
  # show the figure
  plt.savefig("Layer_1_Filter_Visualization.png")

def visualize_layer_1_filter_on_image() :
  layer_name = 'conv2d'
  print(model.summary())
  # restrict model till only the first convolution layer
  new_model = tf.keras.models.Model(
    inputs = [model.inputs],    
    outputs = [model.get_layer(layer_name).output]
  )

  for images, labels in test_ds :
    feature_map_list = new_model.predict(images)
    print(feature_map_list[0].shape)
    side1 = 8
    side2 = 4
    index = 1
    for _ in range(side1):
      for _ in range(side2): 
        ax = plt.subplot(side1, side2, index)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature_map_list[0, :, :, index-1], cmap='viridis')
        index += 1
    break 
  plt.savefig("filter_1_visualize_on_image.png")

get_accuracy()
plot_sample_images()
visualize_layer_1_filter()
visualize_layer_1_filter_on_image()
