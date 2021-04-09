import tensorflow as tf
from tensorflow import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback
import data_gen
import model_maker

# For consistency
np.random.seed(0)

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
wandb.run.name = f"org_{config.filter_organisation}_nfLayer1_{config.number_of_filters_first_layer}_dataAug_{config.data_augmentation}_drpout_{config.dropout}_bs_{config.batch_size}_opt_{config.optimizer}"

wandb.run.save()


train_ds, val_ds, test_ds = data_gen.generate_dataset(config)
model = model_maker.make_model(train_ds.image_shape, 10, config)

#print(model.summary())

model.compile(
  optimizer = config.optimizer,
  loss = "categorical_crossentropy",
  metrics = ["accuracy"]
)

model.fit(
  train_ds, 
  epochs = config.epochs, 
  validation_data= val_ds,
  callbacks=[WandbCallback(data_type="image", labels=class_names)]
)

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

get_accuracy()
plot_sample_images()
