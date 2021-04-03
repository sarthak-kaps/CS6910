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

