import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import Input, optimizers
from tensorflow.keras.layers import (Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

wandb.login()

INPUT_SHAPE = (600, 800, 3)
MODEL_INPUT = (256, 256, 3)
BATCH_SIZE = 20
CLASSES = 10


# Funtions to load the required pretrained models

def load_inception_resnet_v2(input_shape=INPUT_SHAPE):
    return tf.keras.applications.InceptionResNetV2(
        include_top=False,
        input_shape=input_shape,
    )


def load_inception_v3(input_shape=INPUT_SHAPE):
    return tf.keras.applications.InceptionV3(
        include_top=False,
        input_shape=input_shape,
    )


def load_resnet_50(input_shape=INPUT_SHAPE):
    return tf.keras.applications.ResNet50V2(
        include_top=False,
        input_shape=input_shape,
    )


def load_xception(input_shape=INPUT_SHAPE):
    return tf.keras.applications.Xception(
        include_top=False,
        input_shape=input_shape,
    )


# Dictionary containing all the required models dispatch functions
models = {
    "InceptionV3": load_inception_v3, "InceptionResNetV2": load_inception_resnet_v2, "ResNet50": load_resnet_50, "Xception": load_xception
}


def generate_dataset(batch_size, dir_name="inaturalist_12K"):
    """
    Extracts the images and instantiates a ImageDataGenerator object for training, validation and testing.
    For the purpose of the current experiments, we are not augmenting the dataset.
    If required it can be added later.
    """
    imagegen_without_aug = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    imagegen = imagegen_without_aug

    train_ds = imagegen.flow_from_directory(
        dir_name+"/train/",
        subset="training",
        seed=1337,
        class_mode="categorical",
        batch_size=batch_size,
        target_size=MODEL_INPUT[:-1]
    )

    val_ds = imagegen.flow_from_directory(
        dir_name+"/train/",
        subset="validation",
        seed=1337,
        class_mode="categorical",
        batch_size=batch_size,
        target_size=MODEL_INPUT[:-1]

    )

    test_ds = test_datagen.flow_from_directory(
        dir_name+"/val/",
        seed=1337,
        batch_size=batch_size,
        target_size=MODEL_INPUT[:-1]

    )

    return train_ds, val_ds, test_ds


def train():
    """
    This function is called by wandb.sweep for training and logging.
    """

    # Wandb default config
    config_defaults = {
        "epochs": 5,
        "batch_size": 32,
        "model_name": "InceptionV3",
    }

    # Initialize the project
    wandb.init(project='assignment2-partB',
               group='experiment2',
               config=config_defaults)

    # Initialize the model
    model_func = models[wandb.config.model_name]
    base_model = model_func(MODEL_INPUT)
    base_model.trainable = False

    inputs = Input(MODEL_INPUT)

    # Add norm layer to scale inputs to [-1, +1]
    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([0.5] * 3)
    var = mean ** 2
    x = norm_layer(inputs)
    norm_layer.set_weights([mean, var])

    # Define the final model
    base_model_layer = base_model(x, training=False)
    # Global average pooling helps to reduce the params of FC layers
    avg_pooling = GlobalAveragePooling2D()(base_model_layer)
    dropout1 = Dropout(rate=0.2)(avg_pooling)  # For l2 regularization
    class1 = Dense(512, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(dropout1)  # Dense 1 layers
    dropout2 = Dropout(rate=0.2)(class1)
    output = Dense(10, activation='softmax')(dropout2)  # Final softmax

    # Update the model
    model = Model(inputs=inputs, outputs=output)

    model.summary()
    # Compile
    model.compile(optimizer=optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Get the dataset
    train_ds, val_ds, test_ds = generate_dataset(wandb.config.batch_size)

    # Fit the model
    model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds,
        callbacks=[WandbCallback(save_model=False)])

    _, accuracy = model.evaluate(
        test_ds, callbacks=[WandbCallback(save_model=False)])

    # wandb.log to track custom metrics
    print('Test_Accuracy: ', round((accuracy)*100, 2))
    wandb.log({'Test_Accuracy_before_tuning': round((accuracy)*100, 2)})

    # Fine tuning the base
    base_model.trainable = True
    model.compile(optimizer=optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        train_ds,
        epochs=3,
        validation_data=val_ds,
        callbacks=[WandbCallback(save_model=False)])

    # Evaluate on the test dataset
    loss, accuracy = model.evaluate(
        test_ds, callbacks=[WandbCallback(save_model=False)])
    print('Test_Accuracy: ', round((accuracy)*100, 2))

    # wandb.log to track custom metrics
    wandb.log({'Test_Accuracy': round((accuracy)*100, 2)})


# sweep config to run
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5
    },
    'parameters': {
        'batch_size': {
            'values': [32]
        },
        'model_name': {
            'values': ["InceptionV3", "InceptionResNetV2", "ResNet50", "Xception"]
        }
    }
}

# Get the sweep and run
sweep_id = wandb.sweep(sweep_config, project="Assignment-partB")

wandb.agent(sweep_id, function=train)
