import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os


# check the corrupted images
def remove_corrupted_images() :
    path_to_dataset = "inaturalist_12K"
    ignore_files = {".DS_Store"}

    num_corrupted = 0
    for folder_name in ("train", "val") :
        folder_path = os.path.join(path_to_dataset, folder_name)
        for fname in os.listdir(folder_path) :
            class_path = os.path.join(folder_path, fname) 
            if fname in ignore_files :
                continue
            for class_name in os.listdir(class_path) :
                fpath = os.path.join(class_path, class_name)
                try :
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                finally :
                    fobj.close()
                if not is_jfif :
                    num_corrupted += 1
                    #os.remove(fpath)

    print("Number Of Corrupted Images : ", num_corrupted)

# generating the Dataset, returns train, validation and test data
# TODO : add more support for data augmentation 

def generate_dataset(config) :
     
    imagegen_without_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, 
        validation_split = 0.1
    )
 
    if config.data_augmentation == "No" :
        imagegen = imagegen_without_aug    
    else :
        imagegen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255, 
            validation_split = 0.1, 
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            zoom_range = 0.3,
            zca_epsilon = 1e-6,
        )
  
    train_ds = imagegen.flow_from_directory(
        "inaturalist_12K/train/",
        subset = "training", 
        seed = 1337, 
        class_mode = "categorical",
        batch_size = config.batch_size
    )
  
    val_ds = imagegen.flow_from_directory(
        "inaturalist_12K/train/", 
        subset = "validation",
        seed = 1337,
        class_mode = "categorical", 
        batch_size = config.batch_size
    )
  
    test_ds = imagegen_without_aug.flow_from_directory(
        "inaturalist_12K/val/",  
        seed = 1337,  
        batch_size = config.batch_size
    ) 
  
    return train_ds, val_ds, test_ds

# A helper function to view images
def visualize_images() :
    for images, labels in train_ds :
        fig, axes = plt.subplots(len(labels), 1, figsize = (20, 20))
        axes.flatten()
        for (im, ax) in zip(images, axes) :
            ax.imshow(im)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        break 
