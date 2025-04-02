import os
import sys
import keras
import matplotlib
if sys.platform.startswith("linux"):
    matplotlib.use("TkAgg")  # for linux
import matplotlib.pyplot as plt
import numpy as np

def filter_images(path='data/kagglecatsanddogs_5340/PetImages'):
    """
    Filters corrupt images from the cat vs dogs dataset from kaggle.
    """
    print('Starting image filtering.')
    num_skipped = 0
    for folder_name in os.listdir(path):
        #print('FOLDER NAMES:',folder_name) debugging
        folder_path = os.path.join(path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print(f"Deleted {num_skipped} images.")

def train_val_split(image_size, batch_size, path='data/kagglecatsanddogs_5340/PetImages'):
    """Creates a training and validation split of data"""
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
            path,
            validation_split=0.2,
            subset="both",
            seed=42,
            image_size=image_size,
            batch_size=batch_size,
        )
    return train_ds, val_ds

def vis_data(data):
    """
    Visualizes first 9 images in data
    """
    plt.figure(figsize=(10, 10))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

def data_augmentation(images, data_augmentation_layers):
    """
    Data augmentation for images, augment images according to the data_augmentation_layers.
    """
    for layer in data_augmentation_layers:
        images = layer(images, )
    return images

def vis_data_augmentations(data, data_augmentation_layers):
    """
    Augments a number of images in the dataset.
    """
    for images, _ in data.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images, data_augmentation_layers)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")
    plt.show()