import os
import sys
import keras
import matplotlib
if sys.platform.startswith("linux"):
    matplotlib.use("TkAgg")  # for linux
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


import tensorflow as tf
import tensorflow_datasets as tfds

def load_stanford_dogs_dataset1(image_size=(180, 180), batch_size=128, val_split=0.2):
    """
    Creates a validation set from the training data.
    """
    (ds_train, ds_test), ds_info = tfds.load(
        'stanford_dogs',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='data/tfds'
    )

    def preprocess(img, label):
        img = tf.image.resize(img, image_size)
        return img, label

    # Combine and shuffle before splitting
    full_ds = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    full_ds = full_ds.shuffle(ds_info.splits['train'].num_examples, reshuffle_each_iteration=False)

    total_train = ds_info.splits['train'].num_examples
    val_size = int(val_split * total_train)

    ds_val = full_ds.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_train = full_ds.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_info

def load_stanford_dogs_dataset(image_size=(180, 180), batch_size=128):
    """
    Uses the test set as validation data.
    """
    (ds_train, ds_test), ds_info = tfds.load('stanford_dogs', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True, data_dir='data/tfds')
    
    def preprocess(img, label):
        img = tf.image.resize(img, image_size)
        return img, label

    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info

def filter_images_cats_dogs(path='kagglecatsanddogs_5340/PetImages'):
    """
    Filters corrupt images from the cat vs dogs dataset from kaggle.

    """
    print('Starting image filtering.')
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
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

def train_val_split(image_size, batch_size, path='kagglecatsanddogs_5340/PetImages'):
    """Creates a training and validation split of data"""
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
            path,
            validation_split=0.2,
            subset="both",
            seed=1337,
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