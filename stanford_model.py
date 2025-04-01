"""
Script to run and train a model on the sStanford Dogs dataset using the architecture from model.py.
Stores the model during epochs and when completed training.
"""

import datamodule as dm
import model as custom_model
from tensorflow import data as tf_data
import keras
from keras import layers

IMAGE_SIZE=(180, 180)
BATCH_SIZE=128
VAL_SPLIT=0.2
EPOCHS=50
LR=1e-4

#load the stanford dataset
#data path: data/stanford_dogs/Images
dm.filter_images(path="data/stanford_dogs/Images") #uncomment if running first time

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 128
NUM_CLASSES_STANFORD_DOGS = 120


train_ds, val_ds = dm.train_val_split(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, path="data/stanford_dogs/Images")

dm.vis_data(train_ds)

#set how you want to augment the data
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (dm.data_augmentation(img, data_augmentation_layers), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

dm.vis_data_augmentations(train_ds, data_augmentation_layers)

model = custom_model.make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES_STANFORD_DOGS)
keras.utils.plot_model(model, show_shapes=True)

#train a model based on stanford dataset
history = custom_model.train_model(
    model=model,
    train_ds=train_ds,
    val_ds=val_ds,
    epochs=EPOCHS,
    ckpt_path="trained_models/stanford_dogs_epoch_{epoch}.keras",
    learning_rate=LR,
)

#save model
model.save("trained_models/stanford_dogs_model.h5")
print("Saved trained Stanford Dogs model to stanford_dogs_model.h5")
