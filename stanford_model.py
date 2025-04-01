"""
Script to run and train a model on the sStanford Dogs dataset using the architecture from model.py.
Stores the model during epochs and when completed training.
"""

import datamodule as dm
import model as custom_model

IMAGE_SIZE=(180, 180)
BATCH_SIZE=128
VAL_SPLIT=0.2
EPOCHS=50
LR=1e-4

#load the stanford dataset
train_ds, val_ds, ds_info = dm.load_stanford_dogs_dataset(
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

#create model
num_classes = ds_info.features["label"].num_classes
model = custom_model.make_model(input_shape=IMAGE_SIZE + (3,), num_classes=num_classes)

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
