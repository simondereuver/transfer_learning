import datamodule as dm
import model as custom_model
from tensorflow import data as tf_data
from keras import layers
import keras
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #path_to_images = "kagglecatsanddogs_5340" #uncomment and adjust path if needed

    #filter out corrupted images
    #dm.filter_images() #uncomment if running first time
    
    IMAGE_SIZE = (180, 180)
    BATCH_SIZE = 128
    NUM_CLASSES_STANFORD_DOGS = 120
    NUM_CLASSES_CATS_DOGS = 2

    # TODO: fix train_val_split to also create a test split so we can evaluate models
    train_ds, val_ds = dm.train_val_split(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

    #dm.vis_data(train_ds)

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

    #dm.vis_data_augmentations(train_ds, data_augmentation_layers)

    ##### EXPERIMENTS #####

    EPOCHS=50
    LR=1e-4
    
    #experiment 2: load the saved model and replace only the output layer of the model (to align it to the new problem). 

    #load model
    stanford_model = keras.models.load_model("trained_models/stanford_dogs_epoch_50.keras") #base model
    #stanford_model.summary() #useful to see what current model looks like

    #get output before last layer
    x = stanford_model.get_layer('dropout').output
    
    #connect to output layer
    new_output = layers.Dense(NUM_CLASSES_CATS_DOGS, activation=None)(x)

    #create the new model
    ex2_model = keras.Model(inputs=stanford_model.input, outputs=new_output)

    #compare summaries to make sure last layer was replaced as intended
    #stanford_model.summary()
    #ex2_model.summary()

    #train and evaluate the model (for 50 epochs) on the cats and dogs dataset.

    ex2_model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint("trained_models/ex2_model_{epoch}.keras"),
    ]

    ex2_model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    """
    model = custom_model.make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES_STANFORD_DOGS)
    keras.utils.plot_model(model, show_shapes=True)

    history = custom_model.train_model(model=model, train_ds=train_ds, val_ds=val_ds, epochs=5, ckpt_path="save_at_{epoch}.keras", learning_rate=1e-4,)

    img = keras.utils.load_img("kagglecatsanddogs_5340/PetImages/Cat/6779.jpg", target_size=IMAGE_SIZE)
    plt.imshow(img)
    plt.show()

    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = float(keras.ops.sigmoid(predictions[0][0]))
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

    #load model like this:
    #stanford_model = keras.models.load_model("stanford_dogs_model.h5")
    """