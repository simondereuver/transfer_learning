import datamodule as dm
import model as custom_model
from tensorflow import data as tf_data
from keras import layers
import keras
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #path_to_images = "kagglecatsanddogs_5340" #uncomment and adjust path if needed

    #filter out corrupted images
    dm.filter_images(path="data/stanford_dogs/Images") #uncomment if running first time
    
    IMAGE_SIZE = (180, 180)
    BATCH_SIZE = 128
    NUM_CLASSES_STANFORD_DOGS = 120
    NUM_CLASSES_CATS_DOGS = 2

    train_ds, val_ds = dm.train_val_split(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, path="stanford_dogs/Images")

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
