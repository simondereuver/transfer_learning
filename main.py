import datamodule as dm
from tensorflow import data as tf_data
from tensorflow.keras import layers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

def tl_experiment_one(base_model, epochs, learning_rate, train_ds, val_ds, test_ds):
    ex2_inputs = base_model.input

    #get output before last layer
    x = base_model.get_layer('dropout').output
    
    #connect to output layer
    new_output = layers.Dense(1, activation=None)(x)

    #create the new model
    ex2_model = keras.Model(inputs=ex2_inputs, outputs=new_output)

    #compare summaries to make sure last layer was replaced as intended
    #base_model.summary()
    #ex2_model.summary()
    
    #train and evaluate the model (for 50 epochs) on the cats and dogs dataset.
    ex2_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint("trained_models/ex2_model_{epoch}.keras"),]
    
    ex2_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,)

    test_loss, test_accuracy = ex2_model.evaluate(test_ds)
    return test_loss, test_accuracy

def tl_experiment_two(base_model, epochs, learning_rate, train_ds, val_ds, test_ds):
    #experiment 3: load the saved model and replace the output layer of the model, as well as the first two convolutional layers (keep weights of all other layers).
    temp_model = keras.models.clone_model(base_model)  #copy the model to get a new object
    temp_model.set_weights(base_model.get_weights())  #transfer weights from old to new model

    layers_to_reset = ['conv2d', 'separable_conv2d'] #swap for whatever other layer you want to reset
    for layer in temp_model.layers:
        if layer.name in layers_to_reset:
            print("RESETTING LAYER:", layer.name)
            if isinstance(layer, keras.layers.Conv2D):
                #For Conv2D, reset kernel and bias
                new_kernel = keras.initializers.GlorotNormal()(shape=layer.kernel.shape) #shape=layer[0].shape layer.kernel.shape
                new_bias = keras.initializers.Zeros()(shape=layer.bias.shape)
                layer.set_weights([new_kernel, new_bias])
            elif isinstance(layer, keras.layers.SeparableConv2D):
                #SeparableConv2D, reset depthwise_kernel, pointwise_kernel, and bias
                depthwise_shape = layer.depthwise_kernel.shape
                pointwise_shape = layer.pointwise_kernel.shape
                bias_shape = layer.bias.shape
                new_depthwise = keras.initializers.GlorotNormal()(shape=depthwise_shape)
                new_pointwise = keras.initializers.GlorotNormal()(shape=pointwise_shape)
                new_bias = keras.initializers.Zeros()(shape=bias_shape)
                layer.set_weights([new_depthwise, new_pointwise, new_bias])
        else:
            print(f"Skipping layer {layer.name} as it is not in layers_to_reset.")

    x = temp_model.get_layer('dropout').output
    new_output = keras.layers.Dense(1, name='dense', activation=None)(x)
    #swap output layer and use new input layers
    ex3_model = keras.Model(inputs=temp_model.input, outputs=new_output)
    #ex3_model.summary()

    ex3_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],)
    
    callbacks = [keras.callbacks.ModelCheckpoint("trained_models/ex3_model_{epoch}.keras"),]

    ex3_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,)
    
    #eval
    test_loss, test_accuracy = ex3_model.evaluate(test_ds)
    return test_loss, test_accuracy

def main():
    #filter out corrupted images
    dm.filter_images(path="data/kagglecatsanddogs_5340/PetImages") #uncomment if running first time
    
    IMAGE_SIZE = (180, 180)
    BATCH_SIZE = 128

    train_ds, val_ds, test_ds = dm.train_val_test_split(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, path="data/kagglecatsanddogs_5340/PetImages")
    #dm.vis_data(train_ds)

    #set how you want to augment the data
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),]
    # Apply `data_augmentation` to the training images.
    train_ds = train_ds.map(
        lambda img, label: (dm.data_augmentation(img, data_augmentation_layers), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf_data.AUTOTUNE)

    #dm.vis_data_augmentations(train_ds, data_augmentation_layers)

    ##### EXPERIMENTS #####

    #load base model
    stanford_model = keras.models.load_model("trained_models/stanford_dogs_epoch_50.keras")
    #stanford_model.summary() #useful to see what current model looks like
    
    EPOCHS=10
    LR=1e-4

    test_loss_ex1, test_accuracy_ex1 = tl_experiment_one(stanford_model, EPOCHS, LR, train_ds, val_ds, test_ds)
    test_loss_ex2, test_accuracy_ex2 = tl_experiment_two(stanford_model, EPOCHS, LR, train_ds, val_ds, test_ds)

    print(f"Test Loss ex2_model: {test_loss_ex1:.4f}")
    print(f"Test Accuracy ex2_model: {test_accuracy_ex1:.4f}")

    print(f"Test Loss ex3_model: {test_loss_ex2:.4f}")
    print(f"Test Accuracy ex3_model: {test_accuracy_ex2:.4f}")

if __name__ == "__main__":
    main()
