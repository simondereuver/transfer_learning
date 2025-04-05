import os
from tabulate import tabulate
from tensorflow import data as tf_data
from tensorflow.keras import layers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import datamodule as dm
from model import make_model


NUM_CLASSES_STANFORD_DOGS = 120
IMAGE_SIZE=(180, 180)

def tl_ex1(data_path, epochs, lr, batch_size):
    """implement stanford_model.py here instead. don't save the stanford model just return it (the best one)"""
    dm.filter_images(path=data_path)

    train_ds, val_ds = dm.train_val_split(image_size=IMAGE_SIZE, batch_size=batch_size, path=data_path)

    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]

    train_ds = train_ds.map(
        lambda img, label: (dm.data_augmentation(img, data_augmentation_layers), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    stanford = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES_STANFORD_DOGS)

    ckpt_path = "trained_models/best_models/stanford_model_best.keras"

    stanford.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    callbacks = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    stanford.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )

    stanford_model_best = keras.models.load_model(ckpt_path)
    val_loss, val_acc = stanford_model_best.evaluate(val_ds)
    results = {"Validation loss": val_loss, "Validation accuracy": val_acc}
    return stanford_model_best, results

def tl_ex2(data_path, stanford_model, epochs, lr, batch_size):
    dm.filter_images(path=data_path)

    train_ds, val_ds, test_ds = dm.train_val_test_split(image_size=IMAGE_SIZE, batch_size=batch_size, path=data_path)

    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
        ]

    train_ds = train_ds.map(
        lambda img, label: (dm.data_augmentation(img, data_augmentation_layers), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf_data.AUTOTUNE)
    ex2_inputs = stanford_model.input

    #get output before last layer
    x = stanford_model.get_layer('dropout').output
    
    #connect to output layer
    new_output = layers.Dense(1, activation=None)(x)

    #create the new model
    ex2_model = keras.Model(inputs=ex2_inputs, outputs=new_output)
    
    #train and evaluate the model (for 50 epochs) on the cats and dogs dataset.
    ex2_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],)
    
    ckpt_path = "trained_models/best_models/experiment2_best.keras"

    callbacks = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    ex2_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds
    )

    ex2_model_best = keras.models.load_model(ckpt_path)
    test_loss, test_acc = ex2_model_best.evaluate(test_ds)
    results = {"Test loss": test_loss, "Test accuracy": test_acc}
    return (train_ds, val_ds, test_ds), results

def tl_ex3(stanford_model, epochs, lr, train_ds, val_ds, test_ds):
    #experiment 3: load the saved model and replace the output layer of the model, as well as the first two convolutional layers (keep weights of all other layers).
    temp_model = keras.models.clone_model(stanford_model)  #copy the model to get a new object
    temp_model.set_weights(stanford_model.get_weights())  #transfer weights from old to new model

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
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    
    ckpt_path = "trained_models/best_models/experiment3_best.keras"

    callbacks = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    ex3_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds
    )

    ex3_model_best = keras.models.load_model(ckpt_path)
    test_loss, test_acc = ex3_model_best.evaluate(test_ds)
    results = {"Test loss": test_loss, "Test accuracy": test_acc}
    return results

def tl_ex4(stanford_model, epochs, lr, train_ds, val_ds, test_ds):
    """Experiment 4: Load the saved model and replace the output layer as well as the last two convolutional layers."""
    temp_model = keras.models.clone_model(stanford_model)
    temp_model.set_weights(stanford_model.get_weights())

    layers_to_reset = ['conv2d_3', 'separable_conv2d_6'] #swap for whatever other layer you want to reset
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
    ex4_model = keras.Model(inputs=temp_model.input, outputs=new_output)

    ex4_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()]
    )

    ckpt_path = "trained_models/best_models/experiment4_best.keras"
    callbacks = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    ex4_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds
    )

    ex4_model_best = keras.models.load_model(ckpt_path)
    test_loss, test_acc = ex4_model_best.evaluate(test_ds)
    results = {"Test loss": test_loss, "Test accuracy": test_acc}
    return results

def main():
    """Main loop handles execution of all experiments. Each experiment is divided into a separate function returning best model and the results."""
    ##### EXPERIMENTS #####
    # pre
    EPOCHS=50
    LR=1e-4
    BATCH_SIZE = 64

    # experiment 1
    print("Starting experiment 1 ...")
    data_path_stanford = "data/stanford_dogs/Images"

    stanford_model, results1 = tl_ex1(data_path_stanford, EPOCHS, LR, BATCH_SIZE)
    for key, value in results1.items():
        print(f"{key}: {value}")
    stanford_model.summary()
    print("Experiment 1 finished.")

    # experiment 2
    print("Starting experiment 2 ...")
    data_path_binary_class = "data/PetImages"
    
    datasets, results2 = tl_ex2(data_path_binary_class, stanford_model, EPOCHS, LR, BATCH_SIZE)
    for key, value in results2.items():
        print(f"{key}: {value}")

    print("Experiment 2 finished.")

    # experiment 3
    print("Starting experiment 3 ...")

    results3 = tl_ex3(stanford_model, EPOCHS, LR, *datasets)
    for key, value in results3.items():
        print(f"{key}: {value}")

    print("Experiment 3 finished.")

    # experiment 4
    print("Starting experiment 4 ...")

    results4 = tl_ex4(stanford_model, EPOCHS, LR, *datasets)
    for key, value in results4.items():
        print(f"{key}: {value}")

    print("Experiment 4 finished.")

    # combine all results from each experiment and print in a tabular table with tablefmt="mixed_outline"
    # and put each experiment result on a new row, and the val/test accuracy and loss in the columns
    # where if a results dict does not contain test or val put NaN instead. Also save to results/results.txt 

    rows = []
    for i, res in enumerate([results1, results2, results3, results4], start=1):
        val_loss = res.get("Validation loss", "NaN")
        val_acc  = res.get("Validation accuracy", "NaN")
        test_loss = res.get("Test loss", "NaN")
        test_acc  = res.get("Test accuracy", "NaN")
        rows.append([f"Experiment {i}", val_loss, val_acc, test_loss, test_acc])

    headers = ["Experiment", "Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"]

    table_str = tabulate(rows, headers=headers, tablefmt="mixed_outline")
    print(table_str)

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/results.txt", "w") as f:
        f.write(table_str)

    print("All experiments finished, saved results to results/results.txt")

if __name__ == "__main__":
    main()
