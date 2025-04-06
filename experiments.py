from tensorflow import data as tf_data
from tensorflow.keras import layers
import tensorflow.keras as keras
import numpy as np
import datamodule as dm
from model import make_model
from settings import NUM_CLASSES_STANFORD_DOGS, NUM_CLASSES_CATS_VS_DOGS, IMAGE_SIZE

class TestMetricsCallback(keras.callbacks.Callback):
    def __init__(self, test_ds):
        super().__init__()
        self.test_ds = test_ds
        self.epoch_test_loss = []
        self.epoch_test_acc = []
    
    # for storing epoch losses and accuracies
    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_ds, verbose=0)
        self.epoch_test_loss.append(loss)
        self.epoch_test_acc.append(acc)

def tl_ex1(data_path_catsvdogs, data_path_stanford, epochs, lr, batch_size):
    """Experiment 1: Train and evaluate base cats vs dogs model. Train and evaluate the stanford model."""
    # barebone cnn on the cats vs dogs
    dm.filter_images(path=data_path_catsvdogs)
    train_ds_cvd, val_ds_cvd, test_ds_cvd = dm.train_val_test_split(image_size=IMAGE_SIZE, batch_size=batch_size, path=data_path_catsvdogs)
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]

    train_ds_cvd = train_ds_cvd.map(
        lambda img, label: (dm.data_augmentation(img, data_augmentation_layers), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    train_ds_cvd = train_ds_cvd.prefetch(tf_data.AUTOTUNE)
    val_ds_cvd = val_ds_cvd.prefetch(tf_data.AUTOTUNE)
    test_ds_cvd = test_ds_cvd.prefetch(tf_data.AUTOTUNE)

    cvd = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES_CATS_VS_DOGS)

    ckpt_path_cvd = "trained_models/best_models/cvd_base_model_best.keras"

    cvd.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    val_cb = keras.callbacks.ModelCheckpoint(
        ckpt_path_cvd,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    test_cb_cvd = TestMetricsCallback(test_ds_cvd)
    history_cvd = cvd.fit(
        train_ds_cvd,
        epochs=epochs,
        callbacks=[val_cb, test_cb_cvd],
        validation_data=val_ds_cvd,
    )
    best_model_cvd = keras.models.load_model(ckpt_path_cvd)
    final_test_loss_cvd, final_test_acc_cvd = best_model_cvd.evaluate(test_ds_cvd, verbose=0)
    results_cvd = {
        "epoch_train_loss": history_cvd.history["loss"],
        "epoch_train_acc": history_cvd.history.get("binary_accuracy"),
        "epoch_val_loss": history_cvd.history["val_loss"],
        "epoch_val_acc": history_cvd.history.get("val_binary_accuracy"),
        "epoch_test_loss": test_cb_cvd.epoch_test_loss,
        "epoch_test_acc": test_cb_cvd.epoch_test_acc,
        "final_test_loss": final_test_loss_cvd,
        "final_test_acc": final_test_acc_cvd,
    }
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy('mixed_float16')
    # stanford model
    dm.filter_images(path=data_path_stanford)

    train_ds_stanford, val_ds_stanford, test_ds_stanford = dm.train_val_test_split(image_size=IMAGE_SIZE, batch_size=batch_size, path=data_path_stanford)

    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]

    train_ds_stanford = train_ds_stanford.map(
        lambda img, label: (dm.data_augmentation(img, data_augmentation_layers), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )

    train_ds_stanford = train_ds_stanford.prefetch(tf_data.AUTOTUNE)
    val_ds_stanford = val_ds_stanford.prefetch(tf_data.AUTOTUNE)
    test_ds_stanford = test_ds_stanford.prefetch(tf_data.AUTOTUNE)

    stanford = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES_STANFORD_DOGS)

    ckpt_path_stanford = "trained_models/best_models/stanford_model_best.keras"

    stanford.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    val_cb = keras.callbacks.ModelCheckpoint(
        ckpt_path_stanford,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    test_cb_stanford = TestMetricsCallback(test_ds_stanford)
    history_stanford = stanford.fit(
        train_ds_stanford,
        epochs=epochs,
        callbacks=[val_cb, test_cb_stanford],
        validation_data=val_ds_stanford,
    )

    stanford_model_best = keras.models.load_model(ckpt_path_stanford)
    test_loss_stanford, test_acc_stanford = stanford_model_best.evaluate(test_ds_stanford)
    results_stanford = {
        "epoch_train_loss": history_stanford.history["loss"],
        "epoch_train_acc": history_stanford.history.get("acc", history_stanford.history.get("sparse_categorical_accuracy")),
        "epoch_val_loss": history_stanford.history["val_loss"],
        "epoch_val_acc": history_stanford.history.get("val_acc", history_stanford.history.get("val_sparse_categorical_accuracy")),
        "epoch_test_loss": test_cb_stanford.epoch_test_loss,
        "epoch_test_acc": test_cb_stanford.epoch_test_acc,
        "final_test_loss": test_loss_stanford,
        "final_test_acc": test_acc_stanford,
    }
    results = {
        "cvd": results_cvd,
        "stanford": results_stanford
    }
    return stanford_model_best, (train_ds_cvd, val_ds_cvd, test_ds_cvd), results

def tl_ex2(stanford_model, epochs, lr, train_ds, val_ds, test_ds):
    temp_model = keras.models.clone_model(stanford_model)  #copy the model to get a new object
    temp_model.set_weights(stanford_model.get_weights())  #transfer weights from old to new model

    ex2_inputs = temp_model.input

    #get output before last layer
    x = temp_model.get_layer('dropout').output
    
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

    val_cb = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    test_cb = TestMetricsCallback(test_ds)
    history = ex2_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=[val_cb, test_cb],
        validation_data=val_ds
    )

    ex2_model_best = keras.models.load_model(ckpt_path)
    test_loss, test_acc = ex2_model_best.evaluate(test_ds)
    results = {
        "epoch_train_loss": history.history["loss"],
        "epoch_train_acc": history.history.get("binary_accuracy"),
        "epoch_val_loss": history.history["val_loss"],
        "epoch_val_acc": history.history.get("val_binary_accuracy"),
        "epoch_test_loss": test_cb.epoch_test_loss,
        "epoch_test_acc": test_cb.epoch_test_acc,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
    }
    return results

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

    val_cb = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    test_cb = TestMetricsCallback(test_ds)
    history = ex3_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=[val_cb, test_cb],
        validation_data=val_ds
    )

    ex3_model_best = keras.models.load_model(ckpt_path)
    test_loss, test_acc = ex3_model_best.evaluate(test_ds)
    results = {
        "epoch_train_loss": history.history["loss"],
        "epoch_train_acc": history.history.get("binary_accuracy"),
        "epoch_val_loss": history.history["val_loss"],
        "epoch_val_acc": history.history.get("val_binary_accuracy"),
        "epoch_test_loss": test_cb.epoch_test_loss,
        "epoch_test_acc": test_cb.epoch_test_acc,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
    }
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
    val_cb = keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    test_cb = TestMetricsCallback(test_ds)
    history = ex4_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=[val_cb, test_cb],
        validation_data=val_ds
    )

    ex4_model_best = keras.models.load_model(ckpt_path)
    test_loss, test_acc = ex4_model_best.evaluate(test_ds)
    results = {
        "epoch_train_loss": history.history["loss"],
        "epoch_train_acc": history.history.get("binary_accuracy"),
        "epoch_val_loss": history.history["val_loss"],
        "epoch_val_acc": history.history.get("val_binary_accuracy"),
        "epoch_test_loss": test_cb.epoch_test_loss,
        "epoch_test_acc": test_cb.epoch_test_acc,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
    }
    return results
