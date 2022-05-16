import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import os
import numpy as np
import random


def root_mean_squared_error(y_true, y_pred):
    """RMSE"""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def reset_random_seeds(seed=2022):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model(X_train, y_train, input_shape):
    reset_random_seeds()
    # model hyperparameters
    l = 0.00039105919431420147
    batch_size = 20616
    epochs = 67
    learning_rate = 0.007676170623122377
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)
    activation_layers = "relu"
    activation_prediction = "tanh"
    dropout = 0.1062921408338591
    patience = 30
    units_layer_1 = 2048
    units_layer_2 = 512
    metric = tf.keras.metrics.RootMeanSquaredError()
    loss = root_mean_squared_error

    model = Sequential(
        [
            Dense(
                units=units_layer_1,
                activation=activation_layers,
                kernel_regularizer=tf.keras.regularizers.l2(l=l),
                input_shape=input_shape,
            ),
            Dropout(dropout, seed=10),
            Dense(units=units_layer_2, activation=activation_layers),
            Dropout(dropout, seed=20),
            Dense(units=y_train.shape[1], activation=activation_prediction),
        ]
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.summary()

    # train the model
    history = model.fit(
        X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True
    )

    # plot metric vs epoh chart
    plt.plot(history.history["root_mean_squared_error"])
    plt.ylabel("RMSE")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.yscale("log")
    plt.grid(ls="--")

    return model
