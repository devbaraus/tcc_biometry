from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from praudio import utils


def build_perceptron(output_size: int, shape_size, dense1=512, dense2=256, dense3=128, learning_rate=0.0001):
    model = keras.Sequential()

    model.add(keras.layers.Flatten(
        input_shape=shape_size[1:]))

    # dense 90
    model.add(keras.layers.Dense(dense1, activation='relu',
              kernel_initializer='he_uniform'))
    # dropout 0.3
    # dense 90
    # model.add(keras.layers.Dense(dense2, activation='relu', kernel_initializer='he_uniform'))
    # dropout 0.3
    # dense 10
    # model.add(keras.layers.Dense(dense3, activation='relu', kernel_initializer='he_uniform'))
    model.add(keras.layers.Dense(output_size, activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model

    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set

    :return history: Training history
    """

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history



