from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, AveragePooling1D, Flatten, \
    Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import CategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop


def cnn_110_model(classes=256):
    """
    CNN with input size 110.
    :param classes:
    :return: Keras/TF sequential CNN model with input size 110, classes 256.
    """
    sequential_model = Sequential()
    sequential_model.add(
        Conv1D(
            input_shape=(110, 1),
            filters=4,
            kernel_size=3,
            activation='relu',
            padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Conv1D(
        filters=8,
        kernel_size=3,
        activation='relu',
        padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Flatten())
    # model.add(Dropout(0.2))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(
        Dense(units=classes, activation='softmax', name='predictions')
    )
    optimizer = RMSprop(lr=0.00005)
    sequential_model.compile(
        loss=CategoricalCrossentropy(name="loss"),
        optimizer=optimizer,
        # metrics=['accuracy']
        metrics=[CategoricalAccuracy(name="accuracy")],
    )
    return sequential_model


def cnn_110_sgd_model(classes=256):
    """
    CNN with input size 110.
    :param classes:
    :return: Keras/TF sequential CNN model with input size 110, classes 256.
    """
    sequential_model = Sequential()
    sequential_model.add(
        Conv1D(
            input_shape=(110, 1),
            filters=4,
            kernel_size=3,
            activation='relu',
            padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Conv1D(
        filters=8,
        kernel_size=3,
        activation='relu',
        padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Flatten())
    # model.add(Dropout(0.2))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(
        Dense(units=classes, activation='softmax', name='predictions')
    )
    optimizer = SGD(learning_rate=0.0001, name="SGD", nesterov=True)
    sequential_model.compile(
        loss=CategoricalCrossentropy(name="loss"),
        optimizer=optimizer,
        metrics=[CategoricalAccuracy(name="accuracy")],
    )

    return sequential_model


def cnn_110_model_grid_search():
    """
    CNN with input size 110.
    :return: Keras/TF sequential CNN model with input size 110, classes 256.
    """
    sequential_model = Sequential()
    sequential_model.add(
        Conv1D(
            input_shape=(110, 1),
            filters=4,
            kernel_size=3,
            activation='relu',
            padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Conv1D(
        filters=8,
        kernel_size=3,
        activation='relu',
        padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Flatten())
    # model.add(Dropout(0.2))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(
        Dense(units=256, activation='softmax', name='predictions')
    )
    optimizer = RMSprop(lr=0.00005)
    sequential_model.compile(
        loss=CategoricalCrossentropy(name="loss"),
        optimizer=optimizer,
        # metrics=['accuracy']
        metrics=[CategoricalAccuracy(name="accuracy")],
    )
    return sequential_model


def cnn_110_model_simpler(classes=256):
    """
    CNN with input size 110.
    :param classes:
    :return: Keras/TF sequential CNN model with input size 110, classes 256.
    """
    sequential_model = Sequential()
    sequential_model.add(
        Conv1D(
            input_shape=(110, 1),
            filters=4,
            kernel_size=3,
            activation='relu',
            padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Conv1D(
        filters=8,
        kernel_size=3,
        activation='relu',
        padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    # sequential_model.add(
    #     Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
    # )
    # sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Flatten())
    # model.add(Dropout(0.2))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(
        Dense(units=classes, activation='softmax', name='predictions')
    )
    optimizer = RMSprop(lr=0.00005)
    sequential_model.compile(
        loss=CategoricalCrossentropy(name="loss"),
        optimizer=optimizer,
        # metrics=['accuracy']
        metrics=[CategoricalAccuracy(name="accuracy")],
    )
    return sequential_model


def cnn_110_model_more(classes=256):
    """
    CNN with input size 110.
    :param classes:
    :return: Keras/TF sequential CNN model with input size 110, classes 256.
    """
    sequential_model = Sequential()
    sequential_model.add(
        Conv1D(
            input_shape=(110, 1),
            filters=4,
            kernel_size=3,
            activation='relu',
            padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Conv1D(
        filters=8,
        kernel_size=3,
        activation='relu',
        padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
    )
    sequential_model.add(AveragePooling1D(pool_size=2, strides=1))
    sequential_model.add(Flatten())
    # model.add(Dropout(0.2))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(Dense(units=200, activation='relu'))
    sequential_model.add(
        Dense(units=classes, activation='softmax', name='predictions')
    )
    optimizer = RMSprop(lr=0.00005)
    sequential_model.compile(
        loss=CategoricalCrossentropy(name="loss"),
        optimizer=optimizer,
        metrics=[CategoricalAccuracy(name="accuracy")],
    )
    return sequential_model
