import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as keras_backend
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tqdm import tqdm


def check_if_file_exists(file_path):
    """
    Helper function to check if a file-path exists.
    :param file_path: path to file.
    :return:
    """
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print(f"Error: provided file path {file_path} does not exist!")
        sys.exit(-1)
    return


def mean_squared_error(y_true, y_predicted):
    """

    :param y_true: The true value.
    :param y_predicted: The predicted value.
    :return: The mean square error.
    """
    return keras_backend.mean(
        keras_backend.square(y_predicted - y_true), axis=-1
    )


def cnn_110_model(classes=256):
    """
    CNN with input size 110, batch-size 128.
    :param classes:
    :return:
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
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return sequential_model


def train_model(
        x_profiling,
        y_profiling,
        dl_model,
        save_file_name,
        epochs,
        batch_size,
        model_type
):
    """

    :param x_profiling:
    :param y_profiling:
    :param dl_model:
    :param save_file_name:
    :param epochs:
    :param batch_size:
    :param model_type:
    :return:
    """
    check_if_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)
    callbacks = [save_model]

    # Get the input layer shape
    input_layer_shape = dl_model.get_layer(index=0).input_shape

    # Sanity check
    if input_layer_shape[1] != len(x_profiling[0]):
        print(
            f"Error: model input shape {input_layer_shape[1]} instead of "
            f"{len(x_profiling[0])} is not expected ..."
        )
        sys.exit(-1)
    elif len(input_layer_shape) == 3:
        reshaped_x_profiling = x_profiling.reshape(
            (x_profiling.shape[0], x_profiling.shape[1], 1)
        )
        reshaped_y_profiling = to_categorical(y_profiling, num_classes=256)
        if model_type == 6:
            reshaped_y_profiling = y_profiling.reshape(
                (y_profiling.shape[0], y_profiling.shape[1], 1)
            )
    else:
        print(
            f"Error: model input shape length "
            f"{len(input_layer_shape)} is not expected ...")
        sys.exit(-1)

    history = dl_model.fit(x=reshaped_x_profiling,
                           y=reshaped_y_profiling,
                           batch_size=batch_size,
                           verbose=1,
                           epochs=epochs,
                           callbacks=callbacks,
                           validation_split=0.1
                           )
    return history

MODEL_TYPE = 2 #CNN
INTEREST_BYTE = 0
MODEL_FOLDER = "cnn_110_maf_n3/"
TRAINING_MODEL = MODEL_FOLDER + 'cnn_model-{epoch:01d}.h5'
USER = os.getenv("USER")
traces = np.load('data/nor_traces_maxmin.npy')
#traces = traces[:,[i for i in range(204,314)]]


#traces_fixed = traces.copy()
#noise_traces_flattened_fixed = traces_fixed.flatten()

noise_data = np.load(f"/Users/{USER}/Documents/MASTER-THESIS/datasets/last_round_aes/experiment_axel/test_1/traces.npy")
index_1 = int(512400 / 400)
index_2 = int(1824800 / 400)
noise_data[index_1] = noise_data[index_1 -20]
noise_data[index_2] = noise_data[index_2 -20]
noise_data = noise_data[:,[i for i in range(204,314)]]
noise_data_flatten = noise_data.flatten()
scale = 75
translation = 0
noise_data = (noise_data - translation) * scale

nr = int(len(traces) / len(noise_data))
noise_data_2 = np.tile(noise_data, (nr,1))


# Add noise traces to traces
noise_traces = traces.copy()

for i in range(len(traces)):
    noise_traces[i] = noise_data_2[i] + noise_traces[i]

# Add modelled noise to training traces
# Rayleigh

for i in range(len(traces)):
    #noise = np.random.rayleigh(0.0138*2,110)
    noise = np.random.normal(0,0.05,110)
    noise_traces[i] = noise_traces[i] + noise


# MAF filter training data

def moving_avg_trace(trace, n):
    #trace = trace.copy()
    cumsum = np.cumsum(np.insert(trace, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

#filtered_traces = traces.copy()
filtered_traces = np.empty_like(traces)

for i in tqdm(range(len(traces))):
    #np.append(filtered_traces, moving_avg_trace(traces[i], 3))
    filtered_traces[i] = np.pad(moving_avg_trace(traces[i], 3), (0,2),'constant')
filtered_traces = filtered_traces[:,[i for i in range(203,313)]]


gauss_noise = np.random.normal(0,0.02,110)
rayleigh_noise = np.random.rayleigh(0.02, 110)

#plt.plot(traces[0])
plt.plot(filtered_traces[0])
plt.plot(filtered_traces[2])
#plt.plot(noise_traces[2])
#plt.plot(noise_data_2[0])
#plt.plot(gauss_noise)
#plt.plot(noise)
#plt.axvline(x=204)
#plt.axvline(x=314)
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.grid(True)
plt.show()


ct = np.load('data/ct.npy')
key=np.load('data/lastroundkey.npy')
key= key.astype(int)
#print(key[:,interest_byte].shape)
lastround_sboxout= np.bitwise_xor(ct[:,INTEREST_BYTE],key[:,INTEREST_BYTE])
#lastround_input= Inv_SBox[lastround_sboxout]
labels=lastround_sboxout

# CNN model
model = cnn_110_model()
print(model.summary())

EPOCHS = 100
BATCH_SIZE = 256


history_log = train_model(filtered_traces, labels, model, TRAINING_MODEL, EPOCHS, BATCH_SIZE, MODEL_TYPE)
np.save(MODEL_FOLDER + "history_log.npy", history_log.history)