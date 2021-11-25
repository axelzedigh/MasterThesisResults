import os
import sys
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as keras_backend
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from numba import jit

from plots.plot_functions import plot_history_log
from utils.db_utils import get_training_trace_path__raw_200k_data
from utils.denoising_utils import moving_average_filter_n3, \
    moving_average_filter_n5
from utils.statistic_utils import maxmin_scaling_of_trace_set__per_trace_fit
from utils.trace_utils import get_training_model_file_save_path


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
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return sequential_model


def train_model(
        x_profiling,
        y_profiling,
        deep_learning_model,
        model_save_path,
        epochs,
        batch_size,
) -> Callable:
    """

    :param x_profiling:
    :param y_profiling:
    :param deep_learning_model:
    :param model_save_path:
    :param epochs:
    :param batch_size:
    :return: History-function.
    """
    # Check if file-path exists
    check_if_file_exists(os.path.dirname(model_save_path))

    # Save model every epoch
    save_model = ModelCheckpoint(model_save_path)
    callbacks = [save_model]

    # Get the input layer shape
    input_layer_shape = deep_learning_model.get_layer(index=0).input_shape

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
    else:
        print(
            f"Error: model input shape length "
            f"{len(input_layer_shape)} is not expected ...")
        sys.exit(-1)

    history = deep_learning_model.fit(
        x=reshaped_x_profiling,
        y=reshaped_y_profiling,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.1
    )
    return history


@jit
def cut_trace_set__column_range(
        trace_set, range_start=204, range_end=314
) -> np.array:
    """
    :param trace_set: Trace set to cut.
    :param range_start: Start column position.
    :param range_end: End column position.
    :return:
    """
    assert (range_start and range_end) < len(trace_set)
    return trace_set[:, range_start:range_end]


def additive_noise_to_trace_set(
        trace_set: np.array,
        additive_noise_method_id: int
) -> Tuple[np.array, np.array]:
    """
    :param trace_set: Trace set to add noise to.
    :param additive_noise_method_id: Additive noise to add (according to table).
    :return: Trace set with noise and example noise trace.
    """
    if additive_noise_method_id is None:
        return trace_set
    elif additive_noise_method_id == 1:
        return additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.01)
    elif additive_noise_method_id == 2:
        return additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.02)
    elif additive_noise_method_id == 3:
        return additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.03)
    elif additive_noise_method_id == 4:
        return additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.04)
    elif additive_noise_method_id == 5:
        return additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.05)
    elif additive_noise_method_id == 6:
        return additive_noise__collected_noise__office_corridor(
            trace_set=trace_set, scaling_factor=25
        )
    elif additive_noise_method_id == 7:
        return additive_noise__collected_noise__office_corridor(
            trace_set=trace_set, scaling_factor=50
        )
    elif additive_noise_method_id == 8:
        return additive_noise__collected_noise__office_corridor(
            trace_set=trace_set, scaling_factor=75
        )
    elif additive_noise_method_id == 9:
        return additive_noise__collected_noise__office_corridor(
            trace_set=trace_set, scaling_factor=105
        )
    elif additive_noise_method_id == 10:
        return additive_noise__rayleigh(trace_set=trace_set, mode=0.0138)
    elif additive_noise_method_id == 11:
        return additive_noise__rayleigh(trace_set=trace_set, mode=0.0276)


@jit
def additive_noise__gaussian(
        trace_set: np.array, mean: float, std: float
) -> Tuple[np.array, np.array]:
    """
    Applies Gaussian distributed noise to the trace set.
    :param trace_set: The training trace set.
    :param mean: µ of the noise (usually 0)
    :param std: ∂ of the noise.
    :return: Trace set with Gaussian distributed noise and example noise trace.
    """
    noise_traces = trace_set.copy()
    ex_additive_noise_trace = np.random.normal(mean, std, trace_set.shape[1])
    for i in range(len(trace_set)):
        gaussian_noise = np.random.normal(mean, std, trace_set.shape[1])
        noise_traces[i] += gaussian_noise
    return noise_traces, ex_additive_noise_trace


@jit
def additive_noise__rayleigh(
        trace_set: np.array, mode: float
) -> Tuple[np.array, np.array]:
    """
    Applies Rayleigh distributed noise to the trace set.
    :param trace_set: The training trace set.
    :param mode: The mode of the distribution.
    :return: Trace set with Rayleigh distributed noise and example noise trace.
    """
    noise_traces = trace_set.copy()
    example_additive_noise_trace = np.random.rayleigh(mode, trace_set.shape[1])
    for i in range(len(trace_set)):
        rayleigh_noise = np.random.rayleigh(mode, trace_set.shape[1])
        noise_traces[i] += rayleigh_noise
    return noise_traces, example_additive_noise_trace


def additive_noise__collected_noise__office_corridor(
        trace_set: np.array,
        scaling_factor: float,
        mean_adjust: bool = False,
) -> Tuple[np.array, np.array]:
    """
    Applies collected noise to the trace set.
    :param trace_set: Needs to have column-size 400 atm! TODO: fix?
    :param scaling_factor: Scaling factor of the noise.
    :param mean_adjust:
    :return: Trace set with additive collected noise and example noise trace.
    """
    # Load the office corridor noise.
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    noise_trace_path = os.path.join(
        project_dir,
        "datasets/test_traces/Zedigh_2021/office_corridor/Noise",
        "data",
        "traces.npy"
    )
    noise_set = np.load(noise_trace_path)

    # Remove outlier noise traces from the set.
    index_1 = int(512400 / 400)
    index_2 = int(1824800 / 400)
    noise_set[index_1] = noise_set[index_1 - 20]
    noise_set[index_2] = noise_set[index_2 - 20]

    # Q: Randomize the data points in the collected trace?
    # A: No. Not atm.

    # Make noise trace set equally long as training trace set.
    multiplier = int(len(trace_set) / len(noise_set))
    noise_set = np.tile(noise_set, (multiplier, 1))

    # Q: Transform noise to have mean around 0?
    # A: No. Not atm.
    if mean_adjust:
        pass

    # Scale the noise
    noise_set *= scaling_factor

    # Get example additive noise trace
    example_additive_noise_trace = noise_set[1]

    # Apply the noise to trace set
    for i in range(len(trace_set)):
        trace_set[i] += noise_set[i]

    return trace_set, example_additive_noise_trace


def denoising_of_trace_set(
        trace_set,
        denoising_method_id
) -> Tuple[np.array, int, int, np.array]:
    """
    :param trace_set:
    :param denoising_method_id:
    :return:
    """
    example_not_denoised_trace = trace_set[1].copy()
    if denoising_method_id == 1:
        filtered_set, range_start, range_end = moving_average_filter_n3(
            trace_set
        )
        return filtered_set, range_start, range_end, example_not_denoised_trace
    if denoising_method_id == 2:
        filtered_set, range_start, range_end = moving_average_filter_n5(
            trace_set
        )
        return filtered_set, range_start, range_end, example_not_denoised_trace
    else:
        raise f"Denoising method id {denoising_method_id} is not correct."


def training_cnn_110(
        keybyte: int = 0,
        epochs: int = 100,
        batch_size: int = 256,
        additive_noise_method_id: Optional[int] = None,
        denoising_method_id: Optional[int] = None,
        trace_process_id: int = 3,
        verbose: bool = False,
) -> None:
    """
    The main function for training the CNN 110 classifier.
    Uses training traces from Wang_2021 (5 devices).
    Only CNN with input size 110 and output size 256 is used now.

    :param keybyte: The keybyte classifier to train.
    :param epochs: Number of epochs to perform.
    :param batch_size: The batch-size used in training.
    :param additive_noise_method_id: Id to additive noise method.
    :param denoising_method_id: Id to denoising method.
    :param trace_process_id: The trace pre-process done.
    :param verbose: Show plot and extra information if true.
    :return: None.
    """
    # Initialise variables
    raw_data_path = os.getenv("MASTER_THESIS_RESULTS_RAW_DATA")
    training_model_id = 1
    additive_noise_trace = None
    clean_trace = None
    start = 204
    end = 314
    if trace_process_id in [6]:
        # start = 200
        # end = 320
        # start = 209
        # end = 329
        start = 130
        end = 240

    # Get training traces numpy array.
    training_set_path = get_training_trace_path__raw_200k_data()
    if trace_process_id == 3:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 4 or trace_process_id == 5:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 6:
        trace_set_file_path = os.path.join(
            raw_data_path,
            "datasets/training_traces/Zedigh_2021/Cable/100k_5devices_joined",
            "nor_maxmin_traces__130_240.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    else:
        raise "Trace process id is not correct."

    if trace_process_id == 6:
        labels_path = os.path.join(
            raw_data_path,
            "datasets/training_traces/Zedigh_2021/Cable/100k_5devices_joined",
            "labels.npy"
        )
        labels = np.load(labels_path)
    else:
        # Get cipher-text numpy array..
        cipher_text_file_path = os.path.join(
            training_set_path, "ct.npy"
        )
        cipher_text = np.load(cipher_text_file_path)

        # Get last round key numpy array, transform to labels array.
        last_roundkey_file_path = os.path.join(
            training_set_path, "lastroundkey.npy"
        )
        last_roundkey = np.load(last_roundkey_file_path)
        last_roundkey = last_roundkey.astype(int)
        last_round_sbox_output = np.bitwise_xor(
            cipher_text[:, keybyte], last_roundkey[:, keybyte]
        )
        labels = last_round_sbox_output

    # Get path to store model
    model_save_file_path = get_training_model_file_save_path(
        keybyte=keybyte,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        training_model_id=training_model_id,
        trace_process_id=trace_process_id
    )

    # Get the DL-model
    if training_model_id == 1:
        deep_learning_model = cnn_110_model()
    else:
        raise "No other model is currently investigated."
    if verbose:
        print(deep_learning_model.summary())

    # Apply additive noise
    if additive_noise_method_id is not None:
        training_trace_set, additive_noise_trace = additive_noise_to_trace_set(
            trace_set=training_trace_set,
            additive_noise_method_id=additive_noise_method_id
        )

    # Denoise the trace set.
    if denoising_method_id is not None:
        training_trace_set, start, end, clean_trace = denoising_of_trace_set(
            trace_set=training_trace_set,
            denoising_method_id=denoising_method_id,
        )

    # Cut trace set to the sbox output range
    training_trace_set = cut_trace_set__column_range(
        trace_set=training_trace_set,
        range_start=start,
        range_end=end,
    )

    # Normalize the trace set in sbox range
    if trace_process_id == 4:
        training_trace_set = maxmin_scaling_of_trace_set__per_trace_fit(
            trace_set=training_trace_set,
            range_start=0,
            range_end=len(training_trace_set[1])
        )

    # Plot the traces as a final check
    if verbose:
        plt.plot(training_trace_set[0], color="deepskyblue")
        plt.plot(training_trace_set[1], color="seagreen")
        plt.plot(training_trace_set[2], color="blueviolet")
        if additive_noise_method_id is not None:
            plt.plot(additive_noise_trace[start:end], color="lightcoral")
        if denoising_method_id is not None:
            plt.plot(clean_trace[start:end], color="orange")
        trace_fig_save_path_dir = os.path.dirname(model_save_file_path)
        trace_fig_file_path = os.path.join(
            trace_fig_save_path_dir,
            "training_trace_and_processing_attribute.png"
        )
        plt.savefig(fname=trace_fig_file_path)

    # Train the model
    history_log = train_model(
        x_profiling=training_trace_set,
        y_profiling=labels,
        deep_learning_model=deep_learning_model,
        model_save_path=model_save_file_path,
        epochs=epochs,
        batch_size=batch_size
    )

    # Store the accuracy and loss data
    model_save_path_dir = os.path.dirname(model_save_file_path)
    history_log_file_path = os.path.join(model_save_path_dir, "history_log.npy")
    np.save(history_log_file_path, history_log.history)

    if verbose:
        plot_history_log(
            trace_process_id=trace_process_id,
            keybyte=keybyte,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
        )

    return
