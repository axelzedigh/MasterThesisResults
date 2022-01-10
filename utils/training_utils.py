"""Training utils."""
import os
import sys
from typing import Callable, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from keras import backend as keras_backend
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from numba import jit
from tensorflow.python.keras.callbacks import Callback

from configs.variables import PROJECT_DIR, NORD_LIGHT_ORANGE, \
    NORD_LIGHT_MPL_STYLE_PATH
from plots.history_log_plots import plot_history_log
from scripts.model_training.deep_learning_models import cnn_110_model, \
    cnn_110_sgd_model, cnn_110_model_more

from scripts.model_training.deep_learning_models import cnn_110_model_simpler
from utils.db_utils import get_test_trace_path
from utils.denoising_utils import moving_average_filter_n3, \
    moving_average_filter_n5, wiener_filter_trace_set, moving_average_filter_n11
from utils.statistic_utils import maxmin_scaling_of_trace_set__per_trace_fit
from utils.trace_utils import get_training_model_file_save_path, \
    get_training_trace_path, unison_shuffle_traces_and_labels, \
    get_validation_data_path__8m, get_normalized_test_traces


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


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either
    acc or val_acc reaches a specified baseline.
    """
    def __init__(self, monitor='accuracy', baseline=0.0095):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True


class EvaluateCallback(Callback):
    """
    Evaluate on test.
    """
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        y_eval = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test: {y_eval}")


def train_model(
        x_profiling: np.array,
        y_profiling: np.array,
        deep_learning_model: Any,
        model_save_path: str,
        epochs: int,
        batch_size: int,
        mode: int,
        x_validation: Optional[np.array] = None,
        y_validation: Optional[np.array] = None,
) -> Callable:
    """

    :param x_profiling:
    :param y_profiling:
    :param deep_learning_model:
    :param model_save_path:
    :param epochs:
    :param batch_size:
    :param mode:
    :param x_validation:
    :param y_validation:
    :return: History-function.

    """
    # Test trace set path
    test_path = get_test_trace_path(
        database="main.db",
        test_dataset_id=1,
        environment_id=1,
        distance=15,
        device=10
    )

    number_total_trace = 4900
    testing_traces = get_normalized_test_traces(
        trace_process_id=8,
        test_dataset_id=1,
        environment_id=1,
        distance=15,
        device=10,
        save=False
    )
    eval_callback_labels = np.load(os.path.join(test_path, "label_lastround_Sout_0.npy"))
    testing_traces = testing_traces[:number_total_trace]
    testing_traces = cut_trace_set__column_range(trace_set=testing_traces)
    eval_callback_labels = eval_callback_labels[:number_total_trace]
    testing_traces = testing_traces.reshape(
        (testing_traces.shape[0], testing_traces.shape[1], 1)
    )
    eval_callback_labels = to_categorical(eval_callback_labels, 256)

    # Check if file-path exists
    check_if_file_exists(os.path.dirname(model_save_path))

    # Save model every epoch
    if mode == 1:
        save_model = ModelCheckpoint(model_save_path)
        callbacks = [save_model]
    elif mode == 2:
        save_model = ModelCheckpoint(
            model_save_path,
            monitor='acc',
            mode='max',
            save_best_only=True
        )
        callbacks = [save_model]
    elif mode == 3:
        callbacks = [
            ModelCheckpoint(model_save_path),
            EvaluateCallback(x_test=testing_traces, y_test=eval_callback_labels),
        ]
    elif mode == 4:
        callbacks = [
            ModelCheckpoint(model_save_path),
            EvaluateCallback(x_test=testing_traces, y_test=eval_callback_labels),
            TerminateOnBaseline(monitor='val_accuracy', baseline=0.01)
        ]

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

    # Import validation data
    if x_validation is not None and y_validation is not None:
        reshaped_x_val, reshaped_y_val = preprocess_validation_data(
            x_validation=x_validation,
            y_validation=y_validation,
        )

        history = deep_learning_model.fit(
            x=reshaped_x_profiling,
            y=reshaped_y_profiling,
            batch_size=batch_size,
            verbose=1,
            epochs=epochs,
            callbacks=callbacks,
            shuffle=True,
            validation_data=(reshaped_x_val, reshaped_y_val),
        )
        return history
    else:
        history = deep_learning_model.fit(
            x=reshaped_x_profiling,
            y=reshaped_y_profiling,
            batch_size=batch_size,
            verbose=1,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.1,
            shuffle=True,
        )
        return history


def preprocess_validation_data(x_validation, y_validation):
    """

    :param x_validation:
    :param y_validation:
    """
    x_validation, y_validation = unison_shuffle_traces_and_labels(
        trace_set=x_validation,
        labels=y_validation,
    )

    # Use 20k for validation
    x_validation = x_validation[:20000]
    y_validation = y_validation[:20000]

    # Balance the dataset
    undersample = RandomUnderSampler(
        sampling_strategy="all",
        random_state=101
    )
    x_validation, y_validation = undersample.fit_resample(
        x_validation, y_validation
    )

    # Shuffle the dataset
    x_validation, y_validation = unison_shuffle_traces_and_labels(
        trace_set=x_validation,
        labels=y_validation,
    )

    # Reshape y
    reshaped_y_val = to_categorical(y_validation, num_classes=256)

    # Cut and reshape x
    x_validation = cut_trace_set__column_range(
        trace_set=x_validation,
        range_start=130,
        range_end=240,
    )
    reshaped_x_val = x_validation.reshape(
        (x_validation.shape[0], x_validation.shape[1], 1)
    )

    return reshaped_x_val, reshaped_y_val


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
    assert (range_start and range_end) < len(trace_set[0])
    return trace_set[:, range_start:range_end]


def cut_trace_set__column_range__randomized(
        trace_set, range_start=204, range_end=314, randomize=1
) -> np.array:
    """
    :param trace_set: Trace set to cut.
    :param range_start: Start column position.
    :param range_end: End column position.
    :param randomize:
    :return:
    """
    assert (range_start and range_end) < len(trace_set[0])
    new_trace_set = np.empty(shape=(trace_set.shape[0], 110))
    rng = np.random.default_rng()
    for i, trace in enumerate(trace_set):
        k = int(rng.integers(low=-randomize, high=randomize+1, size=1))
        rand_start = range_start + k
        rand_end = range_end + k
        new_trace_set[i] = trace[rand_start:rand_end]
    return np.array(new_trace_set)


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
    noise_trace_path = os.path.join(
        PROJECT_DIR,
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
    # rand_index = np.random.random_integers(low=0, high=100)
    # example_additive_noise_trace = noise_set[rand_index]
    example_additive_noise_trace = noise_set[100]

    # Apply the noise to trace set
    for i in range(len(trace_set)):
        trace_set[i] += noise_set[i]

    return trace_set, example_additive_noise_trace


def denoising_of_trace_set(
        trace_set: np.array,
        denoising_method_id: int,
        training_dataset_id: int,
) -> Tuple[np.array, int, int, np.array]:
    """
    :param trace_set:
    :param denoising_method_id:
    :param training_dataset_id:
    :return:
    """
    example_not_denoised_trace = trace_set[1].copy()
    if denoising_method_id == 1:
        filtered_set, range_start, range_end = moving_average_filter_n3(
            test_trace_set=trace_set,
            training_dataset_id=training_dataset_id,
        )
        return filtered_set, range_start, range_end, example_not_denoised_trace
    elif denoising_method_id == 2:
        filtered_set, range_start, range_end = moving_average_filter_n5(
            test_trace_set=trace_set,
            training_dataset_id=training_dataset_id,
        )
        return filtered_set, range_start, range_end, example_not_denoised_trace
    elif denoising_method_id == 3:
        range_start = 130
        range_end = 240
        filtered_set, _, __ = wiener_filter_trace_set(trace_set, 2e-2)
        return filtered_set, range_start, range_end, example_not_denoised_trace
    elif denoising_method_id == 4:
        range_start = 130
        range_end = 240
        filtered_set, _, __ = wiener_filter_trace_set(trace_set, 2e-1)
        return filtered_set, range_start, range_end, example_not_denoised_trace
    elif denoising_method_id == 5:
        filtered_set, range_start, range_end = moving_average_filter_n11(
            test_trace_set=trace_set,
            training_dataset_id=training_dataset_id,
        )
        return filtered_set, range_start, range_end, example_not_denoised_trace
    else:
        raise f"Denoising method id {denoising_method_id} is not correct."


def training_deep_learning_model(
        training_model_id: int = 1,
        training_dataset_id: int = 1,
        keybyte: int = 0,
        epochs: int = 100,
        batch_size: int = 256,
        additive_noise_method_id: Optional[int] = None,
        denoising_method_id: Optional[int] = None,
        trace_process_id: int = 3,
        verbose: bool = False,
        mode: int = 1,
        shuffle_trace_and_label_sets: bool = False,
        separate_validation_dataset: bool = False,
        balance_datasets: bool = False,
        grid_search: bool = False,
):
    """
    The main function for training the deep learning classifier.
    Uses training traces from Wang_2021 (5 devices).
    Only CNN with input size 110 and output size 256 is used now.

    :param training_model_id:
    :param training_dataset_id: The id for the dl-model.
    :param keybyte: The keybyte classifier to train.
    :param epochs: Number of epochs to perform.
    :param batch_size: The batch-size used in training.
    :param additive_noise_method_id: Id to additive noise method.
    :param denoising_method_id: Id to denoising method.
    :param trace_process_id: The trace pre-process done.
    :param verbose: Show plot and extra information if true.
    :param mode: If 1: save all epochs. If 2: save only best epochs (min loss).
    :param shuffle_trace_and_label_sets: Shuffle the datasets.
    :param separate_validation_dataset: Use a separate validation dataset (8m).
    :param balance_datasets: Balance the datasets (undersample).
    :param grid_search: If a grid search is to be performed or not. Experimental.
    :return: None.
    """
    # Initialise variables
    validation_set = None
    validation_labels = None
    additive_noise_trace = None
    clean_trace = None

    if training_dataset_id in [2, 3]:
        if trace_process_id == 14:
            start = 130 - 4
            end = 240 - 4
        else:
            start = 130
            end = 240
    else:
        if trace_process_id == 14:
            start = 200
            end = 310
        else:
            start = 204
            end = 314

    # Get training traces path.
    training_set_path = get_training_trace_path(training_dataset_id)
    validation_set_path = get_validation_data_path__8m()

    # Get training traces (based on trace process)
    if trace_process_id == 2:
        trace_set_file_path = os.path.join(
            training_set_path, "traces.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id in [3, 13]:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id in [4, 5]:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
        if separate_validation_dataset:
            validation_set_file_path = os.path.join(
                validation_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
            )
            validation_set = np.load(validation_set_file_path)
    elif trace_process_id == 6:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_6-max_avg(before_sbox).npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 7:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_7-max_avg(sbox).npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id in [8, 11, 12]:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_8-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
        if separate_validation_dataset:
            validation_set_file_path = os.path.join(
                validation_set_path, "trace_process_8-standardization_sbox.npy"
            )
            validation_set = np.load(validation_set_file_path)
    elif trace_process_id == 9:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_9-maxmin_[-1_1]_[0_400].npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 10:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_10-maxmin_[-1_1]_[204_314].npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 14:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_14-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    else:
        return "Trace_process_id is wrong!"

    # Get training labels
    if training_dataset_id == 1:
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
    elif training_dataset_id in [2, 3]:
        labels_path = os.path.join(
            training_set_path,
            "labels.npy"
        )
        labels = np.load(labels_path)
    else:
        return "Something's wrong with the labels."

    # Get validation labels
    if separate_validation_dataset:
        val_labels_path = os.path.join(
            validation_set_path,
            "labels.npy"
        )
        validation_labels = np.load(val_labels_path)

    # Limit the dataset
    # training_trace_set = training_trace_set[:100000]
    # labels = labels[:100000]

    # Balance the training dataset
    if balance_datasets:
        undersample = RandomUnderSampler(
            sampling_strategy="auto",
            # random_state=10
        )
        training_trace_set, labels = undersample.fit_resample(
            training_trace_set, labels
        )

    # Shuffle training data
    if shuffle_trace_and_label_sets:
        training_trace_set, labels = unison_shuffle_traces_and_labels(
            trace_set=training_trace_set,
            labels=labels
        )

    # Get path to store model
    model_save_file_path = get_training_model_file_save_path(
        training_dataset_id=training_dataset_id,
        keybyte=keybyte,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        training_model_id=training_model_id,
        trace_process_id=trace_process_id
    )

    # Get the DL-model
    if training_model_id == 1:
        deep_learning_model = cnn_110_model()
    elif training_model_id == 2:
        deep_learning_model = cnn_110_sgd_model()
    elif training_model_id == 3:
        deep_learning_model = cnn_110_model_simpler()
    elif training_model_id == 4:
        deep_learning_model = cnn_110_model_more()
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
            training_dataset_id=training_dataset_id,
        )

    if trace_process_id == 11:
        training_trace_set -= np.mean(training_trace_set, axis=0)
        # training_trace_set *= 20

    # Cut trace set to the sbox output range
    if trace_process_id in [12, 13, 14]:
        training_trace_set = cut_trace_set__column_range__randomized(
            trace_set=training_trace_set,
            range_start=start,
            range_end=end,
            randomize=1,
        )
    else:
        training_trace_set = cut_trace_set__column_range(
            trace_set=training_trace_set,
            range_start=start,
            range_end=end,
        )

    # Cut clean trace if denoising
    if clean_trace is not None:
        clean_trace = cut_trace_set__column_range(
            trace_set=np.atleast_2d(clean_trace),
            range_start=start,
            range_end=end,
        )

    # Re-normalize the trace set in sbox range
    if trace_process_id == 5:
        training_trace_set = maxmin_scaling_of_trace_set__per_trace_fit(
            trace_set=training_trace_set,
            range_start=0,
            range_end=len(training_trace_set[1])
        )

    # If performing a grid search:
    if grid_search:
        reshaped_x_profiling = training_trace_set.reshape(
            (training_trace_set.shape[0], training_trace_set.shape[1], 1)
        )
        reshaped_y_profiling = to_categorical(labels, num_classes=256)
        if separate_validation_dataset:
            return (
                reshaped_x_profiling,
                reshaped_y_profiling,
                validation_set,
                validation_labels
            )
        else:
            return reshaped_x_profiling, reshaped_y_profiling

    # Plot the traces as a final check
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.plot(
        training_trace_set[0],
        label="Training trace 1"
    )
    ax.plot(
        training_trace_set[1],
        label="Training trace 2"
    )
    ax.plot(
        training_trace_set[2],
        label="Training trace 3"
    )
    if additive_noise_method_id is not None:
        ax.plot(
            additive_noise_trace[start:end],
            label="Additive noise"
        )
    if denoising_method_id is not None:
        ax.plot(
            clean_trace[0],
            color=NORD_LIGHT_ORANGE,
            label="Clean trace."
        )
    trace_fig_save_path_dir = os.path.dirname(model_save_file_path)
    trace_fig_file_path = os.path.join(
        trace_fig_save_path_dir,
        "training_trace_and_processing_attribute.png"
    )
    ax.legend()
    fig.savefig(fname=trace_fig_file_path)
    if verbose:
        plt.show()

    # Train the model
    history_log = train_model(
        x_profiling=training_trace_set,
        y_profiling=labels,
        deep_learning_model=deep_learning_model,
        model_save_path=model_save_file_path,
        epochs=epochs,
        batch_size=batch_size,
        mode=mode,
        x_validation=validation_set,
        y_validation=validation_labels,
    )

    # Store the accuracy and loss data
    model_save_path_dir = os.path.dirname(model_save_file_path)
    history_log_file_path = os.path.join(model_save_path_dir, "history_log.npy")
    np.save(history_log_file_path, history_log.history)

    if verbose:
        plot_history_log(
            training_dataset_id=training_dataset_id,
            trace_process_id=trace_process_id,
            keybyte=keybyte,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
            save=True,
            show=True
        )
    else:
        plot_history_log(
            training_dataset_id=training_dataset_id,
            trace_process_id=trace_process_id,
            keybyte=keybyte,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
            save=True,
            show=False,
        )

    return
