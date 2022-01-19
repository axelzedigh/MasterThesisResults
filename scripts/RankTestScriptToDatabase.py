"""Functions for testing the model."""
import os.path
import random
import sys
from typing import Optional, Callable, Tuple
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from utils.trace_utils import get_normalized_test_traces, \
    get_training_trace_path
from utils.db_utils import get_test_trace_path, get_training_model_file_path, \
    get_db_file_path


def load_sca_model(model_file):
    """
    :param model_file: Path to the model.
    :return: The Keras model.
    """
    try:
        model = tf.keras.models.load_model(model_file)
    except OSError:
        raise f"Error: can't load Keras model file {model_file}."
    return model


def get_prediction(model, traces):
    """

    :param model:
    :param traces:
    :return:
    """
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    if input_layer_shape[1] != len(traces[0]):
        print(
            "Error: model input shape %d instead of %d is not expected ..."
            % (input_layer_shape[1], len(traces[0]))
        )
        sys.exit(-1)
    # Adapt the data shape according our model input
    elif len(input_layer_shape) == 3:
        # This is a CNN: reshape the data
        input_data = traces
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        print(
            "Error: model input shape length %d is not expected ..."
            % len(input_layer_shape)
        )
        sys.exit(-1)

    # Predict our probabilities
    predictions = model.predict(input_data)

    return predictions


def prediction_to_probability(
        selected_cts_interest: np.array,
        selected_predictions: np.array,
        number: int,
) -> np.array:
    """

    :param selected_cts_interest: Ciphertexts.
    :param selected_predictions: Predicted
    :param number: Usually 1500 (maximum traces needed to get key).
    :return:
    """
    probabilities_array = []

    for i in range(number):
        probabilities = np.zeros(256)
        for j in range(256):
            # value = AES_Sbox[selected_Pts_interest[i] ^ j]

            last_round_sbox_out = selected_cts_interest[i] ^ j
            # var = Inv_SBox[last_round_sbox_out]

            value = last_round_sbox_out
            # value = SBox_in^SBox_out

            # value = selected_Pts_interest[i] ^ j
            # value = Inv_Sbox[selected_Pts_interest[i] ^ j]

            probabilities[j] = selected_predictions[i][value]

        # print(probabilities)
        probabilities_array.append(probabilities)
        # print(probabilities_array)

    probabilities_array = np.array(probabilities_array)

    for i in range(len(probabilities_array)):
        if np.count_nonzero(probabilities_array[i]) != 256:
            none_zero_predictions = [a for a in probabilities_array[i] if a != 0]
            min_v = min(none_zero_predictions)
            probabilities_array[i] = probabilities_array[i] + min_v ** 2

    return probabilities_array


def rank_cal(selected_probabilities, key_interest, number):
    """

    :param selected_probabilities:
    :param key_interest:
    :param number:
    :return:
    """
    rank = []
    total_pro = np.zeros(256)

    for i in range(number):
        # epsilon = 4*10**-12
        # selected_probabilities[i] = selected_probabilities[i] +epsilon
        total_pro = total_pro + np.log(selected_probabilities[i])

        # Find the rank of real key in the total probabilities
        sorted_probability = np.array(
            list(map(lambda a: total_pro[a], total_pro.argsort()[::-1]))
        )
        real_key_rank = np.where(sorted_probability == total_pro[key_interest])[0][0]
        rank.append(real_key_rank)

    rank = np.array(rank)

    return rank


def termination_point_test(
        database: str,
        filter_function: Optional[Callable],
        test_dataset_id: int,
        environment_id: int,
        distance: float,
        device: int,
        training_model_id: int,
        keybyte: int,
        epoch: int,
        additive_noise_method_id: int,
        denoising_method_id: int,
        trace_process_id: int,
        training_dataset_id: int,
) -> Optional[int]:
    """

    :param database:
    :param filter_function:
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param training_model_id:
    :param keybyte:
    :param epoch:
    :param additive_noise_method_id:
    :param denoising_method_id:
    :param trace_process_id:
    :param training_dataset_id:
    :return:
    """

    # Range in traces to test.
    range_start = 204
    range_end = 314

    database = get_db_file_path(database)

    # Test trace set path
    test_path = get_test_trace_path(
        database=database,
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device
    )

    number_total_trace = 4900
    # testing_traces_path = os.path.join(test_path, trace_set_file_name)
    # testing_traces = np.load(testing_traces_path)
    testing_traces = get_normalized_test_traces(
        trace_process_id=trace_process_id,
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
        save=False
    )
    testing_traces = testing_traces[:number_total_trace]

    tenth_roundkey = "10th_roundkey.npy"
    ct = "ct.npy"
    keys_path = os.path.join(test_path, tenth_roundkey)
    ciphertexts_path = os.path.join(test_path, ct)

    # Filter traces
    if filter_function is not None:
        if denoising_method_id == 3:
            testing_traces, _, __ = filter_function(testing_traces, 2e-2)
        else:
            testing_traces, range_start, range_end = filter_function(testing_traces)

    # Select range in traces to test.
    testing_traces = testing_traces[:, [i for i in range(range_start, range_end)]]

    # load key
    key = np.load(keys_path)

    # load plaintext (all bytes)
    cts = np.load(ciphertexts_path)
    cts = cts[:number_total_trace]

    # choose interest keybyte and pt byte
    key_interest = key[keybyte]
    cts_interest = cts[:, keybyte]

    if trace_process_id == 11:
        training_set_path = get_training_trace_path(training_dataset_id)
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_8-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
        training_trace_set = training_trace_set[:, [i for i in range(130, 240)]]
        cts = cts[testing_traces[:, 0] > 2.1]
        testing_traces = testing_traces[testing_traces[:, 0] > 2.1]
        testing_traces -= np.mean(training_trace_set, axis=0)

    # Load training model
    training_model_path = get_training_model_file_path(
        database=database,
        training_model_id=training_model_id,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        epoch=epoch,
        keybyte=keybyte,
        trace_process_id=trace_process_id,
        training_dataset_id=training_dataset_id,
    )
    training_model = load_sca_model(training_model_path)

    # get predictions for all traces
    predictions = get_prediction(training_model, testing_traces)

    # randomly select trace for testing
    number = 1500
    average = 50   # 50
    ranks_array = []

    for i in range(average):
        select = random.sample(range(len(testing_traces)), number)
        selected_cts_interest = cts_interest[select]
        selected_predictions = predictions[select]

        # Calculate subkey probability for selected traces
        probabilities = prediction_to_probability(
            selected_cts_interest, selected_predictions, number
        )
        ranks = rank_cal(probabilities, key_interest, number)
        ranks_array.append(ranks)

    ranks_array = np.array(ranks_array)

    term_point = None
    for i in range(ranks_array.shape[1]):
        if np.count_nonzero(ranks_array[:, i]) < int(average / 2):
            term_point = i
            break

    # average_ranks = np.sum(ranks_array, axis=0) / average
    # plt.plot(average_ranks)
    # plt.show()
    return term_point


def termination_point_test_setup(
        database: str,
        filter_function: Optional[Callable],
        test_dataset_id: int,
        environment_id: int,
        distance: float,
        device: int,
        training_model_id: int,
        keybyte: int,
        epoch: int,
        additive_noise_method_id: int,
        denoising_method_id: int,
        trace_process_id: int,
        training_dataset_id: int,
        plot: bool = False,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    :param database:
    :param filter_function:
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param training_model_id:
    :param keybyte:
    :param epoch:
    :param additive_noise_method_id:
    :param denoising_method_id:
    :param trace_process_id:
    :param training_dataset_id:
    :param plot:
    :return:
    """

    # Range in traces to test.
    if trace_process_id == 14:
        range_start = 200
        range_end = 310
    else:
        range_start = 204
        range_end = 314

    database = get_db_file_path(database)

    # Test trace set path
    test_path = get_test_trace_path(
        database=database,
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device
    )

    number_total_trace = 4900
    # testing_traces_path = os.path.join(test_path, trace_set_file_name)
    # testing_traces = np.load(testing_traces_path)
    testing_traces = get_normalized_test_traces(
        trace_process_id=trace_process_id,
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
        save=False
    )
    testing_traces = testing_traces[:number_total_trace]

    tenth_roundkey = "10th_roundkey.npy"
    ct = "ct.npy"
    keys_path = os.path.join(test_path, tenth_roundkey)
    ciphertexts_path = os.path.join(test_path, ct)

    # load key
    key = np.load(keys_path)

    # load plaintext (all bytes)
    cts = np.load(ciphertexts_path)
    cts = cts[:number_total_trace]

    # choose interest keybyte and pt byte
    key_interest = key[keybyte]
    cts_interest = cts[:, keybyte]

    if trace_process_id == 11:
        training_set_path = get_training_trace_path(training_dataset_id)
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_8-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
        training_trace_set = training_trace_set[:, [i for i in range(130, 240)]]
        # cts = cts[testing_traces[:, 0] > 2.1]
        # testing_traces = testing_traces[testing_traces[:, 0] > 2.1]

    # Filter traces
    if filter_function is not None:
        if denoising_method_id == 3:
            testing_traces, _, __ = filter_function(testing_traces, 2e-2)
        else:
            testing_traces, range_start, range_end = filter_function(
                testing_traces
            )

    # Select range in traces to test.
    testing_traces = testing_traces[:, [i for i in range(
        range_start,
        range_end
    )]]

    if trace_process_id == 11:
        if denoising_method_id in range(10):
            training_trace_set, range_start, range_end = filter_function(
                training_trace_set
            )
        # testing_traces -= np.mean(testing_traces, axis=0)
        testing_traces -= np.mean(training_trace_set, axis=0)
        testing_traces *= 40

    if plot:
        # Plot traces
        plt.plot(testing_traces[0])
        plt.plot(testing_traces[1])
        plt.plot(testing_traces[2])
        plt.show()

    # Load training model
    training_model_path = get_training_model_file_path(
        database=database,
        training_model_id=training_model_id,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        epoch=epoch,
        keybyte=keybyte,
        trace_process_id=trace_process_id,
        training_dataset_id=training_dataset_id,
    )
    training_model = load_sca_model(training_model_path)

    # get predictions for all traces
    predictions = get_prediction(training_model, testing_traces)

    return testing_traces, predictions, key_interest, cts_interest


def termination_point_test__rank_test(
        testing_traces: np.array,
        predictions: np.array,
        key_interest: np.array,
        cts_interest: np.array,
        plot: bool = False,
) -> Optional[int]:
    """

    :param testing_traces:
    :param predictions:
    :param key_interest:
    :param cts_interest:
    :param plot:
    :return:
    """

    # randomly select trace for testing
    number = 1500
    average = 50   # 50
    ranks_array = []

    for i in range(average):
        select = random.sample(range(len(testing_traces)), number)
        selected_cts_interest = cts_interest[select]
        selected_predictions = predictions[select]

        # Calculate subkey probability for selected traces
        probabilities = prediction_to_probability(
            selected_cts_interest, selected_predictions, number
        )
        ranks = rank_cal(probabilities, key_interest, number)
        ranks_array.append(ranks)

    ranks_array = np.array(ranks_array)

    term_point = None
    for i in range(ranks_array.shape[1]):
        if np.count_nonzero(ranks_array[:, i]) < int(average / 2):
            term_point = i
            break

    if plot:
        average_ranks = np.sum(ranks_array, axis=0) / average
        plt.plot(average_ranks)
        plt.show()

    return term_point


def termination_point_test__rank_test__2000(
        testing_traces: np.array,
        predictions: np.array,
        key_interest: np.array,
        cts_interest: np.array,
        plot: bool = False,
) -> Optional[int]:
    """

    :param testing_traces:
    :param predictions:
    :param key_interest:
    :param cts_interest:
    :param plot:
    :return:
    """

    # randomly select trace for testing
    number = 3000
    average = 50   # 50
    ranks_array = []

    for i in range(average):
        select = random.sample(range(len(testing_traces)), number)
        selected_cts_interest = cts_interest[select]
        selected_predictions = predictions[select]

        # Calculate subkey probability for selected traces
        probabilities = prediction_to_probability(
            selected_cts_interest, selected_predictions, number
        )
        ranks = rank_cal(probabilities, key_interest, number)
        ranks_array.append(ranks)

    ranks_array = np.array(ranks_array)

    term_point = None
    for i in range(ranks_array.shape[1]):
        if np.count_nonzero(ranks_array[:, i]) < int(average / 2):
            term_point = i
            break

    if plot:
        average_ranks = np.sum(ranks_array, axis=0) / average
        plt.plot(average_ranks)
        plt.show()

    return term_point
