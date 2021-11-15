"""Math- and statistical utils."""
import os

import numpy as np
from scipy import stats
from typing import Tuple, Optional, List
from numpy.matlib import repmat

from utils.db_utils import get_test_trace_path, get_db_file_path


def hamming_weight__single(value) -> int:
    """

    :param value: The value to count the hamming-weight.
    :return: Hamming weight int.
    """
    hamming_weight = bin(value).count("1")
    return hamming_weight


def hamming_weight__vector(vector) -> np.array:
    """

    :param vector: The vector with values to calculate.
    :return: np.array with hamming-weights.
    """
    hamming_weight_function = np.vectorize(hamming_weight__single)
    return hamming_weight_function(vector)


def cross_correlation_matrix(trace_1, trace_2) -> np.array:
    """

    :param trace_1:
    :param trace_2:
    :return: Cross correlation matrix.
    """
    return np.corrcoef(trace_1, trace_2)


def pearson_correlation_coefficient(a, b) -> Tuple:
    """

    :param a: Dataset A.
    :param b: Dataset B.
    :return: Pearson's correlation coefficients, tuple.
    """
    return stats.pearsonr(a, b)


def mycorr(x, y):
    xr, xc = x.shape
    yr, yc = y.shape
    assert xr == yr, "Matrix row count mismatch"

    x = x - x.mean(0)
    y = y - y.mean(0)
    corr = x.T.dot(y)
    xsq = np.atleast_2d(np.sqrt(np.sum(x ** 2, 0)))
    ysq = np.atleast_2d(np.sqrt(np.sum(y ** 2, 0)))
    corr = np.divide(corr, repmat(xsq.T, 1, yc))
    corr = np.divide(corr, repmat(ysq, xc, 1))
    return corr


def snr_calculator(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    mean_tmp = []
    var_tmp = []
    for i in np.unique(y):
        index = np.where(y == i)[0]
        mean_tmp.append(np.mean(x[index, :], axis=0))
        var_tmp.append(np.var(x[index, :], axis=0))
    snr = np.var(mean_tmp, axis=0) / np.mean(var_tmp, axis=0)
    return snr


def root_mean_square(vector):
    """

    :param vector: vector with values to calculate.
    :return: RMS.
    """
    vector_squared = np.array(vector) ** 2
    vector_squared_sum = np.sum(vector_squared)
    rms = np.sqrt(vector_squared_sum / vector_squared.size)
    return rms


def get_trace_set_metadata__depth(
        database: str,
        test_dataset_id: Optional[int],
        training_dataset_id: Optional[int],
        environment_id: int,
        distance: int,
        device: int,
        additive_noise_method_id: Optional[int],
        trace_process_id: int,
) -> np.array:
    """

    :param trace_process_id:
    :param database:
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param additive_noise_method_id:
    :return:
    """
    if type(test_dataset_id) == type(training_dataset_id):
        print("Dataset must be either test dataset or training dataset")
        return

    # Get path load raw data if trace_process_id is in {1, 2}
    if trace_process_id == 1:
        pass
    else:
        trace = get_trace_set__processed(
            database,
            test_dataset_id,
            environment_id,
            distance,
            device,
            trace_process_id,
        )
        meta_data = get_trace_metadata__depth__processed(trace)
        return meta_data


def get_trace_set__processed(
        database,
        test_dataset_id,
        environment_id,
        distance,
        device,
        trace_process_id,
) -> np.array:
    """

    :param database:
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param trace_process_id:
    :return:
    """
    trace_path = get_test_trace_path(
        database,
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device
    )

    if trace_process_id == 2:
        file_path = os.path.join(trace_path, "traces.npy")
        traces = np.load(file_path)
    elif trace_process_id == 3:
        file_path = os.path.join(trace_path, "nor_traces_maxmin.npy")
        traces = np.load(file_path)
    else:
        print("Something went wrong.")
        return 1

    return traces


def get_trace_metadata__depth__processed(trace_set):
    """

    :param trace_set:
    :return:
    """
    meta_data = []
    for index in range(trace_set.shape[1]):
        max_value = np.max(trace_set[:, index], axis=0)
        min_value = np.min(trace_set[:, index], axis=0)
        mean_value = np.mean(trace_set[:, index], axis=0)
        rms_value = root_mean_square(trace_set[:, index])
        std_value = np.std(trace_set[:, index], axis=0)
        snr_value = (mean_value ** 2) / (std_value ** 2)
        meta_data.append([max_value, min_value, mean_value, rms_value, std_value, snr_value])

    return np.array(meta_data)
