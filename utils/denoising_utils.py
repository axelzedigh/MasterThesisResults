"""
Denoising functions.
"""
import sys
from typing import Tuple

import numpy as np
import scipy.signal as signal


def moving_average_filter(test_trace, n):
    """

    :param test_trace: The trace to filter.
    :param n: Number of data-points to use when averaging.
    :return: A filtered trace.
    """
    cumsum = np.cumsum(np.insert(test_trace, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def moving_average_filter_n3(
        test_trace_set: np.array,
        training_dataset_id: int = 1,
) -> Tuple[np.array, int, int]:
    """

    :param test_trace_set: The trace set to filter.
    :param training_dataset_id:
    :return: Filtered trace set and ranges.
    """
    filtered_trace_set = np.empty_like(test_trace_set)
    n = 3
    if training_dataset_id == 1:
        range_start = 203
        range_end = 313
    elif training_dataset_id in [2, 3]:
        range_start = 129
        range_end = 239
    else:
        print("Wrong training dataset id.")
        sys.exit(-1)

    for i in range(len(test_trace_set)):
        filtered_trace_set[i] = np.pad(
            moving_average_filter(test_trace_set[i], n),
            (0, 2),
            'constant'
        )

    return filtered_trace_set, range_start, range_end


def moving_average_filter_n5(
        test_trace_set: np.array,
        training_dataset_id: int = 1,
) -> Tuple[np.array, int, int]:
    """

    :param test_trace_set: The trace set to filter.
    :param training_dataset_id:
    :return: Filtered trace set and ranges.
    """
    filtered_trace_set = np.empty_like(test_trace_set)
    n = 5
    if training_dataset_id == 1:
        range_start = 202
        range_end = 312
    elif training_dataset_id in [2, 3]:
        range_start = 128
        range_end = 238

    for i in range(len(test_trace_set)):
        filtered_trace_set[i] = np.pad(
            moving_average_filter(test_trace_set[i], n),
            (0, 4),
            'constant'
        )

    return filtered_trace_set, range_start, range_end


def wiener_filter(trace, noise_power=0.0001):
    """

    :param trace:
    :param noise_power:
    :return:
    """
    shape = None
    filtered_trace = signal.wiener(trace, noise=noise_power)
    return filtered_trace


def wiener_filter_trace_set(
        test_trace_set: np.array,
        noise_power: float,
) -> Tuple[np.array, int, int]:
    """

    :param noise_power:
    :param test_trace_set:
    :return:
    """
    range_start = 204
    range_end = 314
    filtered_trace_set = np.empty_like(test_trace_set)

    for i in range(len(test_trace_set)):
        filtered_trace_set[i] = wiener_filter(test_trace_set[i], noise_power)

    return filtered_trace_set, range_start, range_end
