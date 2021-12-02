"""
Denoising functions.
"""
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
        test_trace_set
) -> Tuple[np.array, int, int]:
    """

    :param test_trace_set: The trace set to filter.
    :return: Filtered trace set and ranges.
    """
    filtered_trace_set = np.empty_like(test_trace_set)
    n = 3
    range_start = 203
    range_end = 313

    for i in range(len(test_trace_set)):
        filtered_trace_set[i] = np.pad(
            moving_average_filter(test_trace_set[i], n),
            (0, 2),
            'constant'
        )

    return filtered_trace_set, range_start, range_end


def moving_average_filter_n5(
        test_trace_set
) -> Tuple[np.array, int, int]:
    """

    :param test_trace_set: The trace set to filter.
    :return: Filtered trace set and ranges.
    """
    filtered_trace_set = np.empty_like(test_trace_set)
    n = 5
    range_start = 202
    range_end = 312

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
