"""Math- and statistical utils."""

import numpy as np
from scipy import stats
from typing import Tuple, List
from numpy.matlib import repmat
from sklearn import preprocessing


def hamming_weight__single(value: int) -> int:
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


def correlation_matrix(x, y):
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


def snr_calculator(trace_set, label_set):
    """

    :param trace_set:
    :param label_set:
    :return:
    """
    mean_tmp = []
    var_tmp = []
    for i in np.unique(label_set):
        index = np.where(label_set == i)[0]
        mean_tmp.append(np.mean(trace_set[index, :], axis=0))
        var_tmp.append(np.var(trace_set[index, :], axis=0))
    snr = np.var(mean_tmp, axis=0) / np.mean(var_tmp, axis=0)
    return snr


def signal_to_noise_ratio__sqrt_mean_std(mean, std):
    """

    :param mean: µ
    :param std: ∂
    :return:
    """
    return (mean ** 2) / (std ** 2)


def root_mean_square(vector):
    """

    :param vector: vector with values to calculate.
    :return: RMS.
    """
    vector_squared = np.array(vector) ** 2
    vector_squared_sum = np.sum(vector_squared)
    rms = np.sqrt(vector_squared_sum / vector_squared.size)
    return rms


def signal_to_noise_ratio__amplitude(
        rms_signal: float, rms_noise: float
) -> float:
    """

    :param rms_signal: The RMS amplitude of the signal.
    :param rms_noise: The RMS amplitude of the noise.
    :return: The SNR metric value.
    """
    return (rms_signal ** 2) / (rms_noise ** 2)


def sklearn_normalizing__max(trace):
    """

    :param trace:
    :return:
    """
    normalized_trace = preprocessing.normalize(trace, norm='max')
    return normalized_trace


def maxmin_scaling_of_trace_set__whole_trace_set_fit(trace_set):
    """
    Scaler that performs normalization based on max and min in WHOLE trace-set.
    :param trace_set:
    :return:
    """
    maxmin_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    maxmin_scaler.fit_transform(trace_set)
    return trace_set


def maxmin_scaling_of_trace_set__per_trace_fit(
        trace_set: np.array,
        range_start: int = 204,
        range_end: int = 314,
        scaling_range: Tuple[int, int] = (0, 1),
) -> np.array:
    """
    MaxMin-scaler that performs normalization based on max and min per trace in
    the set.

    :param trace_set:
    :param range_start:
    :param range_end:
    :param scaling_range: Tuple
    :return:
    """
    scaling_scalar = np.abs(scaling_range[0] - scaling_range[1])
    scaling_transl = scaling_range[0]
    scaled_trace_set = np.empty_like(trace_set)
    for i, trace in enumerate(trace_set):
        max_value = np.max(trace[range_start:range_end])
        min_value = np.min(trace[range_start:range_end])
        unit_scaled_trace = (trace - min_value) / (max_value - min_value)
        scaled_trace = (unit_scaled_trace * scaling_scalar) + scaling_transl
        scaled_trace_set[i] = scaled_trace

    return scaled_trace_set


def maxmin_scaling_of_trace_set__per_trace_fit__max_avg(
        trace_set: np.array,
        range_start: int,
        range_end: int,
        avg_start: int = 0,
        avg_end: int = 100,
        scale: float = 2.2
) -> np.array:
    """
    Scaler that performs normalization. Max value is derived from
    avg(avg_start:avg_end). Min is derived from min(range_start:range_end).
    Used in trace_process_id 6 & 7.

    :param trace_set:
    :param range_start:
    :param range_end:
    :param avg_start:
    :param avg_end:
    :param scale:
    :return:
    """
    scaled_trace_set = np.empty_like(trace_set)
    for i, trace in enumerate(trace_set):
        max_value = np.mean(trace[avg_start:avg_end]) * scale
        min_value = np.min(trace[range_start:range_end])
        scaled_trace = (trace - min_value) / (max_value - min_value)
        scaled_trace_set[i] = scaled_trace

    return scaled_trace_set


def standardization_of_trace_set__per_trace_fit(
        trace_set: np.array,
        range_start: int = 204,
        range_end: int = 314,
) -> np.array:
    """
    Standardization of trace_set (centered on mean µ, scaled to 1 std ∂).

    :param trace_set:
    :param range_start:
    :param range_end:
    :return:
    """
    standardized_trace_set = np.empty_like(trace_set)
    for i, trace in enumerate(trace_set):
        standardized_trace = (trace - np.mean(trace[range_start:range_end])) / np.std(trace[range_start:range_end])
        standardized_trace_set[i] = standardized_trace

    return standardized_trace_set
