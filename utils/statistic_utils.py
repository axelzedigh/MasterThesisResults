"""Math- and statistical utils."""
import numpy as np
from scipy import stats
from typing import Tuple
from numpy.matlib import repmat


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
