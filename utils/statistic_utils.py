"""Math- and statistical utils."""
import numpy as np
from scipy import stats
from typing import Tuple

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
    :return: Pearsons correlation coefficients, tuple.
    """
    return stats.pearsonr(a, b)
