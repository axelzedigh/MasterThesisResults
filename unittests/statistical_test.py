"""Unittests for statistical functions."""

import unittest

import numpy as np

from utils.statistic_utils import hamming_weight__single, hamming_weight__vector, cross_correlation_matrix, \
    pearson_correlation_coefficient


class StatisticalFuncitonsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_hamming_weight_func(self):
        val = 2
        hw = hamming_weight__single(val)
        self.assertEqual(1, hw)

        val = 9
        hw = hamming_weight__single(val)
        self.assertEqual(2, hw)

        values = [2, 9]
        hw = hamming_weight__vector(values)
        arr = np.array([1, 2])
        self.assertEqual(arr.tolist(), hw.tolist())

    def test_cross_correlation__traces(self):
        a = [1, 2, 3, 4]
        b = [2, 4, 6, 8]
        ccr = cross_correlation_matrix(a, b)
        self.assertEqual(ccr.tolist(), [[1, 1], [1, 1]])

        a = [-1, -2, -3, -4]
        b = [2, 4, 6, 8]
        ccr = cross_correlation_matrix(a, b)
        self.assertEqual(ccr.tolist(), [[1, -1], [-1, 1]])

    def test_pearson_correlation_coefficient(self):
        a = [1, 2, 3, 4]
        b = [2, 4, 6, 8]
        pc = pearson_correlation_coefficient(a, b)
        self.assertEqual(pc, (1, 0))

        a = [-1, -2, -3, -4]
        b = [2, 4, 6, 8]
        pc = pearson_correlation_coefficient(a, b)
        self.assertEqual(pc, (-1, 0))
