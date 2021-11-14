"""Unittests for statistical functions."""

import unittest

import numpy as np

from utils.statistic_utils import hamming_weight__single, hamming_weight__vector, cross_correlation__traces


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
        trace_1 = [1, 2, 3, 4]
        trace_2 = [2, 4, 6, 8]
        ccr = cross_correlation__traces(trace_1, trace_2)
        self.assertEqual(ccr.tolist(), [[1, 1], [1, 1]])

        trace_1 = [-1, -2, -3, -4]
        trace_2 = [2, 4, 6, 8]
        ccr = cross_correlation__traces(trace_1, trace_2)
        self.assertEqual(ccr.tolist(), [[1, -1], [-1, 1]])

