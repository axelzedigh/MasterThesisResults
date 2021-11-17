"""Unittests for statistical functions."""

import unittest

import numpy as np
import sqlite3 as lite
import os

from utils.db_utils import create_db_with_tables, initialize_table_data, get_db_file_path
from utils.statistic_utils import hamming_weight__single, \
    hamming_weight__vector, cross_correlation_matrix, \
    pearson_correlation_coefficient, mycorr, snr_calculator, root_mean_square
from utils.trace_utils import get_trace_set_metadata__depth, get_training_trace_set__processed, \
    get_trace_metadata__depth__processed


class StatisticalFunctionsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.database = "test_database.db"
        create_db_with_tables(self.database)
        initialize_table_data(self.database)
        self.con = lite.connect(get_db_file_path(self.database))

    def tearDown(self) -> None:
        self.con.close()
        os.remove(get_db_file_path(self.database))

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

    # def test_mycorr(self):
    #     x = np.array([1, 4], [1, 3])
    #     y = np.array((4, 6), (1, 4))
    #     corr = mycorr(x, y)
    #     print(corr)
    #     self.assertEqual(corr.tolist(), [1, 0])

    def test_snr_calculator(self):
        x = [1, 2, 3, 4]
        y = [2, 4, 6, 8]
        snr = snr_calculator(x, y)
        print(snr)
        self.assertEqual(snr, 1)

    def test_root_mean_square(self):
        x = np.array([1, 2, 3, 4])
        rms = root_mean_square(x)
        stub = np.sqrt(30/4)
        self.assertEqual(stub, rms)

    def test_get_trace_metadata__depth__processed(self):
        x = np.array([[1, 2, 3, 4], [1, 2, 2, 4]])
        metadata = get_trace_metadata__depth__processed(x)
        self.assertEqual(metadata.tolist()[0][0], 1)

    def test_get_trace_set_metadata__depth(self):
        test_dataset_id = 1
        training_dataset_id = None
        environment_id = 1
        distance = 15
        device = 7
        additive_noise_method_id = None
        trace_process_id = 2
        metadata = get_trace_set_metadata__depth(
            database=self.database,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
            additive_noise_method_id=additive_noise_method_id,
            trace_process_id=trace_process_id,
        )
        print(metadata[:, 0])
        max_val = int(np.max(metadata[:, 0]))
        print(max_val)
        self.assertNotEqual(1, max_val)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.shape, (400, 6))

        test_dataset_id = 1
        training_dataset_id = None
        environment_id = 1
        distance = 15
        device = 7
        additive_noise_method_id = None
        trace_process_id = 3
        metadata = get_trace_set_metadata__depth(
            database=self.database,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
            additive_noise_method_id=additive_noise_method_id,
            trace_process_id=trace_process_id,
        )
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.shape, (400, 6))
