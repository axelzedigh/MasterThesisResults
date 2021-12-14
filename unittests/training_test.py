"""Tests for training util functions."""
import os
import unittest
import sqlite3 as lite

from utils.db_utils import create_db_with_tables, initialize_table_data, \
    get_db_file_path
from utils.trace_utils import get_trace_set__processed, \
    get_training_model_file_save_path
from utils.training_utils import cut_trace_set__column_range, \
    additive_noise__gaussian, additive_noise__rayleigh, \
    additive_noise__collected_noise__office_corridor, denoising_of_trace_set, \
    cut_trace_set__column_range__randomized


class TrainingTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.database = "test_database.db"
        create_db_with_tables(self.database)
        initialize_table_data(self.database)
        self.con = lite.connect(get_db_file_path(self.database))
        self.trace_set = get_trace_set__processed(
            self.database,
            test_dataset_id=1,
            training_dataset_id=None,
            environment_id=1,
            distance=15,
            device=10,
            trace_process_id=2,
        )

    def tearDown(self) -> None:
        self.con.close()
        os.remove(get_db_file_path(self.database))

    def test_cut_trace_set__column_range(self):
        self.assertEqual(self.trace_set.shape[1], 400)
        trace_set_110 = cut_trace_set__column_range(self.trace_set)
        self.assertEqual(trace_set_110.shape[1], 110)

    def test_cut_trace_set__column_range__randomized(self):
        self.assertEqual(self.trace_set.shape[1], 400)
        trace_set = cut_trace_set__column_range__randomized(self.trace_set, randomize=5)
        self.assertEqual(trace_set.shape[1], 110)

    def test_additive_noise__gaussian(self):
        trace_set, noise_trace = additive_noise__gaussian(
            self.trace_set, mean=0, std=0.04
        )
        self.assertEqual(type(trace_set.tolist()), list)
        self.assertEqual(noise_trace.shape, (400,))

    def test_additive_noise__rayleigh(self):
        trace_set, noise_trace = additive_noise__rayleigh(
            self.trace_set, 0.0138
        )
        self.assertEqual(type(trace_set.tolist()), list)
        self.assertEqual(noise_trace.shape, (400,))

    def test_additive_noise__collected_noise__office_corridor(self):
        trace_set, noise_t = additive_noise__collected_noise__office_corridor(
            self.trace_set, scaling_factor=25, mean_adjust=False
        )
        self.assertEqual(type(trace_set.tolist()), list)
        self.assertEqual(noise_t.shape, (400,))

    def test_denoising_of_trace_set(self):
        trace_set, start, end, origin_trace = denoising_of_trace_set(
            self.trace_set, denoising_method_id=1
        )
        self.assertEqual(type(trace_set.tolist()), list)
        self.assertEqual(origin_trace.shape, (400,))
        self.assertEqual(start, 203)
        self.assertEqual(end, 313)

        trace_set, start, end, origin_trace = denoising_of_trace_set(
            self.trace_set, denoising_method_id=2
        )
        self.assertEqual(type(trace_set.tolist()), list)
        self.assertEqual(origin_trace.shape, (400,))
        self.assertEqual(start, 202)
        self.assertEqual(end, 312)

    def test_get_training_model_file_save_path(self):
        model_save_path = get_training_model_file_save_path(
            keybyte=0,
            additive_noise_method_id=1,
            denoising_method_id=None,
            training_model_id=1,
            trace_process_id=3,
        )
        self.assertEqual(
            model_save_path[-62:],
            "models/trace_process_3/keybyte_0/1_None/cnn_110-{epoch:01d}.h5"
        )

