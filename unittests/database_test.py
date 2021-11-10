""" Unit tests concerning retrieval and insertion of data to database."""
import unittest
import sqlite3 as lite
import os

import numpy as np

from database.queries import QUERY_FULL_RANK_TEST_GROUPED_A, \
    QUERY_RANK_TEST_GROUPED_A
from database.variables import VIEW_RANK_TEST_INDEX
from scripts.test_models__termination_point_to_db import termination_point_test_and_insert_to_db
from utils.db_utils import (
    create_db_with_tables,
    initialize_table_data,
    insert_data_to_db,
    fetchall_query,
    get_additive_noise_method_id,
    get_denoising_method_id, insert_legacy_rank_test_numpy_file_to_db,
    create_md__option_tables, get_db_absolute_path,
    get_test_trace_path, get_training_model_file_path, get_db_file_path,
)


class AddToDatabaseTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.database = "test_database.db"
        create_db_with_tables(self.database)
        initialize_table_data(self.database)
        self.con = lite.connect(get_db_file_path(self.database))
        self.cur = self.con.cursor()

    def tearDown(self) -> None:
        self.con.close()
        os.remove(get_db_file_path(self.database))

    def test_fetch_environments(self):
        fetchall = self.cur.execute("SELECT * FROM Environments;").fetchall()
        environments = [(1, "office_corridor"), (2, "big_hall")]
        self.assertEqual(fetchall[0], environments[0])
        self.assertEqual(fetchall[1], environments[1])

    def test_fetch_test_datasets(self):
        fetchall = self.cur.execute("SELECT * FROM Test_Datasets;").fetchall()
        test_datasets = [(1, "Wang_2021"), (2, "Zedigh_2021")]
        self.assertIsNotNone(fetchall)
        self.assertEqual(fetchall[0], test_datasets[0])
        self.assertEqual(fetchall[1], test_datasets[1])

    def test_fetch_training_datasets(self):
        fetchall = self.cur.execute(
            "SELECT * FROM Training_Datasets;").fetchall()
        self.assertIsNotNone(fetchall)
        training_datasets = [(1, "Wang_2021-Cable")]
        self.assertEqual(fetchall[0], training_datasets[0])

    def test_fetch_training_models(self):
        fetchall = self.cur.execute("SELECT * FROM Training_Models;").fetchall()
        self.assertIsNotNone(fetchall)
        training_models = [(1, "cnn_110")]
        self.assertEqual(fetchall[0], training_models[0])

    def test_fetch_additive_noise_methods(self):
        fetchall = self.cur.execute(
            "SELECT * FROM Additive_Noise_Methods;").fetchall()
        self.assertIsNotNone(fetchall)
        additive_noise_methods = [
            (1, "Gaussian", "Std", 0.01, "Mean", 0.0),
            (2, "Gaussian", "Std", 0.02, "Mean", 0.0),
            (3, "Gaussian", "Std", 0.03, "Mean", 0.0),
            (4, "Gaussian", "Std", 0.04, "Mean", 0.0),
            (5, "Gaussian", "Std", 0.05, "Mean", 0.0),
            (6, "Collected", "Scale", 25.0, None, None),
            (7, "Collected", "Scale", 50.0, None, None),
            (8, "Collected", "Scale", 75.0, None, None),
            (9, "Collected", "Scale", 105.0, None, None),
            (10, "Rayleigh", "Mode", 0.0138, None, None),
            (11, "Rayleigh", "Mode", 0.0276, None, None),
        ]
        self.assertEqual(fetchall[10], additive_noise_methods[10])

    def test_fetch_denoising_methods(self):
        fetchall = self.cur.execute(
            "SELECT * FROM Denoising_Methods;").fetchall()
        self.assertIsNotNone(fetchall)
        denoising_methods = [(1, "Moving Average Filter", "N", 3.0, None, None)]
        self.assertEqual(fetchall[0], denoising_methods[0])

    def test_insert_denoising_method(self):
        self.cur.execute(
            "INSERT INTO Denoising_Methods VALUES(NULL,'CDAE',NULL,NULL, "
            "NULL, NULL) "
        )
        fetchall = self.cur.execute(
            "SELECT * FROM Denoising_Methods;").fetchall()
        self.assertEqual("CDAE", fetchall[-1][1])

    def test_insert_rank_test_data(self):
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=2, environment_id=1, distance=15,
                          device=8, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None,
                          denoising_method_id=None, termination_point=9999,
                          average_rank=9999)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=7, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None,
                          denoising_method_id=None, termination_point=101,
                          average_rank=102)
        fetchall = self.cur.execute("SELECT * FROM Rank_Test;").fetchall()
        self.assertIsNotNone(fetchall)
        device = fetchall[0][5]
        self.assertEqual(8, device)

    def test_view_query(self):
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=7, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None,
                          denoising_method_id=None, termination_point=101,
                          average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=8, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=1, denoising_method_id=None,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=9, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None, denoising_method_id=1,
                          termination_point=101, average_rank=102)
        query = "SELECT * FROM full_rank_test;"
        data = fetchall_query(self.database, query)
        self.assertNotEqual(data, [])
        self.assertEqual(3, len(data))
        self.assertEqual(data[0][VIEW_RANK_TEST_INDEX["device"]], 7)
        self.assertEqual(data[1][VIEW_RANK_TEST_INDEX["device"]], 8)
        self.assertEqual(data[2][VIEW_RANK_TEST_INDEX["device"]], 9)
        self.assertIsNone(
            data[0][VIEW_RANK_TEST_INDEX["additive_noise_method"]])
        self.assertEqual(
            data[1][VIEW_RANK_TEST_INDEX["additive_noise_method"]], "Gaussian"
        )
        self.assertEqual(
            data[2][VIEW_RANK_TEST_INDEX["denoising_method"]],
            "Moving Average Filter"
        )
        self.assertEqual(data[2][VIEW_RANK_TEST_INDEX["denoising_param_1"]],
                         "N")

    def test_get_additive_noise_method_id(self):
        method = "Gaussian"
        param_1 = "Std"
        param_1_value = 0.03
        param_2 = "Mean"
        param_2_value = 0
        additive_id = get_additive_noise_method_id(
            self.database, method, param_1, param_1_value, param_2,
            param_2_value
        )
        self.assertIsNotNone(additive_id)
        self.assertEqual(additive_id, 3)

        method = "Rayleigh"
        param_1 = "Mode"
        param_1_value = 0.0138
        param_2 = None
        param_2_value = None
        additive_id = get_additive_noise_method_id(
            self.database, method, param_1, param_1_value, param_2,
            param_2_value
        )
        self.assertIsNotNone(additive_id)
        self.assertEqual(additive_id, 10)

    def test_get_denoising_method_id(self):
        method = "Moving Average Filter"
        param_1 = "N"
        param_1_value = 3
        param_2 = None
        param_2_value = None
        denoising_id = get_denoising_method_id(
            self.database, method, param_1, param_1_value, param_2,
            param_2_value
        )
        self.assertIsNotNone(denoising_id)
        self.assertEqual(denoising_id, 1)

        method = "Moving Average Filter"
        param_1 = "N"
        param_1_value = 5
        param_2 = None
        param_2_value = None
        denoising_id = get_denoising_method_id(
            self.database, method, param_1, param_1_value, param_2,
            param_2_value
        )
        self.assertIsNotNone(denoising_id)
        self.assertEqual(denoising_id, 2)

    def test_insert_rank_test_data_to_db_from_numpy(self):
        # Setup variables
        test_dataset_id = 1
        training_dataset_id = 1
        environment_id = 1
        distance = 15
        training_model_id = 1
        additive_noise_method_id = 7  # Collected scale 50
        denoising_method_id = None

        device = 4
        epoch = 33
        keybyte = 8

        # Setup numpy file
        termination_points = [50, 60, 70]
        termination_point_numpy_list = np.array(termination_points)

        # Save list as file
        filename = (
            f"rank_test-device-{device}-epoch-{epoch}-keybyte-{keybyte}-runs"
            f"-100-cnn_110-some_noise_processing "
        )
        np.save(filename, termination_point_numpy_list)

        # reference to the temp numpy file
        real_file_path = os.path.join("../unittests/", filename + ".npy")

        # Insert numpy-file data to db
        insert_legacy_rank_test_numpy_file_to_db(
            self.database,
            real_file_path,
            test_dataset_id,
            training_dataset_id,
            environment_id,
            distance,
            training_model_id,
            additive_noise_method_id,
            denoising_method_id)

        # Get rank test data from db
        data = fetchall_query(self.database)
        self.assertIsNotNone(data)
        self.assertNotEqual(data, [])
        self.assertEqual(data[0][VIEW_RANK_TEST_INDEX["device"]], 4)
        self.assertEqual(data[1][VIEW_RANK_TEST_INDEX["epoch"]], 33)
        self.assertEqual(data[2][VIEW_RANK_TEST_INDEX["keybyte"]], 8)
        self.assertEqual(data[0][VIEW_RANK_TEST_INDEX["termination_point"]], 50)
        self.assertEqual(data[1][VIEW_RANK_TEST_INDEX["termination_point"]], 60)

        # Remove numpy-file
        os.remove(filename + ".npy")

    def test_create_pre_processing_table_info_file(self):
        # Get file path
        project_dir = os.getenv("MASTER_THESIS_RESULTS")
        path = "unittests"
        file_path = os.path.join(project_dir, path, "pre_processing_tables.md")

        # Update file
        create_md__option_tables(self.database, path)

        file = open(file_path, "r")
        first_line = file.readline()
        file.close()

        first_line_stub = "# Pre-processing tables\n"
        self.assertIsNotNone(file)
        self.assertEqual(first_line, first_line_stub)

        # Delete file
        os.remove(file_path)

    def test_get_number_of_rows_in_rank_test_table(self):
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=9, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None, denoising_method_id=1,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=8, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=1, denoising_method_id=None,
                          termination_point=101, average_rank=102)
        data = fetchall_query(self.database, "SELECT Count(*) FROM Rank_Test;")
        self.assertNotEqual(data, [])
        self.assertEqual(data[0][0], 2)

    def test_get_db_absolute_path(self):
        db_path = get_db_absolute_path(self.database, "unittest")
        self.assertIsNotNone(db_path)

    def test_get_rank_test_grouped_a_query(self):
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=9, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None, denoising_method_id=1,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=9, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None, denoising_method_id=1,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=8, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=1, denoising_method_id=None,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=8, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=1, denoising_method_id=None,
                          termination_point=101, average_rank=102)
        data = fetchall_query(self.database, QUERY_RANK_TEST_GROUPED_A)
        self.assertIsNotNone(data)
        self.assertNotEqual(data, [])

    def test_get_full_rank_test_grouped_a_query(self):
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=9, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None, denoising_method_id=1,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=9, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=None, denoising_method_id=1,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=8, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=1, denoising_method_id=None,
                          termination_point=101, average_rank=102)
        insert_data_to_db(database=self.database, test_dataset_id=1,
                          training_dataset_id=1, environment_id=1, distance=15,
                          device=8, training_model_id=1, keybyte=0, epoch=100,
                          additive_noise_method_id=1, denoising_method_id=None,
                          termination_point=101, average_rank=102)
        data = fetchall_query(self.database, QUERY_FULL_RANK_TEST_GROUPED_A)
        self.assertIsNotNone(data)
        self.assertNotEqual(data, [])

    def test_get_test_trace_path(self):
        test_dataset_id = 1
        environment_id = 1
        distance = 15
        device = 6

        test_path = get_test_trace_path(
            database=self.database,
            test_dataset_id=test_dataset_id,
            environment_id=environment_id,
            distance=distance,
            device=device
        )
        self.assertIsNotNone(test_path)

    def test_get_training_model_file_path(self):
        training_model_id = 1
        additive_noise_method_id = None
        denoising_method_id = 1
        epoch = 50
        keybyte = 0

        training_model_file_path = get_training_model_file_path(
            database=self.database,
            training_model_id=training_model_id,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
            epoch=epoch,
            keybyte=keybyte
        )

        self.assertIsNotNone(training_model_file_path)


class TerminationPointTest(unittest.TestCase):

    def setUp(self) -> None:
        self.database = "test_database.db"
        create_db_with_tables(self.database)
        initialize_table_data(self.database)
        self.con = lite.connect(get_db_file_path(self.database))
        self.cur = self.con.cursor()

    def tearDown(self) -> None:
        self.con.close()
        os.remove(get_db_file_path(self.database))

    def test_termination_point_test_and_insert_to_db__additive_None(self):
        runs = 1
        test_dataset_id = 1
        training_dataset_id = 1
        environment_id = 1
        training_model_id = 1
        distance = 15
        device = 6
        keybyte = 0
        epoch = 65
        additive_noise_method_id = None
        denoising_method_id = None
        termination_point_test_and_insert_to_db(
            database=self.database,
            runs=runs,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            training_model_id=training_model_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
            keybyte=keybyte,
            epoch=epoch,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
        )

        data = fetchall_query(self.database, "select Count(*) from rank_test;")
        self.assertNotEqual(data, [])
        self.assertEqual(data[0][0], 1)

    def test_termination_point_test_and_insert_to_db__additive_1(self):
        runs = 1
        test_dataset_id = 1
        training_dataset_id = 1
        environment_id = 1
        training_model_id = 1
        distance = 15
        device = 6
        keybyte = 0
        epoch = 65
        additive_noise_method_id = 1
        denoising_method_id = None
        termination_point_test_and_insert_to_db(
            database=self.database,
            runs=runs,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            training_model_id=training_model_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
            keybyte=keybyte,
            epoch=epoch,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
        )

        data = fetchall_query(self.database, "select Count(*) from rank_test;")
        self.assertNotEqual(data, [])
        self.assertEqual(data[0][0], 1)

    def test_termination_point_test_and_insert_to_db__denoising_1(self):
        runs = 1
        test_dataset_id = 1
        training_dataset_id = 1
        environment_id = 1
        training_model_id = 1
        distance = 15
        device = 6
        keybyte = 0
        epoch = 65
        additive_noise_method_id = None
        denoising_method_id = 1
        termination_point_test_and_insert_to_db(
            database=self.database,
            runs=runs,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            training_model_id=training_model_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
            keybyte=keybyte,
            epoch=epoch,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
        )

        data = fetchall_query(self.database, "select Count(*) from rank_test;")
        self.assertNotEqual(data, [])
        # self.assertEqual(data[0][0], 0)
