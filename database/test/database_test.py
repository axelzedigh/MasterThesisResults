import unittest
import sqlite3 as lite
import os

from database.db_utils import (
    create_db_with_tables,
    initialize_table_data,
    insert_data_to_db,
    fetchall_query,
)


class AddToDatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.database = "test_database.db"
        create_db_with_tables(self.database)
        initialize_table_data(self.database)
        self.con = lite.connect(self.database)
        self.cur = self.con.cursor()

    def tearDown(self) -> None:
        self.con.close()
        os.remove(self.database)

    def test_fetch_environments(self):
        fetchall = self.cur.execute("SELECT * FROM Environments;").fetchall()
        environments = [(1, "office corridor"), (2, "big hall")]
        self.assertEqual(fetchall[0], environments[0])
        self.assertEqual(fetchall[1], environments[1])

    def test_fetch_test_datasets(self):
        fetchall = self.cur.execute("SELECT * FROM Test_Datasets;").fetchall()
        test_datasets = [(1, "Wang2021"), (2, "Zedigh2021")]
        self.assertIsNotNone(fetchall)
        self.assertEqual(fetchall[0], test_datasets[0])
        self.assertEqual(fetchall[1], test_datasets[1])

    def test_fetch_training_datasets(self):
        fetchall = self.cur.execute("SELECT * FROM Training_Datasets;").fetchall()
        self.assertIsNotNone(fetchall)
        training_datasets = [(1, "Wang2021 - Cable")]
        self.assertEqual(fetchall[0], training_datasets[0])

    def test_fetch_training_models(self):
        fetchall = self.cur.execute("SELECT * FROM Training_Models;").fetchall()
        self.assertIsNotNone(fetchall)
        training_models = [(1, "CNN110")]
        self.assertEqual(fetchall[0], training_models[0])

    def test_fetch_additive_noise_methods(self):
        fetchall = self.cur.execute("SELECT * FROM Additive_Noise_Methods;").fetchall()
        self.assertIsNotNone(fetchall)
        additive_noise_methods = [(1, "Gaussian", 1, 5)]
        self.assertEqual(fetchall[0], additive_noise_methods[0])

    def test_fetch_denoising_methods(self):
        fetchall = self.cur.execute("SELECT * FROM Denoising_Methods;").fetchall()
        self.assertIsNotNone(fetchall)
        denoising_methods = [(1, "Moving Average Filter", 12, None)]
        self.assertEqual(fetchall[0], denoising_methods[0])

    def test_fetch_parameters(self):
        fetchall = self.cur.execute("SELECT * FROM Parameters;").fetchall()
        self.assertIsNotNone(fetchall)
        parameters = [
            (1, "Std", 0.01),
            (2, "Std", 0.02),
            (3, "Std", 0.03),
            (4, "Std", 0.04),
            (5, "Mean", 0.0),
            (6, "Scale", 25.0),
            (7, "Scale", 50.0),
            (8, "Scale", 75.0),
            (9, "Scale", 105.0),
            (10, "Mode", 0.0138),
            (11, "Mode", 0.0276),
            (12, "N", 3.0),
        ]
        self.assertEqual(fetchall[0], parameters[0])

    def test_insert_denoising_method(self):
        self.cur.execute("INSERT INTO Denoising_Methods VALUES(NULL,'CDAE',NULL,NULL)")
        fetchall = self.cur.execute("SELECT * FROM Denoising_Methods;").fetchall()
        self.assertEqual("CDAE", fetchall[2][1])

    def test_fetch_view_full_additive_noise_methods(self):
        fetchall = self.cur.execute(
            "SELECT * FROM full_additive_noise_methods;"
        ).fetchall()
        self.assertIsNotNone(fetchall)
        data = [
            (1, "Gaussian", "Std", 0.01, "Mean", 0.0),
            (2, "Gaussian", "Std", 0.02, "Mean", 0.0),
            (3, "Gaussian", "Std", 0.03, "Mean", 0.0),
            (4, "Gaussian", "Std", 0.04, "Mean", 0.0),
        ]
        self.assertEqual(fetchall[0], data[0])

    def test_fetch_view_full_denoising_methods(self):
        fetchall = self.cur.execute(
            "SELECT * FROM full_denoising_methods;"
        ).fetchall()
        self.assertIsNotNone(fetchall)
        self.assertNotEqual(fetchall, [])
        print(fetchall)

    def test_insert_rank_test_data(self):
        insert_data_to_db(
            database=self.database,
            test_dataset_id=1,
            training_dataset_id=2,
            environment_id=1,
            distance=15,
            device=8,
            training_model_id=1,
            keybyte=0,
            epoch=100,
            additive_noise_method_id=None,
            denoising_method_id=None,
            termination_point=9999,
            average_rank=9999,
        )
        insert_data_to_db(
            database=self.database,
            test_dataset_id=1,
            training_dataset_id=1,
            environment_id=1,
            distance=15,
            device=7,
            training_model_id=1,
            keybyte=0,
            epoch=100,
            additive_noise_method_id=None,
            denoising_method_id=None,
            termination_point=101,
            average_rank=102,
        )
        fetchall = self.cur.execute("SELECT * FROM Rank_Test;").fetchall()
        self.assertIsNotNone(fetchall)
        device = fetchall[0][5]
        self.assertEqual(8, device)
        self.assertNotEqual(fetchall, [])

    def test_view_query(self):
        insert_data_to_db(
            database=self.database,
            test_dataset_id=1,
            training_dataset_id=1,
            environment_id=1,
            distance=15,
            device=7,
            training_model_id=1,
            keybyte=0,
            epoch=100,
            additive_noise_method_id=2,
            denoising_method_id=None,
            termination_point=101,
            average_rank=102,
        )
        query = "SELECT * FROM full_rank_test;"
        data = fetchall_query(self.database, query)
        self.assertIsNotNone(data)
        self.assertNotEqual(data, [])
        print(data)
