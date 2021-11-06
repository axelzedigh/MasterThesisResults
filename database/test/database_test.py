import unittest
import sqlite3 as lite
import os

from database.db_utils import create_db_with_tables, initialize_table_data, insert_data_to_db


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
        self.assertEqual(fetchall[0], additive_noise_methods[0])

    def test_fetch_denoising_methods(self):
        fetchall = self.cur.execute("SELECT * FROM Denoising_Methods;").fetchall()
        self.assertIsNotNone(fetchall)
        denoising_methods = [(1, 'Moving Average Filter', 'N', 3.0, None, None)]
        self.assertEqual(fetchall[0], denoising_methods[0])

    def test_insert_denoising_method(self):
        self.cur.execute("INSERT INTO Denoising_Methods VALUES(NULL,'CDAE',NULL,NULL, NULL, NULL)")
        fetchall = self.cur.execute("SELECT * FROM Denoising_Methods;").fetchall()
        self.assertEqual('CDAE', fetchall[1][1])

    def test_insert_rank_test_data(self):
        insert_data_to_db(
            database=self.database,
            testing_dataset=1,
            training_dataset=2,
            environment=1,
            distance=15,
            device=8,
            training_model=1,
            keybyte=0,
            epoch=100,
            additive_noise_method=None,
            denoising_method=None,
            termination_point=9999,
            average_rank=9999,
        )
        insert_data_to_db(
            database=self.database,
            testing_dataset=1,
            training_dataset=1,
            environment=1,
            distance=15,
            device=7,
            training_model=1,
            keybyte=0,
            epoch=100,
            additive_noise_method=None,
            denoising_method=None,
            termination_point=101,
            average_rank=102,
        )
        fetchall = self.cur.execute("SELECT * FROM Rank_Test;").fetchall()
        self.assertIsNotNone(fetchall)
        device = fetchall[0][5]
        self.assertEqual(8, device)
