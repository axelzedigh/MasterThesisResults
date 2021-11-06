import sqlite3 as lite
import os
import datetime


def create_db_with_tables(database="TerminationPoints.db") -> None:
    """

    :return:
    """
    # TODO list not just the current path but the absolute path to the database
    # e.g. os.listdir(os.path.notbasename(database))
    dirs = os.listdir()
    if database in dirs:
        # print("Database exists!")
        return
    else:
        con = lite.connect(database)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE Environments(
            id INTEGER PRIMARY KEY,
            environment TEXT NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Test_Datasets(
            id INTEGER PRIMARY KEY,
            test_dataset TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Training_Datasets(
            id INTEGER PRIMARY KEY,
            training_dataset TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Training_Models(
            id INTEGER PRIMARY KEY,
            training_model TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Additive_Noise_Methods(
            id INTEGER PRIMARY KEY,
            additive_noise_method TEXT,
            additive_noise_parameter_1 TEXT,
            additive_noise_parameter_1_value FLOAT,
            additive_noise_parameter_2 TEXT,
            additive_noise_parameter_2_value FLOAT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Denoising_Methods(
            id INTEGER PRIMARY KEY,
            denoising_method TEXT,
            denoising_parameter_1 TEXT,
            denoising_parameter_1_value FLOAT,
            denoising_parameter_2 TEXT,
            denoising_parameter_2_value FLOAT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Rank_Test(
            id INTEGER PRIMARY KEY,
            test_dataset_id INTEGER,
            training_dataset_id INTEGER,
            environment_id INTEGER,
            distance FLOAT NOT NULL,
            device INT NOT NULL, 
            training_model_id INTEGER,
            keybyte INT NOT NULL,
            epoch INT NOT NULL,
            additive_noise_method_id INTEGER,
            denoising_method_id INTEGER,
            termination_point INT NOT NULL,
            average_rank INT NOT NULL,
            date_added TEXT NOT NULL,
            
            FOREIGN KEY(test_dataset_id) REFERENCES Test_Datasets(test_dataset_id),
            FOREIGN KEY(training_dataset_id) REFERENCES Training_Datasets(training_dataset_id),
            FOREIGN KEY(environment_id) REFERENCES Environments(environments_id),
            FOREIGN KEY(training_model_id) REFERENCES Training_Model(training_model_id),
            FOREIGN KEY(additive_noise_method_id) REFERENCES Additive_Noise_Methods(additive_noise_method_id),
            FOREIGN KEY(denoising_method_id) REFERENCES Denoising_Method(denoising_method_id)
            )
            """
        )

        con.execute(
        """
        CREATE VIEW full_rank_test
        AS
        SELECT
            Rank_Test.id,
            Test_Datasets.test_dataset AS test_dataset,
            Training_Datasets.training_dataset AS training_dataset,
            Environments.environment AS environment,
            Rank_Test.distance,
            Rank_Test.device,
            Training_Models.training_model AS training_model,
            Rank_Test.keybyte,
            Rank_Test.epoch,
            Additive_Noise_Methods.additive_noise_method AS additive_noise_method,
            Additive_Noise_Methods.additive_noise_parameter_1 AS additive_noise_method_parameter_1,
            Additive_Noise_Methods.additive_noise_parameter_1_value AS additive_noise_method_parameter_1_value,
            Additive_Noise_Methods.additive_noise_parameter_2 AS additive_noise_method_parameter_2,
            Additive_Noise_Methods.additive_noise_parameter_2_value AS additive_noise_method_parameter_2_value,
            Denoising_Methods.denoising_method AS denoising_method,
            Denoising_Methods.denoising_parameter_1 AS denoising_method_parameter_1,
            Denoising_Methods.denoising_parameter_1_value AS denoising_method_parameter_1_value,
            Denoising_Methods.denoising_parameter_2 AS denoising_method_parameter_2,
            Denoising_Methods.denoising_parameter_2_value AS denoising_method_parameter_2_value,
            Rank_Test.termination_point,
            Rank_Test.average_rank,
            Rank_Test.date_added
        FROM
            Rank_Test
        INNER JOIN Test_Datasets ON Test_Datasets.id = Rank_Test.id
        INNER JOIN Training_Datasets ON Training_Datasets.id = Rank_Test.id
        INNER JOIN Environments ON Environments.id = Rank_TEst.id
        INNER JOIN Training_Models on Training_Models.id = Rank_Test.id
        INNER JOIN Additive_Noise_Methods ON Additive_Noise_Methods.id = Rank_test.id
        INNER JOIN Denoising_Methods ON Denoising_Methods.id = Rank_Test.id;
        """
        )
    return


def initialize_table_data(database):
    dirs = os.listdir()
    if database in dirs:
        con = lite.connect(database)
        cur = con.cursor()
        cur.execute("INSERT INTO environments VALUES (1,'office corridor');")
        cur.execute("INSERT INTO environments VALUES (2,'big hall');")

        cur.execute("INSERT INTO test_datasets VALUES (1,'Wang2021');")
        cur.execute("INSERT INTO test_datasets VALUES (2,'Zedigh2021');")

        cur.execute("INSERT INTO training_datasets VALUES (1,'Wang2021 - Cable');")

        cur.execute("INSERT INTO training_models VALUES (1,'CNN110');")

        cur.execute("INSERT INTO additive_noise_methods VALUES (1,'Gaussian', 'Std', 0.01, 'Mean', 0);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (2,'Gaussian', 'Std', 0.02, 'Mean', 0);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (3,'Gaussian', 'Std', 0.03, 'Mean', 0);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (4,'Gaussian', 'Std', 0.04, 'Mean', 0);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (5,'Gaussian', 'Std', 0.05, 'Mean', 0);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (6,'Collected', 'Scale', 25, NULL, NULL);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (7,'Collected', 'Scale', 50, NULL, NULL);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (8,'Collected', 'Scale', 75, NULL, NULL);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (9,'Collected', 'Scale', 105, NULL, NULL);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (10,'Rayleigh', 'Mode', 0.0138, NULL, NULL);")
        cur.execute("INSERT INTO additive_noise_methods VALUES (11,'Rayleigh', 'Mode', 0.0276, NULL, NULL);")

        cur.execute("INSERT INTO denoising_methods VALUES (1,'Moving Average Filter', 'N', 3, NULL, NULL);")

        con.commit()
        con.close()
        return
    else:
        print("Database file don't exist!")
        return


def insert_data_to_db(
        database="TerminationPoints.db",
        test_dataset_id: int = 1,
        training_dataset_id: int = 1,
        environment_id: int = 1,
        distance: float = 15,
        device: int = 8,
        training_model_id: int = 1,
        keybyte: int = 0,
        epoch: int = 100,
        additive_noise_method_id: int = None,
        denoising_method_id: int = None,
        termination_point: int = 9999,
        average_rank: int = 9999,
) -> None:
    """

    :param database: The database-file to write to. Standard is "TerminationPoints.db".
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance: Distance between device under test and antenna.
    :param device: Device under test.
    :param training_model_id: The deep learning architecture model used, e.g. CNN110.
    :param keybyte: The keybyte trained and tested. Between 0-15.
    :param epoch: The epoch of the DL model. Between 1-100.
    :param additive_noise_method_id:
    :param denoising_method_id:
    :param termination_point: Termination point from rank test. Dependent variable!
    :param average_rank: Average rank of the
    """
    create_db_with_tables(database)
    con = lite.connect(database)
    cur = con.cursor()
    date_added = str(datetime.datetime.today())

    cur.execute("INSERT INTO Rank_Test VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (test_dataset_id,
                 training_dataset_id,
                 environment_id,
                 distance,
                 device,
                 training_model_id,
                 keybyte,
                 epoch,
                 additive_noise_method_id,
                 denoising_method_id,
                 termination_point,
                 average_rank,
                 date_added)
                )
    con.commit()
    con.close()


def fetch_all_from_db(database="TerminationPoints.db", query="SELECT * from full_rank_test;") -> list:
    """

    :param database:
    :return: a list containing all database entries from Rank_Test table
    """
    con = lite.connect(database)
    cur = con.cursor()
    all_data = cur.execute(query).fetchall()
    con.close()
    return all_data


def fetchall_query(database="TerminationPoints.db", query="SELECT * FROM Rank_Test;"):
    con = lite.connect(database)
    cur = con.cursor()
    query_data = cur.execute(query).fetchall()
    con.close()
    return query_data

