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
        print("Database already exists!")
        return
    else:
        con = lite.connect(database)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE Environments(
            environment_id INTEGER PRIMARY KEY,
            environment TEXT NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Test_Datasets(
            test_dataset_id INTEGER PRIMARY KEY,
            test_dataset TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Training_Datasets(
            training_dataset_id INTEGER PRIMARY KEY,
            training_dataset TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Training_Models(
            training_model_id INTEGER PRIMARY KEY,
            training_model TEXT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Additive_Noise_Methods(
            additive_noise_method_id INTEGER PRIMARY KEY,
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
            denoising_method_id INTEGER PRIMARY KEY,
            denoising_method TEXT,
            denoising_method_parameter_1 TEXT,
            denoising_method_parameter_1_value FLOAT,
            denoising_method_parameter_2 TEXT,
            denoising_method_parameter_2_value FLOAT
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE Rank_Test(
            id INTEGER PRIMARY KEY,
            testing_dataset INTEGER,
            training_dataset INTEGER,
            environment INTEGER,
            distance FLOAT NOT NULL,
            device INT NOT NULL, 
            training_model INTEGER,
            keybyte INT NOT NULL,
            epoch INT NOT NULL,
            additive_noise_method INTEGER,
            denoising_method INTEGER,
            termination_point INT NOT NULL,
            average_rank INT NOT NULL,
            date_added TEXT NOT NULL,
            
            FOREIGN KEY(testing_dataset) REFERENCES Testing_Datasets(testing_dataset_id),
            FOREIGN KEY(training_dataset) REFERENCES Training_Datasets(training_dataset_id),
            FOREIGN KEY(environment) REFERENCES Environments(environments_id),
            FOREIGN KEY(training_model) REFERENCES Training_Model(training_model_id),
            FOREIGN KEY(additive_noise_method) REFERENCES Additive_Noise_Methods(additive_noise_method_id),
            FOREIGN KEY(denoising_method) REFERENCES Denoising_Method(denoising_method_id)
            )
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
        testing_dataset: int = 1,
        training_dataset: int = 1,
        environment: int = 1,
        distance: float = 15,
        device: int = 8,
        training_model: int = 1,
        keybyte: int = 0,
        epoch: int = 100,
        additive_noise_method: str = None,
        additive_noise_parameter_1: str = None,
        additive_noise_parameter_1_value: float = None,
        additive_noise_parameter_2: str = None,
        additive_noise_parameter_2_value: float = None,
        denoising_method: str = None,
        denoising_method_parameter_1: str = None,
        denoising_method_parameter_1_value: float = None,
        denoising_method_parameter_2: str = None,
        denoising_method_parameter_2_value: float = None,
        termination_point: int = 9999,
        average_rank: int = 9999,
) -> None:
    """

    :param database: The database-file to write to. Standard is "TerminationPoints.db".
    :param testing_dataset: The dataset (either 'Wang2021' or 'Zedigh2021')
    :param training_dataset:
    :param environment:
    The environment the testing traces were collected. Either office or KTH Kista Hall.
    :param distance: Distance between device under test and antenna.
    :param device: Device under test.
    :param training_model: The deep learning architecture model used, e.g. CNN110.
    :param keybyte: The keybyte trained and tested. Between 0-15.
    :param epoch: The epoch of the DL model. Between 1-100.
    :param additive_noise_method:
    The additive noise method used during model training. E.g. Gaussian, Collected, Rayleigh etc.
    :param additive_noise_parameter_1:
    First parameter of the additive noise method. E.g Mean, Scale, Mode etc.
    :param additive_noise_parameter_1_value:
    Value of the first additive noise parameter.
    :param additive_noise_parameter_2:
    Second parameter of the additive noise method. E.g. Std, Translation etc.
    :param additive_noise_parameter_2_value:
    Value of the second additive noise parameter.
    :param denoising_method:
    Denoising method used. E.g. Moving average filter, CDAE, Weiner filter.
    :param denoising_method_parameter_1:
    First parameter of the denoising parameter. E.g. N, etc.
    :param denoising_method_parameter_1_value: Value of the first denoising method.
    :param denoising_method_parameter_2: Second parameter of the denoising method.
    :param denoising_method_parameter_2_value: Value of the second denoising parameter.
    :param termination_point: Termination point from rank test. Dependent variable!
    :param average_rank: Average rank of the
    """
    create_db_with_tables(database)
    con = lite.connect(database)
    cur = con.cursor()
    date_added = str(datetime.datetime.today())

    cur.execute("INSERT INTO RankTest VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (testing_dataset,
                 environment,
                 distance,
                 device,
                 training_model,
                 keybyte,
                 epoch,
                 additive_noise_method,
                 denoising_method,
                 termination_point,
                 average_rank,
                 date_added)
                )
    con.commit()
    con.close()


def fetchall_rank_test(database="TerminationPoints.db") -> list:
    """

    :param database:
    :return: a list containing all database entries
    """
    con = lite.connect(database)
    cur = con.cursor()
    all_data = cur.execute("SELECT * FROM RankTest;").fetchall()
    con.close()
    return all_data


if __name__ == "__main__":
    create_db_with_tables()
