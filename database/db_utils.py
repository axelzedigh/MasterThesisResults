import sqlite3 as lite
import os
import datetime
from database.queries import (
    QUERY_CREATE_TABLE_ENVIRONMENTS,
    QUERY_CREATE_TABLE_TEST_DATASETS,
    QUERY_CREATE_TABLE_TRAINING_DATASETS,
    QUERY_CREATE_TABLE_TRAINING_MODELS,
    QUERY_CREATE_TABLE_ADDITIVE_NOISE_METHODS,
    QUERY_CREATE_TABLE_DENOISING_METHODS,
    QUERY_CREATE_TABLE_RANK_TEST,
    QUERY_CREATE_VIEW_FULL_RANK_TEST,
    QUERY_SELECT_ADDITIVE_NOISE_METHOD_ID, QUERY_SELECT_DENOISING_METHOD_ID,
)


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

        # Create tables
        cur.execute(QUERY_CREATE_TABLE_ENVIRONMENTS)
        cur.execute(QUERY_CREATE_TABLE_TEST_DATASETS)
        cur.execute(QUERY_CREATE_TABLE_TRAINING_DATASETS)
        cur.execute(QUERY_CREATE_TABLE_TRAINING_MODELS)
        cur.execute(QUERY_CREATE_TABLE_ADDITIVE_NOISE_METHODS)
        cur.execute(QUERY_CREATE_TABLE_DENOISING_METHODS)
        cur.execute(QUERY_CREATE_TABLE_RANK_TEST)

        # Create views
        con.execute(QUERY_CREATE_VIEW_FULL_RANK_TEST)

        # Close connections
        cur.close()
        con.close()
    return


def initialize_table_data(database):
    """

    :param database:
    :return:
    """
    dirs = os.listdir()
    if database in dirs:
        con = lite.connect(database)
        cur = con.cursor()
        cur.execute("INSERT INTO environments VALUES (1,'office corridor');")
        cur.execute("INSERT INTO environments VALUES (2,'big hall');")

        cur.execute("INSERT INTO test_datasets VALUES (1,'Wang2021');")
        cur.execute("INSERT INTO test_datasets VALUES (2,'Zedigh2021');")

        cur.execute(
            "INSERT INTO training_datasets VALUES (1,'Wang2021 - Cable');")

        cur.execute("INSERT INTO training_models VALUES (1,'CNN110');")

        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (1,'Gaussian', 'Std', 0.01, 'Mean', 0);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (2,'Gaussian', 'Std', 0.02, 'Mean', 0);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (3,'Gaussian', 'Std', 0.03, 'Mean', 0);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (4,'Gaussian', 'Std', 0.04, 'Mean', 0);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (5,'Gaussian', 'Std', 0.05, 'Mean', 0);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (6,'Collected', 'Scale', 25, NULL, NULL);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (7,'Collected', 'Scale', 50, NULL, NULL);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (8,'Collected', 'Scale', 75, NULL, NULL);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (9,'Collected', 'Scale', 105, NULL, NULL);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (10,'Rayleigh', 'Mode', 0.0138, NULL, NULL);"
        )
        cur.execute(
            "INSERT INTO additive_noise_methods VALUES (11,'Rayleigh', 'Mode', 0.0276, NULL, NULL);"
        )

        cur.execute(
            "INSERT INTO denoising_methods VALUES (1,'Moving Average Filter', 'N', 3, NULL, NULL);"
        )
        cur.execute(
            "INSERT INTO denoising_methods VALUES (2,'Moving Average Filter', 'N', 5, NULL, NULL);"
        )

        con.commit()
        con.close()
        return
    else:
        print("Database file don't exist!")
        return


def insert_data_to_db(
        database: str = "TerminationPoints.db",
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

    cur.execute(
        "INSERT INTO Rank_Test VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            test_dataset_id,
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
            date_added,
        ),
    )
    con.commit()
    con.close()


def fetchall_query(database="TerminationPoints.db",
                   query="SELECT * FROM Rank_Test;"):
    con = lite.connect(database)
    cur = con.cursor()
    query_data = cur.execute(query).fetchall()
    con.close()
    return query_data


def get_additive_noise_method_id(
        database: str,
        additive_noise_method: str,
        parameter_1: float,
        parameter_1_value: str,
        parameter_2: str,
        parameter_2_value: str,
):
    query_arguments = (
        additive_noise_method,
        parameter_1,
        parameter_1_value,
        parameter_2,
        parameter_2_value,
    )
    con = lite.connect(database)
    cur = con.cursor()
    additive_noise_method_id = cur.execute(
        QUERY_SELECT_ADDITIVE_NOISE_METHOD_ID, query_arguments
    ).fetchall()
    if len(additive_noise_method_id) == 1:
        return additive_noise_method_id[0][0]
    else:
        print("Something is wrong!")


def get_denoising_method_id(
        database: str,
        denoising_method: str,
        parameter_1: str,
        parameter_1_value: float,
        parameter_2: str,
        parameter_2_value: float,
):
    query_arguments = (
        denoising_method,
        parameter_1,
        parameter_1_value,
        parameter_2,
        parameter_2_value,
    )
    con = lite.connect(database)
    cur = con.cursor()
    denoising_method_id = cur.execute(
        QUERY_SELECT_DENOISING_METHOD_ID, query_arguments
    ).fetchall()
    if len(denoising_method_id) == 1:
        return denoising_method_id[0][0]
    else:
        print("Something is wrong!")
