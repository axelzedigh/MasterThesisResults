import sqlite3 as lite
import os
import numpy as np
import re
from typing import Optional
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
    QUERY_LIST_INITIALIZE_DB,
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
        for QUERY in QUERY_LIST_INITIALIZE_DB:
            cur.execute(QUERY)

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
        average_rank: Optional[int] = 9999,
) -> None:
    """

    :param database: The database-file to write to.
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance: Distance between device under tests and antenna.
    :param device: Device under tests.
    :param training_model_id: The deep learning architecture model used.
    :param keybyte: The keybyte trained and tested [0-15].
    :param epoch: The epoch of the DL model. Between 1-100.
    :param additive_noise_method_id: Foreign key id.
    :param denoising_method_id: Foreign key id.
    :param termination_point: Termination point from rank tests.
    :param average_rank:
    """
    create_db_with_tables(database)
    con = lite.connect(database)
    cur = con.cursor()

    cur.execute(
        "INSERT INTO Rank_Test VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,julianday("
        "'now'))",
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
        ),
    )
    con.commit()
    con.close()


def fetchall_query(database: str = "TerminationPoints.db",
                   query: str = "SELECT * FROM full_rank_test;"):
    con = lite.connect(database)
    cur = con.cursor()
    query_data = cur.execute(query).fetchall()
    con.close()
    return query_data


def get_additive_noise_method_id(
        database: str,
        additive_noise_method: str,
        parameter_1: Optional[str],
        parameter_1_value: Optional[float],
        parameter_2: Optional[str],
        parameter_2_value: Optional[float],
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
        parameter_1: Optional[str],
        parameter_1_value: Optional[float],
        parameter_2: Optional[str],
        parameter_2_value: Optional[float],
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


def insert_legacy_rank_test_numpy_file_to_db(
        database: str,
        filename: str,
        test_dataset_id: int,
        training_dataset_id: int,
        environment_id: int,
        distance: float,
        training_model_id: int,
        additive_noise_method_id: Optional[int],
        denoising_method_id: Optional[int],
):
    """
    Numpy filename-format:
    rank_test-device-{i}-epoch-{i}-keybyte-{i}-runs-{i}-{training_model}-{
    noise processing name}.npy

    :param database: Path to database to write to.
    :param filename: Path to numpy file.
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param training_model_id:
    :param additive_noise_method_id:
    :param denoising_method_id:
    """
    file = np.load(filename)

    device_start = re.search("device-", filename).end()
    device_end = re.search("-epoch", filename).start()
    device = int(filename[device_start:device_end])

    epoch_start = re.search("epoch-", filename).end()
    epoch_end = re.search("-keybyte", filename).start()
    epoch = int(filename[epoch_start:epoch_end])

    keybyte_start = re.search("keybyte-", filename).end()
    keybyte_end = re.search("-runs", filename).start()
    keybyte = int(filename[keybyte_start:keybyte_end])

    # model_start = re.search("cnn_110_", filename).end()
    # model_end = re.search(".npy", filename).start()
    # model = file[model_start:model_end]

    for termination_point in file:
        insert_data_to_db(
            database=database,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
            training_model_id=training_model_id,
            keybyte=keybyte,
            epoch=epoch,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
            termination_point=int(termination_point),
            average_rank=None,
        )
