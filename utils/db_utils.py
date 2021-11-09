import sqlite3 as lite
import os
import numpy as np
import re
from datetime import datetime
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


def create_db_with_tables(database="main.db") -> None:
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


def initialize_table_data(database="main.db"):
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
        database: str,
        test_dataset_id: int,
        training_dataset_id: int,
        environment_id: int,
        distance: float,
        device: int,
        training_model_id: int,
        keybyte: int,
        epoch: int,
        additive_noise_method_id: Optional[int],
        denoising_method_id: Optional[int],
        termination_point: int,
        average_rank: Optional[int],
) -> None:
    """

    :param database: The database-file to write to.
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance: Distance between device under unittests and antenna.
    :param device: Device under unittests.
    :param training_model_id: The deep learning architecture model used.
    :param keybyte: The keybyte trained and tested [0-15].
    :param epoch: The epoch of the DL model. Between 1-100.
    :param additive_noise_method_id: Foreign key id.
    :param denoising_method_id: Foreign key id.
    :param termination_point: Termination point from rank unittests.
    :param average_rank:
    """
    # create_db_with_tables(database)
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


def fetchall_query(database: str = "main.db",
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
        file_path: str,
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
    :param file_path: Path to numpy file.
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param training_model_id:
    :param additive_noise_method_id:
    :param denoising_method_id:
    """
    file = np.load(file_path)

    device_start = re.search("device-", file_path).end()
    device_end = re.search("-epoch", file_path).start()
    device = int(file_path[device_start:device_end])

    epoch_start = re.search("epoch-", file_path).end()
    epoch_end = re.search("-keybyte", file_path).start()
    epoch = int(file_path[epoch_start:epoch_end])

    keybyte_start = re.search("keybyte-", file_path).end()
    keybyte_end = re.search("-runs", file_path).start()
    keybyte = int(file_path[keybyte_start:keybyte_end])

    # model_start = re.search("cnn_110_", filename).end()
    # model_end = re.search(".npy", filename).start()
    # model = file[model_start:model_end]

    for termination_point in file:
        insert_data_to_db(database=database, test_dataset_id=test_dataset_id,
                          training_dataset_id=training_dataset_id,
                          environment_id=environment_id, distance=distance,
                          device=device, training_model_id=training_model_id,
                          keybyte=keybyte, epoch=epoch,
                          additive_noise_method_id=additive_noise_method_id,
                          denoising_method_id=denoising_method_id,
                          termination_point=int(termination_point),
                          average_rank=None)


def create_pre_processing_table_info_md(database="main.db", path="docs"):
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    file_path = os.path.join(project_dir, path, "pre_processing_tables.md")
    additive_data = fetchall_query(
        database, "SELECT * from Additive_Noise_Methods;"
    )
    denoising_data = fetchall_query(
        database, "SELECT * from Denoising_Methods;"
    )
    file = open(file_path, "w")
    file.write("# Pre-processing tables\n")
    file.write(f"Last updated: {datetime.today()}\n\n")
    file.write("## Additive noise methods\n")
    file.close()
    file = open(file_path, "a")
    file.write("| id | additive noise method | parameter 1 | value | "
               "parameter 2 | value |\n")
    file.write("|---|---|---|---|---|---|\n")
    for additive_method in additive_data:
        file.write(
            f"| {additive_method[0]} | {additive_method[1]} |"
            f"{additive_method[2]} | {additive_method[3]} | "
            f"{additive_method[4]} | {additive_method[5]} |\n"
        )

    file.write("\n")
    file.write("## Denoising methods\n")
    file.write("| id | denoising method | parameter 1 | value | parameter 2 | "
               "value |\n")
    file.write("|---|---|---|---|---|---|\n")
    for denoising_method in denoising_data:
        file.write(
            f"| {denoising_method[0]} | {denoising_method[1]} |"
            f"{denoising_method[2]} | {denoising_method[3]} | "
            f"{denoising_method[4]} | {denoising_method[5]} |\n"
        )

    file.write("\n")
    file.close()


def create_rank_test_table_info_md(database="main.db", path="docs"):
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    file_path = os.path.join(project_dir, path, "rank_test_table_info.md")
    rank_test_rows = fetchall_query(
        database, "SELECT Count(*) from Rank_Test;"
    )

    file = open(file_path, "w")
    file.write("# Rank Test Table Info\n")
    file.write(f"Last updated: {datetime.today()}\n\n")
    file.close()

    file = open(file_path, "a")
    file.write(f"Rows: {rank_test_rows[0][0]}")
    file.write("\n")
    file.close()


def get_db_absolute_path(database="main.db", path="database"):
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    database = os.path.join(project_dir, path, database)
    return database


def get_test_trace_path(
        database,
        test_dataset_id,
        environment_id,
        distance,
        device
) -> str:
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    path = "datasets/test_traces"
    test_dataset_query = f"""
    select test_dataset from test_datasets
    where id = {test_dataset_id};"""
    test_dataset = fetchall_query(database, test_dataset_query)[0][0]

    environment_query = f"""
    select environment from environments
    where id = {environment_id};"""
    environment = fetchall_query(database, environment_query)[0][0]

    test_traces_path = os.path.join(
        project_dir,
        path,
        test_dataset,
        environment,
        f"{distance}m",
        f"device_{device}",
        "data"
    )

    return test_traces_path


def get_training_model_file_path(
        database,
        training_model_id,
        additive_noise_method_id,
        denoising_method_id,
        epoch,
        keybyte
) -> str:
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    path = "models"

    training_model_query = f"""
    select training_model from training_models
    where id = {training_model_id};"""
    training_model = fetchall_query(
        database, training_model_query)[0][0]

    if additive_noise_method_id is None:
        additive_noise_method_id = "None"

    if denoising_method_id is None:
        denoising_method_id = "None"

    training_model_file_path = os.path.join(
        project_dir,
        path,
        f"keybyte_{keybyte}",
        f"{additive_noise_method_id}_{denoising_method_id}",
        f"{training_model}-{epoch}.h5"
    )

    return training_model_file_path
