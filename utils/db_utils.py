"""Util functions for interacting with the databases."""
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
    QUERY_LIST_INITIALIZE_DB, QUERY_FULL_RANK_TEST_GROUPED_A,
    QUERY_CREATE_TABLE_TRACE_PROCESSES,
    QUERY_CREATE_TABLE_TRACE_METADATA_DEPTH,
)


def get_db_file_path(database="main.db"):
    """

    :param database: Name of database.
    :return: Path-string to database-file.
    """
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    database_dir = os.path.join(project_dir, "database", database)
    return database_dir


def get_db_dir_path():
    """

    :return: Path-string to database-folder.
    """
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    database_dir = os.path.join(project_dir, "database")
    return database_dir


def create_db_with_tables(database="main.db") -> None:
    """

    :param database: Database name.
    """
    database_dir = get_db_dir_path()
    dirs = os.listdir(database_dir)
    if database in dirs:
        # print("Database exists!")
        pass
    else:
        database = get_db_file_path(database)
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
        cur.execute(QUERY_CREATE_TABLE_TRACE_PROCESSES)
        cur.execute(QUERY_CREATE_TABLE_TRACE_METADATA_DEPTH)

        # Create views
        con.execute(QUERY_CREATE_VIEW_FULL_RANK_TEST)

        # Close connections
        cur.close()
        con.close()


def initialize_table_data(database="main.db"):
    """

    :param database:
    """
    database_dir = get_db_dir_path()
    dirs = os.listdir(database_dir)
    if database in dirs:
        database = get_db_file_path(database)
        con = lite.connect(database)
        cur = con.cursor()
        for QUERY in QUERY_LIST_INITIALIZE_DB:
            cur.execute(QUERY)

        con.commit()
        con.close()
    else:
        print("Database file don't exist!")


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
    database = get_db_file_path(database)
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
    """

    :param database:
    :param query:
    :return:
    """
    database = get_db_file_path(database)
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
    """

    :param database:
    :param additive_noise_method:
    :param parameter_1:
    :param parameter_1_value:
    :param parameter_2:
    :param parameter_2_value:
    :return:
    """
    database = get_db_file_path(database)
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
    """

    :param database:
    :param denoising_method:
    :param parameter_1:
    :param parameter_1_value:
    :param parameter_2:
    :param parameter_2_value:
    :return:
    """
    database = get_db_file_path(database)
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
    database = get_db_file_path(database)
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


def create_md__option_tables(database="main.db", path="docs"):
    """

    :param database: Database to fetch data from.
    :param path: path in project dir to store the doc.
    """
    database = get_db_file_path(database)
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


def create_md__rank_test_tbl__meta_info(database="main.db", path="docs"):
    """

    :param database: Database to fetch data from.
    :param path: path in project dir to store the doc.
    """
    database = get_db_file_path(database)
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    file_path = os.path.join(project_dir, path, "rank_test_table_info.md")
    rank_test_rows = fetchall_query(
        database, "SELECT Count(*) from Rank_Test;"
    )
    duplicate_dates = fetchall_query(
        database,
        "SELECT date_added, COUNT(*) c FROM Rank_Test GROUP BY date_added HAVING c > 1;"
    )

    file = open(file_path, "w")
    file.write("# Rank Test Table Info\n")
    file.write(f"Last updated: {datetime.today()}\n\n")
    file.close()

    file = open(file_path, "a")
    file.write(f"Rows: {rank_test_rows[0][0]}\n\n")
    file.write(f"Duplicate date_added rows: {duplicate_dates}\n")
    file.write("\n")
    file.close()


def create_md__full_rank_test__grouped(database="main.db", path="docs"):
    """

    :param database: Database to fetch data from.
    :param path: path in project dir to store the doc.
    """
    database = get_db_file_path(database)
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    file_path = os.path.join(project_dir, path, "Rank_test__grouped.md")
    full_rank_test_rows = fetchall_query(
        database, QUERY_FULL_RANK_TEST_GROUPED_A
    )
    file = open(file_path, "w")

    file.write("# Full Rank test tables - Unique rows\n")
    file.write(f"Last updated: {datetime.today()}\n\n")
    file.close()
    file = open(file_path, "a")
    file.write("| test_dataset | training_dataset | environment | distance | "
               "device | training_model | keybyte | epoch | additive method | "
               "param 1| value | param 2 | value | denoising method | param 1 "
               "| value | param 2 | value | counted tp | avg tp's |\n")
    file.write(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        "---|---|---|\n"
    )
    for data in full_rank_test_rows:
        file.write(
            f"| {data[0]} | {data[1]} |"
            f"{data[2]} | {data[3]} | "
            f"{data[4]} | {data[5]} | "
            f"{data[6]} | {data[7]} | "
            f"{data[8]} | {data[9]} | "
            f"{data[10]} | {data[11]} | "
            f"{data[12]} | {data[13]} | "
            f"{data[14]} | {data[15]} | "
            f"{data[16]} | {data[17]} | "
            f"{data[18]} | {data[19]} |"
            "\n"
        )

    file.write("\n")
    file.close()


def get_db_absolute_path(database="main.db", path="database"):
    """

    :param database:
    :param path:
    :return:
    """
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
    """

    :param database:
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :return:
    """
    database = get_db_file_path(database)
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


def get_test_trace_path__raw_data(
        database,
        test_dataset_id,
        environment_id,
        distance,
        device
) -> str:
    """

    :param database:
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :return:
    """
    database = get_db_file_path(database)
    project_dir = os.getenv("MASTER_THESIS_RESULTS_RAW_DATA")
    if project_dir is None:
        raise """
        Either your computer don't have raw data or you forgot to set the 
        env-variable MASTER_THESIS_RESULTS_RAW_DATA to the raw data directory.
        """
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
    )

    return test_traces_path


def get_training_trace_path__raw_20k_data(
        device
) -> str:
    """

    :param device:
    :return:
    """
    project_dir = os.getenv("MASTER_THESIS_RESULTS_RAW_DATA")
    path = f"datasets/training_traces/Wang_2021/Cable/original_data/20k_d{device}/100avg"

    training_traces_path = os.path.join(
        project_dir,
        path,
    )

    return training_traces_path


def get_training_trace_path__raw_200k_data() -> str:
    """

    :return: Path to training data (cable, 5 devices, single file 200k).
    """
    project_dir = os.getenv("MASTER_THESIS_RESULTS_RAW_DATA")
    path = f"datasets/training_traces/Wang_2021/Cable/data"

    training_traces_path = os.path.join(
        project_dir,
        path,
    )

    return training_traces_path


def get_training_model_file_path(
        database,
        training_model_id,
        additive_noise_method_id,
        denoising_method_id,
        epoch,
        keybyte
) -> str:
    """

    :param database:
    :param training_model_id:
    :param additive_noise_method_id:
    :param denoising_method_id:
    :param epoch:
    :param keybyte:
    :return:
    """
    database = get_db_file_path(database)
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


def insert_data_and_date_to_db__rank_test(
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
        date: int,
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
    :param date: THIS PARAM DIFFERS FROM insert_data_to_db()
    """
    database = get_db_file_path(database)
    con = lite.connect(database)
    con.execute(
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
            date,
        ),
    )
    con.commit()
    con.close()


def copy_rank_test_from_db1_to_db2(database_from, database_to):
    """

    :param database_from: Database to copy rank data from.
    :param database_to: Database to copy rank data to.
    """
    database_from = get_db_file_path(database_from)
    database_to = get_db_file_path(database_to)
    data_from = fetchall_query(database_from, "select * from rank_test;")
    for data in data_from:
        insert_data_and_date_to_db__rank_test(
            database_to,
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7],
            data[8],
            data[9],
            data[10],
            data[11],
            data[12],
            data[13],
        )


def insert_data_to_db__trace_metadata__depth(
        database: str,
        test_dataset_id: Optional[int],
        training_dataset_id: Optional[int],
        environment_id: Optional[int],
        distance: Optional[float],
        device: Optional[int],
        additive_noise_method_id: Optional[int],
        trace_process_id: int,
        data_point_index: int,
        max_val: float,
        min_val: float,
        mean_val: float,
        rms_val: float,
        std_val: float,
        snr_val: float,
):
    """
    Important: either test_dataset_id or training_dataset id is passed.

    :param database:
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param additive_noise_method_id:
    :param trace_process_id:
    :param data_point_index:
    :param max_val:
    :param min_val:
    :param mean_val:
    :param rms_val:
    :param std_val:
    :param snr_val:
    """
    database = get_db_file_path(database)
    con = lite.connect(database)

    con.execute(
        """
        INSERT INTO 
            Trace_Metadata_Depth 
        VALUES(
            NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        );
        """,
        (
            test_dataset_id,
            training_dataset_id,
            environment_id,
            distance,
            device,
            additive_noise_method_id,
            trace_process_id,
            data_point_index,
            max_val,
            min_val,
            mean_val,
            rms_val,
            std_val,
            snr_val,
        ),
    )
    con.commit()
    con.close()


def insert_data_to_db__trace_metadata__width(
        database: str,
        test_dataset_id: Optional[int],
        training_dataset_id: Optional[int],
        environment_id: Optional[int],
        distance: Optional[float],
        device: Optional[int],
        additive_noise_method_id: Optional[int],
        trace_process_id: int,
        trace_index: int,
        max_val: float,
        min_val: float,
        mean_val: float,
        rms_val: float,
        std_val: float,
):
    """
    Important: either test_dataset_id or training_dataset id is passed.

    :param database:
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param additive_noise_method_id:
    :param trace_process_id:
    :param trace_index:
    :param max_val:
    :param min_val:
    :param mean_val:
    :param rms_val:
    :param std_val:
    """
    database = get_db_file_path(database)
    con = lite.connect(database)

    con.execute(
        """
        INSERT INTO 
            Trace_Metadata_Width
        VALUES(
            NULL,?,?,?,?,?,?,?,?,?,?,?,?,?
        );
        """,
        (
            test_dataset_id,
            training_dataset_id,
            environment_id,
            distance,
            device,
            additive_noise_method_id,
            trace_process_id,
            trace_index,
            max_val,
            min_val,
            mean_val,
            rms_val,
            std_val,
        ),
    )
    con.commit()
    con.close()
