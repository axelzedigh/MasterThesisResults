import os
import sqlite3 as lite
from typing import Optional

import numpy as np

from utils.db_utils import get_test_trace_path, insert_data_to_db__trace_metadata__depth, get_db_file_path, \
    get_training_trace_path__raw_data
from utils.statistic_utils import root_mean_square, signal_to_noise_ratio__sqrt_mean_std


def get_trace_set_metadata__depth(
        database: str,
        test_dataset_id: Optional[int],
        training_dataset_id: Optional[int],
        environment_id: Optional[int],
        distance: Optional[int],
        device: int,
        additive_noise_method_id: Optional[int],
        trace_process_id: int,
) -> np.array:
    """

    :param trace_process_id:
    :param database:
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param additive_noise_method_id:
    :return:
    """
    if type(test_dataset_id) == type(training_dataset_id):
        print("Dataset must be either test dataset or training dataset")
        return

    if trace_process_id == 1:
        raise "This option seems redundant currently."
    else:
        if test_dataset_id:
            trace = get_training_trace_set__processed(
                database=database,
                test_dataset_id=test_dataset_id,
                training_dataset_id=training_dataset_id,
                environment_id=environment_id,
                distance=distance,
                device=device,
                trace_process_id=trace_process_id,
            )
            meta_data = get_trace_metadata__depth__processed(trace)
            return meta_data
        elif training_dataset_id:
            trace = get_training_trace_set__processed(
                database=database,
                test_dataset_id=test_dataset_id,
                training_dataset_id=training_dataset_id,
                environment_id=environment_id,
                distance=distance,
                device=device,
                trace_process_id=trace_process_id,
            )
            meta_data = get_trace_metadata__depth__processed(trace)
            return meta_data


def get_training_trace_set__processed(
        database,
        test_dataset_id,
        training_dataset_id,
        environment_id,
        distance,
        device,
        trace_process_id,
) -> np.array:
    """

    :param database:
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param trace_process_id:
    :return:
    """

    if type(test_dataset_id) == type(training_dataset_id):
        raise "Dataset must be either test dataset or training dataset"

    if test_dataset_id:
        trace_path = get_test_trace_path(
            database,
            test_dataset_id=test_dataset_id,
            environment_id=environment_id,
            distance=distance,
            device=device
        )
    elif training_dataset_id:
        trace_path = get_training_trace_path__raw_data(
            database,
            device=device
        )
    else:
        raise "Both test_dataset_id and training_dataset_id was passed. Not ok!"

    if trace_process_id == 2:
        file_path = os.path.join(trace_path, "traces.npy")
        traces = np.load(file_path)
    elif trace_process_id == 3:
        file_path = os.path.join(trace_path, "nor_traces_maxmin.npy")
        traces = np.load(file_path)
    else:
        print("Something went wrong.")
        return 1

    return traces


def get_trace_metadata__depth__processed(trace_set):
    """
    Input: Trace set (after processing)

    Return value:
        metadata[0]: max value
        metadata[1]: min value
        metadata[2]: mean value
        metadata[3]: rms value
        metadata[4]: std value
        metadata[5]: SNR value (mean^2) / (std^2)
    :param trace_set:
    :return: 400 x 6 np.array.
    """
    meta_data = []
    for index in range(trace_set.shape[1]):
        max_value = np.max(trace_set[:, index], axis=0)
        min_value = np.min(trace_set[:, index], axis=0)
        mean_value = np.mean(trace_set[:, index], axis=0)
        rms_value = root_mean_square(trace_set[:, index])
        std_value = np.std(trace_set[:, index], axis=0)
        snr_value = signal_to_noise_ratio__sqrt_mean_std(mean_value, std_value)
        meta_data.append([max_value, min_value, mean_value, rms_value, std_value, snr_value])

    return np.array(meta_data)


def insert_all_trace_metadata_depth_to_db(database):
    """

    :param database: Database to write to.
    """
    # Remove all previous metadata.
    database = get_db_file_path(database)
    con = lite.connect(database=database)
    con.execute("DELETE FROM Trace_Metadata_Depth;")
    con.commit()
    con.close()

    # Insert all from Wang_2021
    test_dataset_id = 1
    training_dataset_id = None
    environment_id = 1
    distance = 15
    devices = [6, 7, 8, 9, 10]
    additive_noise_method_id = None
    trace_process_ids = [2, 3]

    for device in devices:
        for trace_process_id in trace_process_ids:
            metadata = get_trace_set_metadata__depth(
                database=database,
                test_dataset_id=test_dataset_id,
                training_dataset_id=training_dataset_id,
                environment_id=environment_id,
                distance=distance,
                device=device,
                additive_noise_method_id=additive_noise_method_id,
                trace_process_id=trace_process_id,
            )
            i = 0
            for row in metadata:
                insert_data_to_db__trace_metadata__depth(
                    database=database,
                    test_dataset_id=test_dataset_id,
                    training_dataset_id=training_dataset_id,
                    environment_id=environment_id,
                    distance=distance,
                    device=device,
                    additive_noise_method_id=additive_noise_method_id,
                    trace_process_id=trace_process_id,
                    data_point_index=i,
                    max_val=row[0],
                    min_val=row[1],
                    mean_val=row[2],
                    rms_val=row[3],
                    std_val=row[4],
                    snr_val=row[5],
                )
                i += 1

    # Insert all from Zedigh_2021 (distance 2m)
    test_dataset_id = 2
    training_dataset_id = None
    environment_id = 1
    distance = 2
    devices = [9, 10]
    additive_noise_method_id = None
    trace_process_ids = [2, 3]

    for device in devices:
        for trace_process_id in trace_process_ids:
            metadata = get_trace_set_metadata__depth(
                database=database,
                test_dataset_id=test_dataset_id,
                training_dataset_id=training_dataset_id,
                environment_id=environment_id,
                distance=distance,
                device=device,
                additive_noise_method_id=additive_noise_method_id,
                trace_process_id=trace_process_id,
            )
            i = 0
            for row in metadata:
                insert_data_to_db__trace_metadata__depth(
                    database=database,
                    test_dataset_id=test_dataset_id,
                    training_dataset_id=training_dataset_id,
                    environment_id=environment_id,
                    distance=distance,
                    device=device,
                    additive_noise_method_id=additive_noise_method_id,
                    trace_process_id=trace_process_id,
                    data_point_index=i,
                    max_val=row[0],
                    min_val=row[1],
                    mean_val=row[2],
                    rms_val=row[3],
                    std_val=row[4],
                    snr_val=row[5],
                )
                i += 1

    # Insert all from Zedigh_2021 (distance 5m & 10m)
    test_dataset_id = 2
    training_dataset_id = None
    environment_id = 1
    distances = [5, 10]
    devices = [8, 9, 10]
    additive_noise_method_id = None
    trace_process_ids = [2, 3]

    for distance in distances:
        for device in devices:
            for trace_process_id in trace_process_ids:
                metadata = get_trace_set_metadata__depth(
                    database=database,
                    test_dataset_id=test_dataset_id,
                    training_dataset_id=training_dataset_id,
                    environment_id=environment_id,
                    distance=distance,
                    device=device,
                    additive_noise_method_id=additive_noise_method_id,
                    trace_process_id=trace_process_id,
                )
                i = 0
                for row in metadata:
                    insert_data_to_db__trace_metadata__depth(
                        database=database,
                        test_dataset_id=test_dataset_id,
                        training_dataset_id=training_dataset_id,
                        environment_id=environment_id,
                        distance=distance,
                        device=device,
                        additive_noise_method_id=additive_noise_method_id,
                        trace_process_id=trace_process_id,
                        data_point_index=i,
                        max_val=row[0],
                        min_val=row[1],
                        mean_val=row[2],
                        rms_val=row[3],
                        std_val=row[4],
                        snr_val=row[5],
                    )
                    i += 1

    # Insert all training traces (device 1-5, cable)
    test_dataset_id = None
    training_dataset_id = 1
    environment_id = None
    distance = None
    devices = [1, 2, 3, 4, 5]
    additive_noise_method_id = None
    trace_process_ids = [2, 3]

    for device in devices:
        for trace_process_id in trace_process_ids:
            metadata = get_trace_set_metadata__depth(
                database=database,
                test_dataset_id=test_dataset_id,
                training_dataset_id=training_dataset_id,
                environment_id=environment_id,
                distance=distance,
                device=device,
                additive_noise_method_id=additive_noise_method_id,
                trace_process_id=trace_process_id,
            )
            i = 0
            for row in metadata:
                insert_data_to_db__trace_metadata__depth(
                    database=database,
                    test_dataset_id=test_dataset_id,
                    training_dataset_id=training_dataset_id,
                    environment_id=environment_id,
                    distance=distance,
                    device=device,
                    additive_noise_method_id=additive_noise_method_id,
                    trace_process_id=trace_process_id,
                    data_point_index=i,
                    max_val=row[0],
                    min_val=row[1],
                    mean_val=row[2],
                    rms_val=row[3],
                    std_val=row[4],
                    snr_val=row[5],
                )
                i += 1
