"""
Functions for normalizing previous trace sets to range (0, 1) in the sbox
range (usually 204-314 in 400 data-point trace). Normalizing each trace
separately.
"""

import os
import numpy as np

from utils.db_utils import get_training_trace_path__raw_200k_data, \
    get_test_trace_path
from utils.statistic_utils import maxmin_scaling_of_trace_set__per_trace_fit


def normalize_training_traces_200k():
    """
    Normalize previous combined normalizes trace set to maxmin in sbox-range.
    """
    training_set_path = get_training_trace_path__raw_200k_data()
    trace_set_file_path = os.path.join(
        # training_set_path, "traces.npy"
        training_set_path, "nor_traces_maxmin.npy"
    )
    trace_set = np.load(trace_set_file_path)

    # Normalize trace set
    trace_set = maxmin_scaling_of_trace_set__per_trace_fit(trace_set, 204, 314)

    # Store new trace set
    save_path = os.path.join(
        training_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
    )
    np.save(save_path, trace_set)


def normalize_test_traces__trace_process_4(
        test_dataset_id: int,
        environment_id: int,
        distance: float,
        device: int,
) -> None:
    """
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    """
    test_trace_set_path = get_test_trace_path(
        database="main.db",
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
    )
    test_trace_set_file_path = os.path.join(
        test_trace_set_path, "traces.npy"
    )
    test_trace_set = np.load(test_trace_set_file_path)

    # Normalize trace set
    test_trace_set = maxmin_scaling_of_trace_set__per_trace_fit(
        test_trace_set, 204, 314
    )

    # Store new trace set
    save_path = os.path.join(
        test_trace_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
    )
    np.save(save_path, test_trace_set)


if __name__ == "__main__":
    # normalize_training_traces_200k()
    test_dataset_id = 2
    environment_id = 1
    distance = 2
    devices = [9, 10]

    for device in devices:
        normalize_test_traces__trace_process_4(
            test_dataset_id=test_dataset_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
        )
