"""
Functions for normalizing previous trace sets to range (0, 1) in the sbox
range (usually 204-314 in 400 data-point trace). Normalizing each trace
separately.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from utils.db_utils import get_training_trace_path__combined_200k_data, \
    get_training_trace_path__combined_100k_data, \
    get_training_trace_path__combined_500k_data
from utils.statistic_utils import maxmin_scaling_of_trace_set__per_trace_fit, \
    maxmin_scaling_of_trace_set__per_trace_fit__max_avg, \
    standardization_of_trace_set__per_trace_fit


def normalize_training_traces_200k():
    """
    Normalize previous combined normalizes trace set to maxmin in sbox-range.
    """
    training_set_path = get_training_trace_path__combined_200k_data()
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


def normalize_training_traces__trace_process_6_and_7(
        training_dataset_id: int = 2,
        trace_process_id: int = 6,
):
    """
    :param training_dataset_id:  Either 2 ot 3.
    :param trace_process_id:  Either 6 or 7.
    """
    start = 130
    end = 240
    if trace_process_id == 6:
        avg_start = 0
        avg_end = 100
        scale = 2.2
    elif trace_process_id == 7:
        avg_start = 130
        avg_end = 240
        scale = 1.8
    else:
        print("Wrong choice of trace process id (6 or 7)!")
        sys.exit(-1)

    if training_dataset_id == 1:
        print("Currently not available (no traces.npy file for this dataset).")
        sys.exit(-1)
    elif training_dataset_id == 2:
        training_dataset_path = get_training_trace_path__combined_100k_data()
    elif training_dataset_id == 3:
        training_dataset_path = get_training_trace_path__combined_500k_data()
    else:
        print("Wrong choice of training dataset id.")
        sys.exit(-1)

    # Load trace set file
    training_trace_set_file_name = os.path.join(
        training_dataset_path, "traces.npy"
    )
    training_trace_set = np.load(training_trace_set_file_name)

    # Normalize trace set.
    training_trace_set = maxmin_scaling_of_trace_set__per_trace_fit__max_avg(
        trace_set=training_trace_set, range_start=start, range_end=end,
        avg_start=avg_start, avg_end=avg_end, scale=scale
    )

    # Save file
    if trace_process_id == 6:
        training_save_path = os.path.join(
            training_dataset_path, "trace_process_6-max_avg(before_sbox).npy"
        )
    elif trace_process_id == 7:
        training_save_path = os.path.join(
            training_dataset_path, "trace_process_7-max_avg(sbox).npy"
        )
    else:
        sys.exit(-1)

    np.save(training_save_path, training_trace_set)

    fig = plt.figure(figsize=(22, 7))
    ax = fig.gca()
    ax.plot(training_trace_set[0])
    ax.plot(training_trace_set[1])
    ax.plot(training_trace_set[2])
    plt.show()


def normalize_training_traces__trace_process_8(
        training_dataset_id: int = 2,
):
    """
    Standardize the training trace set.

    :param training_dataset_id:
    """

    start = 130
    end = 240

    if training_dataset_id == 1:
        print("Currently not available (no traces.npy file for this dataset).")
        sys.exit(-1)
    elif training_dataset_id == 2:
        training_dataset_path = get_training_trace_path__combined_100k_data()
    elif training_dataset_id == 3:
        training_dataset_path = get_training_trace_path__combined_500k_data()
    else:
        print("Wrong choice of training dataset id.")
        sys.exit(-1)

    # Load trace set file
    training_trace_set_file_name = os.path.join(
        training_dataset_path, "traces.npy"
    )
    training_trace_set = np.load(training_trace_set_file_name)

    # Normalize trace set.
    training_trace_set = standardization_of_trace_set__per_trace_fit(
        trace_set=training_trace_set, range_start=start, range_end=end
    )

    # Save file
    training_save_path = os.path.join(
        training_dataset_path, "trace_process_8-standardization_sbox.npy"
    )
    np.save(training_save_path, training_trace_set)

    fig = plt.figure(figsize=(22, 7))
    ax = fig.gca()
    ax.plot(training_trace_set[0])
    ax.plot(training_trace_set[1])
    ax.plot(training_trace_set[2])
    plt.show()


def normalize_training_traces__trace_process_9_10(
        training_dataset_id: int = 2,
        trace_process_id: int = 8,
):
    """
    :param training_dataset_id:  Either 2 ot 3.
    :param trace_process_id:  Either 9 0r 10.
    """
    if trace_process_id == 9:
        start = 0
        end = -1
    elif trace_process_id == 10:
        start = 130
        end = 240
    else:
        print(f"Wrong trace process id ({trace_process_id}), should be 9/10!")
        sys.exit(-1)

    if training_dataset_id == 1:
        print("Currently not available (no traces.npy file for this dataset).")
        sys.exit(-1)
    elif training_dataset_id == 2:
        training_dataset_path = get_training_trace_path__combined_100k_data()
    elif training_dataset_id == 3:
        training_dataset_path = get_training_trace_path__combined_500k_data()
    else:
        print("Wrong choice of training dataset id.")
        sys.exit(-1)

    # Load trace set file
    training_trace_set_file_name = os.path.join(
        training_dataset_path, "traces.npy"
    )
    training_trace_set = np.load(training_trace_set_file_name)

    # Normalize trace set.
    training_trace_set = maxmin_scaling_of_trace_set__per_trace_fit(
        trace_set=training_trace_set, range_start=start, range_end=end,
        scaling_range=(-1, 1)
    )

    # Save file
    if trace_process_id == 9:
        training_save_path = os.path.join(
            training_dataset_path, "trace_process_9-maxmin_[-1_1]_[0_400].npy"
        )
    elif trace_process_id == 10:
        training_save_path = os.path.join(
            training_dataset_path,
            "trace_process_10-maxmin_[-1_1]_[204_314].npy"
        )
    else:
        print("Something wrong with trace process id.")
        sys.exit(-1)

    np.save(training_save_path, training_trace_set)

    fig = plt.figure(figsize=(22, 7))
    ax = fig.gca()
    ax.plot(training_trace_set[0])
    ax.plot(training_trace_set[1])
    ax.plot(training_trace_set[2])
    plt.show()


if __name__ == "__main__":
    # normalize_training_traces_200k()
    training_dataset_ids = [2, 3]
    trace_process_ids = [7]
    # test_dataset_id = 2
    # environment_id = 1
    # distance = 2
    # devices = [9, 10]

    for training_dataset_id in training_dataset_ids:
        for trace_process_id in trace_process_ids:
            normalize_training_traces__trace_process_6_and_7(
                training_dataset_id=training_dataset_id,
                trace_process_id=trace_process_id,
            )

    # normalize_training_traces__trace_process_8(training_dataset_id=2)
    # normalize_training_traces__trace_process_8(training_dataset_id=3)
    #
    #
    # for training_dataset_id in training_dataset_ids:
    #     for trace_process_id in trace_process_ids:
    #         normalize_training_traces__trace_process_9_10(
    #             training_dataset_id=training_dataset_id,
    #             trace_process_id=trace_process_id,
    #         )
    pass
