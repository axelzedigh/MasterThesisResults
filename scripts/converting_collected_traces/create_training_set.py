"""Functions which creates training traces."""
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.db_utils import get_training_trace_path__raw_20k_data
from utils.statistic_utils import maxmin_scaling_of_trace_set__per_trace_fit


def create_100k_5device_joined_training_set():
    """
    This function creates a training set from Wang_2021 cable traces
    (5 devices, 20k traces for each device).
    """

    # Initialize arrays
    raw_data_path = os.getenv("MASTER_THESIS_RESULTS_RAW_DATA")
    ex_training_path = get_training_trace_path__raw_20k_data(device=1)
    traces = np.load(os.path.join(ex_training_path, "traces.npy"))
    labels = np.load(os.path.join(ex_training_path, "label_0.npy"))
    all_traces = np.empty_like(np.tile(traces, (5, 1)))
    all_labels = np.empty_like(np.tile(labels, 5))
    del traces, labels

    # Save path
    save_path_traces = os.path.join(
        raw_data_path,
        "datasets/training_traces/Zedigh_2021/Cable/100k_5devices_joined",
        "traces.npy"
    )
    save_path_labels = os.path.join(
        raw_data_path,
        "datasets/training_traces/Zedigh_2021/Cable/100k_5devices_joined",
        "labels.npy"
    )
    save_path_maxmin_traces = os.path.join(
        raw_data_path,
        "datasets/training_traces/Zedigh_2021/Cable/100k_5devices_joined",
        "nor_maxmin_traces__130_240.npy"
    )

    i = 0
    for j in range(1, 6):
        training_path = get_training_trace_path__raw_20k_data(device=j)
        traces = np.load(os.path.join(training_path, "traces.npy"))
        labels = np.load(os.path.join(training_path, "label_0.npy"))
        for k in range(len(traces)):
            all_traces[i] = traces[k]
            all_labels[i] = labels[k]
            i += 1

        del traces, labels
    nor_maxmin_traces = maxmin_scaling_of_trace_set__per_trace_fit(
        all_traces, 130, 240
    )

    np.save(save_path_traces, all_traces)
    np.save(save_path_labels, all_labels)
    np.save(save_path_maxmin_traces, nor_maxmin_traces)

    plt.subplot(1, 2, 1)
    plt.plot(all_traces[0][130:240])
    plt.plot(all_traces[1][130:240])
    plt.plot(all_traces[2][130:240])
    plt.subplot(1, 2, 2)
    plt.plot(nor_maxmin_traces[0][130:240])
    plt.plot(nor_maxmin_traces[1][130:240])
    plt.plot(nor_maxmin_traces[2][130:240])
    plt.show()


if __name__ == "__main__":
    create_100k_5device_joined_training_set()
