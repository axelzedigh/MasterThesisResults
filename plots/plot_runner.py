"""Functions to test plot-functions."""
import os

import numpy as np
from matplotlib import pyplot as plt

from plots.history_log_plots import plot_history_log
from utils.db_utils import get_training_trace_path__combined_200k_data, \
    get_test_trace_path
from utils.denoising_utils import wiener_filter
from utils.statistic_utils import \
    maxmin_scaling_of_trace_set__per_trace_fit__max_avg
from utils.trace_utils import get_trace_set__processed
from utils.training_utils import cut_trace_set__column_range


def wiener_filter__1():
    """
    Base wiener filter function.
    """
    trace = get_trace_set__processed(
        database="main.db",
        test_dataset_id=1,
        training_dataset_id=None,
        environment_id=1,
        distance=15,
        device=10,
        trace_process_id=3,
    )
    trace_set = cut_trace_set__column_range(trace)
    plt.figure(figsize=(15, 10))
    plt.plot(trace_set[1])
    filtered_trace = wiener_filter(trace_set[1], 0.001)
    plt.plot(filtered_trace)
    plt.show()


def training_maxmin_sbox_range():
    """Plot maxmin sbox range"""
    training_set_path = get_training_trace_path__combined_200k_data()
    file_path = os.path.join(
        training_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
    )
    trace_set = np.load(file_path)
    plt.plot(trace_set[0])
    plt.plot(trace_set[1])
    plt.plot(trace_set[2])
    plt.plot(trace_set[3])
    plt.show()


def test_maxmin_sbox_range():
    test_dataset_id = 1
    environment_id = 1
    distance = 15
    device = 6
    test_trace_set_path = get_test_trace_path(
        database="main.db",
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
    )
    test_trace_set_file_path = os.path.join(
        test_trace_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
    )
    test_trace_set = np.load(test_trace_set_file_path)
    plt.plot(test_trace_set[0])
    plt.plot(test_trace_set[1])
    plt.plot(test_trace_set[2])
    plt.plot(test_trace_set[3])
    plt.show()


def test_maxmin_scaling_of_trace_set__per_trace_fit__trace_process_8():
    trace_set = get_trace_set__processed(
        "main.db",
        test_dataset_id=1,
        training_dataset_id=None,
        environment_id=1,
        distance=15,
        device=10,
        trace_process_id=2,
    )
    trace_set = maxmin_scaling_of_trace_set__per_trace_fit__max_avg(
        trace_set=trace_set, range_start=204, range_end=314
    )
    trace_set = cut_trace_set__column_range(trace_set, 204, 314)
    fig = plt.figure(figsize=(22, 7))
    ax = fig.gca()
    ax.plot(trace_set[0])
    ax.plot(trace_set[1])
    ax.plot(trace_set[2])
    plt.show()


if __name__ == "__main__":
    # wiener_filter__1()
    # training_maxmin_sbox_range()
    # test_maxmin_sbox_range()
    plot_history_log(
        training_dataset_id=1,
        trace_process_id=3,
        keybyte=0,
        additive_noise_method_id=None,
        denoising_method_id=1,
        save=True,
        show=True,
    )
    # plot_best_additive_noise_methods()
    # plot_all_of_an_additive_noise(
    #     additive_noise_method="Collected"
    # )
    # test_maxmin_scaling_of_trace_set__per_trace_fit__trace_process_8()
    # plot_dataset_labels_histogram(
    #     training_dataset_id=3,
    # )
    # plot_randomized_trace_cut()
