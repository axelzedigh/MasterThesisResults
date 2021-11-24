"""Functions to test plot-functions."""
import os
import numpy as np

from matplotlib import pyplot as plt

from utils.db_utils import get_training_trace_path__raw_200k_data, \
    get_test_trace_path
from utils.denoising_utils import wiener_filter
from utils.trace_utils import get_trace_set__processed
from utils.training_utils import cut_trace_set__column_range


def wiener_filter__1():
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
    #plt.ioff()
    plt.figure(figsize=(15, 10))
    plt.plot(trace_set[1])
    filtered_trace = wiener_filter(trace_set[1], 0.001)
    plt.plot(filtered_trace)
    plt.show()


def training_maxmin_sbox_range():
    training_set_path = get_training_trace_path__raw_200k_data()
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

if __name__ == "__main__":
    # wiener_filter__1()
    # training_maxmin_sbox_range()
    test_maxmin_sbox_range()
