"""
Functions for normalizing previous trace sets to range (0, 1) in the sbox
range (usually 204-314 in 400 data-point trace). Normalizing each trace
separately.
"""

import os
import numpy as np

from utils.db_utils import get_training_trace_path__raw_200k_data
from utils.statistic_utils import maxmin_scaling_of_trace_set__per_trace_fit


def normalize_training_traces_200k():
    training_set_path = get_training_trace_path__raw_200k_data()
    trace_set_file_path = os.path.join(
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


if __name__ == "__main__":
    normalize_training_traces_200k()
