import os
import sqlite3 as lite

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt

from utils.db_utils import get_db_absolute_path
from utils.trace_utils import get_training_trace_path


def plot_histogram():
    query = """
    TODO: 
    """
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test = pd.read_sql_query(query, con)
    full_rank_test.fillna("None", inplace=True)
    con.close()
    pass


def plot_dataset_labels_histogram(
        training_dataset_id: int,
):
    """

    :param training_dataset_id:
    """
    training_set_path = get_training_trace_path(training_dataset_id)
    traces_path = os.path.join(
        training_set_path,
        "traces.npy"
    )
    traces = np.load(traces_path)
    labels_path = os.path.join(
        training_set_path,
        "labels.npy"
    )
    labels = np.load(labels_path)

    # reshaped_x_profiling = traces.reshape(
    #     (traces.shape[0], traces.shape[1])
    # )
    # reshaped_y_profiling = to_categorical(labels, num_classes=256)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(labels, bins=256)

    undersample = RandomUnderSampler(sampling_strategy="all")

    X_over, y_over = undersample.fit_resample(traces, labels)
    ax2.hist(y_over, bins=256)

    plt.show()