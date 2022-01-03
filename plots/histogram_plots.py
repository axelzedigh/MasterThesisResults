import os
import sqlite3 as lite

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_MPL_STYLE_2_PATH, REPORT_DIR
from utils.db_utils import get_db_absolute_path
from utils.plot_utils import set_size
from utils.trace_utils import get_training_trace_path


def plot_histogram_overview(
        training_dataset_id: int = 3,
        test_dataset_id: int = 1,
        environment_id: int = 1,
        trace_process_id: int = 3,
        device: int = 6,
        distance: float = 15,
        save_path: str = REPORT_DIR,
        file_format: str = "png",
        show: bool = False,
):
    """

    :param training_dataset_id:
    :param test_dataset_id:
    :param environment_id:
    :param trace_process_id:
    :param device:
    :param distance:
    """
    query = """
    select * from rank_test;
    """
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test = pd.read_sql_query(query, con)
    full_rank_test.fillna("None", inplace=True)
    con.close()

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    plt.rcParams.update({
        "ytick.labelsize": "xx-small",
        "xtick.labelsize": "xx-small",
        "axes.titlesize": "x-small",
    })

    # Create 4 x 4 grid
    w, h = set_size(subplots=(4, 3), fraction=1)
    plt.figure(figsize=(w, h))
    plt.subplots_adjust(hspace=0.7, wspace=0.5)
    additive_noise_method_ids = ["None", 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    i = 5
    j = 9
    k = 13
    for additive_noise in additive_noise_method_ids:
        data = full_rank_test.copy()
        data = data[data["test_dataset_id"] == test_dataset_id]
        data = data[data["training_dataset_id"] == training_dataset_id]
        data = data[data["environment_id"] == environment_id]
        data = data[data["distance"] == distance]
        data = data[data["device"] == device]
        data = data[data["additive_noise_method_id"] == additive_noise]
        data = data[data["trace_process_id"] == trace_process_id]
        if not data["termination_point"].empty:
            if additive_noise == "None":
                ax = plt.subplot(4, 4, 1)
            elif additive_noise in [1, 2, 3, 4, 5]:
                ax = plt.subplot(4, 4, i)
                i += 1
            elif additive_noise in [6, 7, 8, 9]:
                ax = plt.subplot(4, 4, j)
                j += 1
            elif additive_noise in [10, 11]:
                ax = plt.subplot(4, 4, k)
                k += 1
            ax.hist(data["termination_point"], bins=10)
            # Labels

            if additive_noise == "None":
                ax.set_ylabel("No noise")
            elif additive_noise == 1:
                ax.set_title("$\sigma$=0.01")
                ax.set_ylabel("GWN")
            elif additive_noise == 3:
                ax.set_title("$\sigma$=0.03")
            elif additive_noise == 4:
                ax.set_title("$\sigma$=0.04")
            elif additive_noise == 5:
                ax.set_title("$\sigma$=0.05")
            elif additive_noise == 6:
                ax.set_ylabel("Collected")
                ax.set_title("scaling=25")
            elif additive_noise == 7:
                ax.set_title("scaling=50")
            elif additive_noise == 8:
                ax.set_title("scaling=75")
            elif additive_noise == 9:
                ax.set_title("scaling=105")
            elif additive_noise == 10:
                ax.set_ylabel("Rayleigh")
                ax.set_title("mode=0.0138")
            elif additive_noise == 11:
                ax.set_title("mode=0.0276")

    plt.tight_layout()
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}",
            f"histogram_view_{device}.{file_format}",
        )
        plt.savefig(path)
    if show:
        plt.show()


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
