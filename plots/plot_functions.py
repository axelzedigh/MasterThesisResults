"""Functions for plotting things."""
from typing import Tuple, List
import numpy as np
import sqlite3 as lite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from imblearn.under_sampling import RandomUnderSampler
from tensorflow.python.keras.utils.np_utils import to_categorical

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_4_CUSTOM_LINES, NORD_LIGHT_BLUE, NORD_LIGHT_LIGHT_BLUE, \
    NORD_LIGHT_RED, NORD_LIGHT_ORANGE
from utils.db_utils import get_db_absolute_path, get_test_trace_path
from utils.statistic_utils import root_mean_square
from utils.trace_utils import get_training_trace_path
from utils.training_utils import additive_noise__gaussian, \
    additive_noise__collected_noise__office_corridor, additive_noise__rayleigh


def plot_trace_metadata_depth__one(test_dataset_id, distance, device,
                                   trace_process_id):
    """
    Legacy...

    :param test_dataset_id:
    :param distance:
    :param device:
    :param trace_process_id:
    :return:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES

    query = f"""
    select
        *
    from 
        trace_metadata_depth
    where
        test_dataset_id = {test_dataset_id}
        AND distance = {distance}
        AND trace_process_id = {trace_process_id}
        AND device = {device}
    ;
    """
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    data = pd.read_sql_query(query, con)
    data[204:314].plot(x="data_point_index", y="mean_val", figsize=(10, 8))
    mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
    mean_rms = root_mean_square(data[204:314]["rms_val"])
    mean_std = np.mean(data[204:314]["std_val"], axis=0)
    mean_snr = np.mean(data[204:314]["snr_val"], axis=0)
    np.mean(data[204:314]["min_val"], axis=0)
    plt.title(
        f"""
        Dataset {test_dataset_id}, Distance {distance}m, Device {device}, Processing_id: {trace_process_id}\n
        Mean: {round(mean_mean, 4)}, RMS: {round(mean_rms, 4)}, Std: {round(mean_std, 5)}, SNR: {round(mean_snr, 2)}
        """
    )
    plt.axhline(mean_mean, c=NORD_LIGHT_RED)
    plt.axhline(mean_rms, c=NORD_LIGHT_ORANGE)
    labels = ["Mean", "Mean Mean", "Mean RMS"]
    plt.legend(custom_lines, labels)
    plt.show()
    con.close()
    return


def plot_trace_metadata_depth__big_plots():
    """
    Plot all trace metadata depths.
    """
    set1 = (1, [6, 7, 8, 9, 10], 15)
    set2 = (2, [9, 10], 2)
    set3 = (2, [8, 9, 10], 5)
    set4 = (2, [8, 9, 10], 10)
    set5 = (2, [8, 9, 10], 15)
    sets = [set1, set2, set3, set4, set5]
    for subset in sets:
        for device in subset[1]:
            test_dataset_id = subset[0]
            distance = subset[2]
            plot_trace_metadata_depth__one(test_dataset_id, distance, device, 2)


def plot_test_trace_metadata_depth__mean(
        test_dataset_id: int = 1,
        distance: float = 15,
        devices: List[int] = (6, 7, 8, 9, 10),
        trace_process_id: int = 2,
        environment_id: int = 1,
):
    """
    :param test_dataset_id:
    :param distance:
    :param devices:
    :param trace_process_id:
    :param environment_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(hspace=0.5)
    i = 1
    for device in devices:
        ax = plt.subplot(1, 5, i)
        if trace_process_id == 2:
            ax.set_ylim(0.0015, 0.014)
        elif trace_process_id == 3 or 4:
            ax.set_ylim(0, 1)
        data = raw_data.copy()
        data = data[data["distance"] == distance]
        data = data[data["device"] == device]
        data = data[data["environment_id"] == environment_id]
        data = data[data["trace_process_id"] == trace_process_id]
        data = data[data["test_dataset_id"] == test_dataset_id]
        data[204:314].plot(x="data_point_index", y="mean_val", ax=ax)
        top_value = np.max(data[204:314]["mean_val"], axis=0)
        bottom_value = np.min(data[204:314]["mean_val"], axis=0)
        dyn_range = top_value - bottom_value
        scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
        mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
        # mean_rms = root_mean_square(data[204:314]["rms_val"])
        mean_std = round(root_mean_square(data[204:314]["std_val"]), 5)
        mean_snr = round(np.mean(data[204:314]["snr_val"], axis=0), 1)
        ax.axhline(top_value, c=NORD_LIGHT_BLUE)
        ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
        ax.axhline(mean_mean, c=NORD_LIGHT_RED)
        if round(top_value, 5) < 0.012:
            ax.text(
                0.02, 0.97,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        else:
            ax.text(
                0.05, 0.1,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        i += 1
        ax.set_title(
            f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
    plt.show()


def plot_test_trace_metadata_depth__rms(
        test_dataset_id: int = 1,
        distance: float = 15,
        devices: List[int] = (6, 7, 8, 9, 10),
        trace_process_id: int = 2,
        environment_id: int = 1,
):
    """
    :param test_dataset_id:
    :param distance:
    :param devices:
    :param trace_process_id:
    :param environment_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(hspace=0.5)
    i = 1
    for device in devices:
        ax = plt.subplot(1, 5, i)
        if trace_process_id == 2:
            ax.set_ylim(0.0015, 0.014)
        elif trace_process_id == 3 or 4:
            ax.set_ylim(0, 1)
        data = raw_data.copy()
        data = data[data["distance"] == distance]
        data = data[data["device"] == device]
        data = data[data["environment_id"] == environment_id]
        data = data[data["trace_process_id"] == trace_process_id]
        data = data[data["test_dataset_id"] == test_dataset_id]
        data[204:314].plot(x="data_point_index", y="rms_val", ax=ax)
        top_value = np.max(data[204:314]["rms_val"], axis=0)
        bottom_value = np.min(data[204:314]["rms_val"], axis=0)
        dyn_range = top_value - bottom_value
        scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
        # mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
        mean_rms = root_mean_square(data[204:314]["rms_val"])
        mean_std = round(root_mean_square(data[204:314]["std_val"]), 5)
        mean_snr = round(np.mean(data[204:314]["snr_val"], axis=0), 1)
        ax.axhline(top_value, c=NORD_LIGHT_BLUE)
        ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
        ax.axhline(mean_rms, c=NORD_LIGHT_ORANGE)
        if round(top_value, 5) < 0.012:
            ax.text(
                0.02, 0.97,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        else:
            ax.text(
                0.05, 0.1,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        i += 1
        ax.set_title(
            f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
    plt.show()


def plot_test_trace_metadata_depth__std(
        test_dataset_id: int = 1,
        distance: float = 15,
        devices: List[int] = (6, 7, 8, 9, 10),
        trace_process_id: int = 2,
        environment_id: int = 1,
):
    """
    :param test_dataset_id:
    :param distance:
    :param devices:
    :param trace_process_id:
    :param environment_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(hspace=0.5)
    i = 1
    for device in devices:
        ax = plt.subplot(1, 5, i)
        # if trace_process_id == 2:
        #     ax.set_ylim(0.0015, 0.014)
        # elif trace_process_id == 3 or 4:
        #     ax.set_ylim(0, 1)
        data = raw_data.copy()
        data = data[data["distance"] == distance]
        data = data[data["device"] == device]
        data = data[data["environment_id"] == environment_id]
        data = data[data["trace_process_id"] == trace_process_id]
        data = data[data["test_dataset_id"] == test_dataset_id]
        data[204:314].plot(x="data_point_index", y="std_val", ax=ax)
        top_value = np.max(data[204:314]["std_val"], axis=0)
        bottom_value = np.min(data[204:314]["std_val"], axis=0)
        dyn_range = top_value - bottom_value
        scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
        # mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
        # mean_rms = root_mean_square(data[204:314]["rms_val"])
        mean_std = round(root_mean_square(data[204:314]["std_val"]), 5)
        mean_snr = round(np.mean(data[204:314]["snr_val"], axis=0), 1)
        ax.axhline(top_value, c=NORD_LIGHT_BLUE)
        ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
        # ax.axhline(mean_mean, c="r")
        # ax.axhline(mean_rms, c="g")
        ax.axhline(mean_std, c=NORD_LIGHT_RED)
        if round(top_value, 5) < 0.012:
            ax.text(
                0.02, 0.97,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        else:
            ax.text(
                0.05, 0.1,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        i += 1
        ax.set_title(
            f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
    plt.show()


def plot_test_trace_metadata_depth__snr(
        test_dataset_id: int = 1,
        distance: float = 15,
        devices: List[int] = (6, 7, 8, 9, 10),
        trace_process_id: int = 2,
        environment_id: int = 1,
):
    """
    :param test_dataset_id:
    :param distance:
    :param devices:
    :param trace_process_id:
    :param environment_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(hspace=0.5)
    i = 1
    for device in devices:
        ax = plt.subplot(1, 5, i)
        # if trace_process_id == 2:
        #     ax.set_ylim(0.0015, 0.014)
        # elif trace_process_id == 3 or 4:
        #     ax.set_ylim(0, 1)
        data = raw_data.copy()
        data = data[data["distance"] == distance]
        data = data[data["device"] == device]
        data = data[data["environment_id"] == environment_id]
        data = data[data["trace_process_id"] == trace_process_id]
        data = data[data["test_dataset_id"] == test_dataset_id]
        data[204:314].plot(x="data_point_index", y="snr_val", ax=ax)
        top_value = np.max(data[204:314]["snr_val"], axis=0)
        bottom_value = np.min(data[204:314]["snr_val"], axis=0)
        dyn_range = top_value - bottom_value
        scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
        # mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
        # mean_rms = root_mean_square(data[204:314]["rms_val"])
        mean_std = round(root_mean_square(data[204:314]["std_val"]), 5)
        mean_snr = round(np.mean(data[204:314]["snr_val"], axis=0), 1)
        ax.axhline(top_value, c=NORD_LIGHT_BLUE)
        ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
        # ax.axhline(mean_mean, c="r")
        # ax.axhline(mean_rms, c="g")
        ax.axhline(mean_snr, c=NORD_LIGHT_RED)
        if round(top_value, 5) < 0.012:
            ax.text(
                0.02, 0.97,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        else:
            ax.text(
                0.05, 0.1,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        i += 1
        ax.set_title(
            f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
    plt.show()


def plot_test_trace_metadata_depth__std__full_trace(
        test_dataset_id: int = 1,
        distance: float = 15,
        device: int = 6,
        trace_process_id: int = 2,
        environment_id: int = 1,
):
    """
    :param test_dataset_id:
    :param distance:
    :param device:
    :param trace_process_id:
    :param environment_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    fig = plt.figure(figsize=(20, 5))
    ax = fig.gca()
    # if trace_process_id == 2:
    #     ax.set_ylim(0.0015, 0.014)
    # elif trace_process_id == 3 or 4:
    #     ax.set_ylim(0, 1)
    data = raw_data.copy()
    data = data[data["distance"] == distance]
    data = data[data["device"] == device]
    data = data[data["environment_id"] == environment_id]
    data = data[data["trace_process_id"] == trace_process_id]
    data = data[data["test_dataset_id"] == test_dataset_id]
    data.plot(x="data_point_index", y="std_val", ax=ax)
    top_value = np.max(data["std_val"], axis=0)
    bottom_value = np.min(data["std_val"], axis=0)
    dyn_range = top_value - bottom_value
    scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
    # mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
    # mean_rms = root_mean_square(data[204:314]["rms_val"])
    mean_std = round(root_mean_square(data["std_val"]), 5)
    mean_snr = round(np.mean(data["snr_val"], axis=0), 1)
    ax.axhline(top_value, c=NORD_LIGHT_BLUE)
    ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
    # ax.axhline(mean_mean, c="r")
    # ax.axhline(mean_rms, c="g")
    ax.axvline(x=204, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    ax.axvline(x=314, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    ax.axhline(mean_std, c=NORD_LIGHT_RED)
    if round(top_value, 5) < 0.012:
        ax.text(
            0.02, 0.97,
            f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
    else:
        ax.text(
            0.05, 0.1,
            f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
    ax.set_title(
        f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
    )
    ax.get_legend().remove()
    ax.set_xlabel("")
    plt.show()


def plot_test_trace_metadata_depth__snr__full_trace(
        test_dataset_id: int = 1,
        distance: float = 15,
        device: int = 6,
        trace_process_id: int = 2,
        environment_id: int = 1,
):
    """
    :param test_dataset_id:
    :param distance:
    :param device:
    :param trace_process_id:
    :param environment_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    fig = plt.figure(figsize=(20, 5))
    ax = fig.gca()
    # if trace_process_id == 2:
    #     ax.set_ylim(0.0015, 0.014)
    # elif trace_process_id == 3 or 4:
    #     ax.set_ylim(0, 1)
    data = raw_data.copy()
    data = data[data["distance"] == distance]
    data = data[data["device"] == device]
    data = data[data["environment_id"] == environment_id]
    data = data[data["trace_process_id"] == trace_process_id]
    data = data[data["test_dataset_id"] == test_dataset_id]
    data.plot(x="data_point_index", y="snr_val", ax=ax)
    top_value = np.max(data["snr_val"], axis=0)
    bottom_value = np.min(data["snr_val"], axis=0)
    dyn_range = top_value - bottom_value
    scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
    # mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
    # mean_rms = root_mean_square(data[204:314]["rms_val"])
    mean_std = round(root_mean_square(data["std_val"]), 5)
    mean_snr = round(np.mean(data["snr_val"], axis=0), 1)
    ax.axhline(top_value, c=NORD_LIGHT_BLUE)
    ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
    # ax.axhline(mean_mean, c="r")
    # ax.axhline(mean_rms, c="g")
    ax.axvline(x=204, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    ax.axvline(x=314, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    ax.axhline(mean_snr, c=NORD_LIGHT_RED)
    if round(top_value, 5) < 0.012:
        ax.text(
            0.02, 0.97,
            f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
    else:
        ax.text(
            0.05, 0.1,
            f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
    ax.set_title(
        f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
    )
    ax.get_legend().remove()
    ax.set_xlabel("")
    plt.show()


def plot_training_trace_metadata_depth__several(range_start=204, range_end=314):
    """

    :param range_start:
    :param range_end:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    # This function is based on legacy data
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con)
    devices = [1, 2, 3, 4, 5]
    trace_process_id = 2

    i = 1
    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(hspace=0.5)
    # plt.suptitle("Daily closing prices", fontsize=18, y=0.95)

    for device in devices:
        ax = plt.subplot(1, 5, i)
        if trace_process_id == 2:
            ax.set_ylim(0, 0.1)
        elif trace_process_id == 3:
            ax.set_ylim(0, 1)
        data = raw_data.copy()
        data = data[data["device"] == device]
        data = data[data["trace_process_id"] == trace_process_id]
        data = data[data["training_dataset_id"] == 1]
        data[range_start:range_end].plot(x="data_point_index", y="mean_val",
                                         ax=ax)

        top_value = np.max(data[range_start:range_end]["mean_val"], axis=0)
        bottom_value = np.min(data[range_start:range_end]["mean_val"], axis=0)
        dyn_range = top_value - bottom_value
        scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
        mean_mean = np.mean(data[range_start:range_end]["mean_val"], axis=0)
        mean_rms = np.mean(data[range_start:range_end]["rms_val"], axis=0)
        mean_std = round(
            np.mean(data[range_start:range_end]["std_val"], axis=0), 5)
        mean_snr = round(
            np.mean(data[range_start:range_end]["snr_val"], axis=0), 1)

        ax.axhline(top_value, c=NORD_LIGHT_BLUE)
        ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
        ax.axhline(mean_mean, c=NORD_LIGHT_RED)
        ax.axhline(mean_rms, c=NORD_LIGHT_ORANGE)
        if round(top_value, 5) < 0.012:
            ax.text(
                0.02, 0.97,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        else:
            ax.text(
                0.05, 0.1,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes
            )
        i += 1
        ax.set_title(
            f"""Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
    plt.show()

    con.close()


def plot_training_trace_metadata_depth__training_set_used():
    """
    Plot all training trace metadata depth.
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
    con2.close()
    trace_process_id = 3
    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(hspace=0.5)
    # plt.suptitle("Daily closing prices", fontsize=18, y=0.95)
    ax = plt.subplot(1, 5, 1)
    if trace_process_id == 2:
        ax.set_ylim(0, 0.1)
    elif trace_process_id == 3:
        ax.set_ylim(0, 1)

    data = raw_data.copy()
    data = data[data["trace_process_id"] == trace_process_id]
    data = data[data["training_dataset_id"] == 1]
    data[204:314].plot(x="data_point_index", y="mean_val", ax=ax)

    top_value = np.max(data[204:314]["mean_val"], axis=0)
    bottom_value = np.min(data[204:314]["mean_val"], axis=0)
    dyn_range = top_value - bottom_value
    scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
    mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
    mean_rms = np.mean(data[204:314]["rms_val"], axis=0)
    mean_std = round(np.mean(data[204:314]["std_val"], axis=0), 5)
    mean_snr = round(np.mean(data[204:314]["snr_val"], axis=0), 1)

    ax.axhline(top_value, c=NORD_LIGHT_BLUE)
    ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
    ax.axhline(mean_mean, c=NORD_LIGHT_RED)
    ax.axhline(mean_rms, c=NORD_LIGHT_ORANGE)

    if round(top_value, 5) < 0.012:
        ax.text(
            0.02, 0.97,
            f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
    else:
        ax.text(
            0.05, 0.1,
            f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
    ax.set_title(
        f"""Std: {mean_std}, SNR: {mean_snr}"""
    )
    ax.get_legend().remove()
    ax.set_xlabel("")
    plt.show()


def plot_all_trace_metadata_depth():
    """
    Plots all training depth metadata.
    """
    plot_training_trace_metadata_depth__training_set_used()
    plot_training_trace_metadata_depth__several(130, 240)


def plot_best_additive_noise_methods(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        trace_process_id: int = 3,
        gaussian_value: float = 0.04,
        collected_value: float = 25,
        rayleigh_value: float = 0.0138,
        save: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param training_dataset:
    :param trace_process_id: The parameter 1 value.
    :param gaussian_value:  The parameter 1 value.
    :param collected_value: The parameter 1 value.
    :param rayleigh_value:  The parameter 1 value.
    :param save: To save figure or not.
    """

    # MPL styling
    sns.set_theme(rc={"figure.figsize": (15, 7)})
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    fig, axs = plt.subplots(1, 2)

    database = get_db_absolute_path("main.db")
    con = lite.connect(database)

    query1 = f"""
    select
        device, 
        epoch, 
        additive_noise_method,
        additive_noise_method_parameter_1, 
        additive_noise_method_parameter_1_value, 
        additive_noise_method_parameter_2, 
        additive_noise_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        trace_process_id = {trace_process_id}
        AND training_dataset = '{training_dataset}'
        AND environment = 'office_corridor'
        AND test_dataset = 'Wang_2021'
        AND epoch = 65 
        AND distance = 15
        AND denoising_method IS NULL
        AND (
            additive_noise_method_parameter_1_value = {gaussian_value}
            OR additive_noise_method_parameter_1_value IS NULL
            OR additive_noise_method_parameter_1_value = {collected_value}
            OR additive_noise_method_parameter_1_value = {rayleigh_value}
        )
    order by
        additive_noise_method
        ;
    """

    query2 = f"""
    select
        device, 
        epoch, 
        additive_noise_method,
        additive_noise_method_parameter_1, 
        additive_noise_method_parameter_1_value, 
        additive_noise_method_parameter_2, 
        additive_noise_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        trace_process_id = {trace_process_id}
        AND training_dataset = '{training_dataset}'
        AND environment = 'office_corridor'
        AND test_dataset = 'Zedigh_2021'
        AND epoch = 65 
        AND distance = 15
        AND denoising_method IS NULL
        AND (
            additive_noise_method_parameter_1_value = {gaussian_value}
            OR additive_noise_method_parameter_1_value IS NULL
            OR additive_noise_method_parameter_1_value = {collected_value}
            OR additive_noise_method_parameter_1_value = {rayleigh_value}
        )
    order by
        additive_noise_method
        ;
    """

    full_rank_test__wang = pd.read_sql_query(query1, con)
    full_rank_test__zedigh = pd.read_sql_query(query2, con)
    con.close()
    full_rank_test__wang.fillna("None", inplace=True)
    full_rank_test__wang.rename(columns={
        "additive_noise_method_parameter_1_value": "Additive parameter 1"},
        inplace=True)
    full_rank_test__zedigh.fillna("None", inplace=True)
    full_rank_test__zedigh.rename(columns={
        "additive_noise_method_parameter_1_value": "Additive parameter 1"},
        inplace=True)
    ylim_bottom = 0
    ylim_top = 1600
    labels = [
        "None",
        f"Collected: Scaling factor = {collected_value}",
        f"Gaussian: ∂ = {gaussian_value}",
        f"Rayleigh: Mode = {rayleigh_value}"
    ]
    sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["additive_noise_method"],
        ax=axs[0],
    )
    sns.barplot(
        x=full_rank_test__zedigh["device"],
        y=full_rank_test__zedigh["termination_point"],
        hue=full_rank_test__zedigh["additive_noise_method"],
        ax=axs[1],
    )
    plt.suptitle(
        f"Best additive noise, 15m, trace process {trace_process_id}",
        fontsize=18,
        y=0.95
    )
    axs[0].set_ylim(ylim_bottom, ylim_top)
    axs[0].set_ylabel("Termination point")
    axs[0].set_xlabel("Device")
    axs[0].text(x=-0.2, y=(ylim_top - 100), s="Wang 2021", fontsize=16)
    axs[1].set_ylim(ylim_bottom, ylim_top)
    axs[1].set_ylabel("Termination point")
    axs[1].set_xlabel("Device")
    axs[1].text(x=-0.2, y=(ylim_top - 100), s="Zedigh 2021", fontsize=16)
    plt.tight_layout()
    axs[0].legend(custom_lines, labels)
    axs[1].legend(custom_lines, labels)
    if save:
        plt.savefig("../docs/figs/Additive_noise_comparison_Wang_Zedigh.png")
    plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_all_of_an_additive_noise(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        additive_noise_method: str = "Gaussian",
        trace_process_id: int = 3,
        epoch: int = 65,
        distance: float = 15,
        environment: str = "office_corridor",
        save: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param training_dataset:
    :param additive_noise_method:
    :param trace_process_id:
    :param epoch:
    :param distance:
    :param environment:
    :param save:
    """
    # MPL styling
    sns.set_theme(rc={"figure.figsize": (15, 7)})
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    fig, axs = plt.subplots(1, 2)

    query_wang = f"""
    select
        environment,
        device, 
        epoch, 
        additive_noise_method,
        additive_noise_method_parameter_1, 
        additive_noise_method_parameter_1_value, 
        additive_noise_method_parameter_2, 
        additive_noise_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        trace_process_id = {trace_process_id}
        AND training_dataset = '{training_dataset}'
        AND test_dataset = 'Wang_2021'
        AND environment = 'office_corridor'
        AND epoch = {epoch} 
        AND distance = {distance}
        AND denoising_method IS NULL
        AND (additive_noise_method IS NULL 
            OR additive_noise_method = '{additive_noise_method}')
    order by 
        additive_noise_method_parameter_1_value
        ;
    """

    query_zedigh = f"""
    select
        environment,
        device, 
        epoch, 
        additive_noise_method,
        additive_noise_method_parameter_1, 
        additive_noise_method_parameter_1_value, 
        additive_noise_method_parameter_2, 
        additive_noise_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        trace_process_id = {trace_process_id}
        AND training_dataset = '{training_dataset}'
        AND test_dataset = 'Zedigh_2021'
        AND environment = '{environment}'
        AND epoch = {epoch} 
        AND distance = {distance}
        AND denoising_method IS NULL
        AND (additive_noise_method IS NULL 
        OR additive_noise_method = '{additive_noise_method}')
    order by 
        additive_noise_method_parameter_1_value
        ;
    """
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test__wang = pd.read_sql_query(query_wang, con)
    full_rank_test__zedigh = pd.read_sql_query(query_zedigh, con)
    full_rank_test__wang.fillna("None", inplace=True)
    full_rank_test__wang.rename(columns={
        "additive_noise_method_parameter_1_value": "Additive parameter 1"},
        inplace=True)
    full_rank_test__zedigh.fillna("None", inplace=True)
    full_rank_test__zedigh.rename(columns={
        "additive_noise_method_parameter_1_value": "Additive parameter 1"},
        inplace=True)
    con.close()
    ylim_bottom = 0
    ylim_top = 1600
    sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["Additive parameter 1"],
        ax=axs[0]
    )
    sns.barplot(
        x=full_rank_test__zedigh["device"],
        y=full_rank_test__zedigh["termination_point"],
        hue=full_rank_test__zedigh["Additive parameter 1"],
        ax=axs[1]
    )
    plt.suptitle(
        f"""{additive_noise_method.capitalize()} additive noise, {distance}m, trace process {trace_process_id}""",
        fontsize=18,
        y=0.95
    )
    axs[0].set_ylim(ylim_bottom, ylim_top)
    axs[0].set_ylabel("Termination point")
    axs[0].set_xlabel("Device")
    axs[0].text(
        x=-0.2,
        y=(ylim_top - 200),
        s=f"Wang 2021\n{environment.replace('_', ' ').capitalize()}",
        fontsize=16
    )
    axs[1].set_ylim(ylim_bottom, ylim_top)
    axs[1].set_ylabel("Termination point")
    axs[1].set_xlabel("Device")
    axs[1].text(
        x=-0.2,
        y=(ylim_top - 200),
        s=f"Zedigh 2021\n{environment.replace('_', ' ').capitalize()}",
        fontsize=16
    )
    plt.tight_layout()
    plt.tight_layout()
    if save:
        plt.savefig(
            f"../docs/figs/{additive_noise_method.replace(' ', '_')}_comparison.png")
    plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_all_of_denoising(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        denoising_method: str = "Moving Average Filter",
        trace_process_id: int = 3,
        epoch: int = 65,
        distance: float = 15,
        save: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param training_dataset:
    :param denoising_method:
    :param trace_process_id:
    :param epoch:
    :param distance:
    :param save:
    """

    # MPL styling
    sns.set(rc={"figure.figsize": (15, 7)})
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    fig, axs = plt.subplots(1, 2)

    query_wang = f"""
    select
        device, 
        epoch, 
        denoising_method,
        denoising_method_parameter_1, 
        denoising_method_parameter_1_value, 
        denoising_method_parameter_2, 
        denoising_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        test_dataset = 'Wang_2021'
        AND training_dataset = '{training_dataset}'
        AND epoch = {epoch}
        AND distance = {distance}
        AND additive_noise_method IS NULL
        AND (denoising_method IS NULL 
        OR denoising_method = '{denoising_method}')
    order by 
        denoising_method_parameter_1_value;
    """

    query_zedigh = f"""
    select
        device, 
        epoch, 
        denoising_method,
        denoising_method_parameter_1, 
        denoising_method_parameter_1_value, 
        denoising_method_parameter_2, 
        denoising_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        test_dataset = 'Zedigh_2021'
        AND training_dataset = '{training_dataset}'
        AND epoch = {epoch}
        AND distance = {distance}
        AND additive_noise_method IS NULL
        AND (denoising_method IS NULL 
        OR denoising_method = '{denoising_method}')
    order by 
        denoising_method_parameter_1_value;
    """
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test__wang = pd.read_sql_query(query_wang, con)
    full_rank_test__zedigh = pd.read_sql_query(query_zedigh, con)
    full_rank_test__wang.fillna("None", inplace=True)
    full_rank_test__wang.rename(columns={
        "denoising_method_parameter_1_value": "Denoising parameter 1"},
        inplace=True)
    full_rank_test__zedigh.fillna("None", inplace=True)
    full_rank_test__zedigh.rename(columns={
        "denoising_method_parameter_1_value": "Denoising parameter 1"},
        inplace=True)
    con.close()
    ylim_bottom = 0
    ylim_top = 1600
    sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["Denoising parameter 1"],
        ax=axs[0]
    )
    sns.barplot(
        x=full_rank_test__zedigh["device"],
        y=full_rank_test__zedigh["termination_point"],
        hue=full_rank_test__zedigh["Denoising parameter 1"],
        ax=axs[1]
    )
    plt.suptitle(
        f"{denoising_method.capitalize()}, {distance}m, trace process {trace_process_id}",
        fontsize=18,
        y=0.95
    )
    axs[0].set_ylim(ylim_bottom, ylim_top)
    axs[0].set_ylabel("Termination point")
    axs[0].set_xlabel("Device")
    axs[0].text(x=-0.2, y=(ylim_top - 100), s="Wang 2021", fontsize=16)
    axs[1].set_ylim(ylim_bottom, ylim_top)
    axs[1].set_ylabel("Termination point")
    axs[1].set_xlabel("Device")
    axs[1].text(x=-0.2, y=(ylim_top - 100), s="Zedigh 2021", fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(
            f"../docs/figs/{denoising_method.replace(' ', '_')}_comparison.png")
    plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_epoch_comparison(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        test_dataset: str = "Wang_2021",
        device: int = 6,
        distance: float = 15,
        additive_noise_method: str = "Gaussian",
        additive_noise_method_parameter_1_value: float = 0.04,
        save: bool = False,
) -> pd.DataFrame:
    """
    :param training_dataset:
    :param test_dataset:
    :param device:
    :param distance:
    :param additive_noise_method:
    :param additive_noise_method_parameter_1_value:
    :param save:
    :return: Pandas DataFrame
    """

    # MPL styling
    sns.set(rc={"figure.figsize": (15, 7)})
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    query = f"""
    select
        device, 
        epoch, 
        additive_noise_method,
        additive_noise_method_parameter_1, 
        additive_noise_method_parameter_1_value, 
        additive_noise_method_parameter_2, 
        additive_noise_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        test_dataset = '{test_dataset}'
        AND training_dataset = '{training_dataset}'
        AND device = {device}
        AND distance = {distance}
        AND denoising_method IS NULL
        AND additive_noise_method_parameter_1_value 
        = {additive_noise_method_parameter_1_value}
        AND additive_noise_method = '{additive_noise_method}'
    order by 
        epoch;
    """

    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test = pd.read_sql_query(query, con)
    full_rank_test.fillna("None", inplace=True)
    full_rank_test.rename(columns={
        "additive_noise_method_parameter_1_value": "Additive parameter 1"},
        inplace=True)
    con.close()
    ylim_bottom = 100
    ylim_top = 800
    sns.barplot(x=full_rank_test["epoch"],
                y=full_rank_test["termination_point"],
                hue=full_rank_test["Additive parameter 1"],
                capsize=0.3, )
    plt.ylim(ylim_bottom, ylim_top)
    plt.tight_layout()
    if save:
        plt.savefig(
            f"../docs/figs/Epoch_{additive_noise_method}_"
            f"{additive_noise_method_parameter_1_value}_comparison_"
            f"{test_dataset}.png"
        )
    plt.show()
    return full_rank_test


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


def plot_additive_noise_comparison_all(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        trace_process_id: int = 3,
        environment: str = "office_corridor",
        gaussian_value: float = 0.04,
        collected_value: float = 25,
        rayleigh_value: float = 0.0138,
        save: bool = False,
) -> pd.DataFrame:
    """
    :param training_dataset:
    :param trace_process_id:
    :param environment:
    :param gaussian_value:
    :param collected_value:
    :param rayleigh_value:
    :param save:
    :return: Pandas DataFrame.
    """

    # MPL styling
    sns.set(rc={"figure.figsize": (15, 7)})
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES

    query = f"""
    select
        environment,
        device, 
        epoch, 
        additive_noise_method,
        additive_noise_method_parameter_1, 
        additive_noise_method_parameter_1_value, 
        additive_noise_method_parameter_2, 
        additive_noise_method_parameter_2_value, 
        termination_point
    from
        full_rank_test
    where
        trace_process_id = {trace_process_id}
        AND training_dataset = '{training_dataset}'
        AND environment = '{environment}'
        AND epoch = 65 
        AND denoising_method IS NULL
        AND (
            additive_noise_method_parameter_1_value = {gaussian_value}
            OR additive_noise_method_parameter_1_value IS NULL
            OR additive_noise_method_parameter_1_value = {collected_value}
            OR additive_noise_method_parameter_1_value = {rayleigh_value}
        )
    order by
        additive_noise_method
        ;
    """
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test = pd.read_sql_query(query, con)
    full_rank_test.fillna("None", inplace=True)
    full_rank_test.rename(columns={
        "additive_noise_method_parameter_1_value": "Additive parameter 1"},
        inplace=True)
    ylim_bottom = 100
    ylim_top = 800
    labels = [
        "None",
        f"Collected - factor={collected_value}",
        f"Gaussian - ∂={gaussian_value}",
        f"Rayleigh - mode={rayleigh_value}"
    ]
    sns.barplot(x=full_rank_test["additive_noise_method"],
                y=full_rank_test["termination_point"])
    plt.ylim(ylim_bottom, ylim_top)
    plt.suptitle(
        f"""
        Additive noise comparison (trace process {trace_process_id}), 
        {environment.replace('_', ' ').capitalize()}.
        """,
        fontsize=18,
        y=0.95
    )
    plt.tight_layout()
    plt.legend(custom_lines, labels)
    if save:
        plt.savefig("../docs/figs/Additive_noise_comparison_ALL.png")
    plt.show()
    return full_rank_test


def plot_example_test_traces_with_max_min(
        test_dataset_id: int = 1,
        environment_id: int = 1,
        distance: float = 15,
        device: int = 6,
        trace_processing_id: int = 2,
        trace_index: int = 1,
):
    """
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param trace_processing_id:
    :param trace_index: The index of the trace in the trace set.
    """

    # MPL styling
    fig = plt.figure(figsize=(22, 6))
    ax = fig.gca()
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    # Load a trace set
    trace_set_path = get_test_trace_path(
        database="main.db",
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
    )
    if trace_processing_id == 2:
        file_path = os.path.join(trace_set_path, "traces.npy")
    elif trace_processing_id == 3:
        file_path = os.path.join(trace_set_path, "nor_traces_maxmin.npy")
    elif trace_processing_id == 4 or 5:
        file_path = os.path.join(
            trace_set_path,
            "nor_traces_maxmin__sbox_range_204_314.npy"
        )
    else:
        raise "Incorrect trace_processing_id"
    trace_set = np.load(file_path)

    # Max/min
    if trace_processing_id == 2 or 3:
        ex_trace = trace_set[trace_index]
        ex_max = float(max(ex_trace))
        ex_min = float(min(ex_trace))
        ex_scaling = round(float(1 / (ex_max - ex_min)), 2)
        arg_max = np.argmax(ex_trace)
        arg_min = np.argmin(ex_trace)
    elif trace_processing_id == 4 or 5:
        ex_trace = trace_set[trace_index]
        ex_max = float(max(ex_trace[204:314]))
        ex_min = float(min(ex_trace[204:314]))
        ex_scaling = round(float(1 / (ex_max - ex_min)), 2)
        arg_max = np.argmax(ex_trace)
        arg_min = np.argmin(ex_trace)
    else:
        raise "Incorrect trace_process_id."

    ax.plot(ex_trace)
    ax.axhline(ex_max)
    ax.axhline(ex_min)
    ax.axvline(x=204, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    ax.axvline(x=314, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    plt.suptitle(
        f"Example trace (index {trace_index}) for trace process id "
        f"{trace_processing_id}. Test dataset id: {test_dataset_id}, "
        f"Environment: {environment_id}, Distance: {distance}m, "
        f"Device: {device}\nMax: {round(ex_max, 4)}, "
        f"Min: {round(ex_min, 4)}, ArgMax: {arg_max}, ArgMin: {arg_min}, "
        f"Scaling: {ex_scaling}",
        fontsize=14,
        y=0.95
    )
    plt.show()


def plot_additive_noises_examples():
    """
    Plots the additive noises.
    """
    test_dataset_id = 1
    environment_id = 1
    distance = 15
    device = 6
    trace_processing_id = 3

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)

    fig, axs = plt.subplots(1, 3, figsize=(22, 7))

    # Load a trace set
    trace_set_path = get_test_trace_path(
        database="main.db",
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
    )
    if trace_processing_id == 2:
        file_path = os.path.join(trace_set_path, "traces.npy")
    elif trace_processing_id == 3:
        file_path = os.path.join(trace_set_path, "nor_traces_maxmin.npy")
    elif trace_processing_id == 4 or 5:
        file_path = os.path.join(
            trace_set_path,
            "nor_traces_maxmin__sbox_range_204_314.npy"
        )
    else:
        raise "Incorrect trace_processing_id"
    trace_set = np.load(file_path)

    # Gaussian
    _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.01)
    axs[0].plot(noise[204:314], label="∂=0.01", alpha=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.02)
    # axs[0].plot(noise[204:314], label="∂=0.02", alpha=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.03)
    # axs[0].plot(noise[204:314], label="∂=0.03", alpha=0.75)
    _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.04)
    axs[0].plot(noise[204:314], label="∂=0.04", alpha=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.05)
    # axs[0].plot(noise[204:314], label="∂=0.05", alpha=0.75)

    # Collected
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=25, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=25", alpha=0.75)
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=50, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=50", alpha=0.75)
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=75, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=75", alpha=0.75)
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=105, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=105", alpha=0.75)

    # Rayleigh
    _, noise = additive_noise__rayleigh(trace_set=trace_set, mode=0.0138)
    axs[2].plot(noise[204:314], label="mode=0.0138", alpha=0.75)
    _, noise = additive_noise__rayleigh(trace_set=trace_set, mode=0.0276)
    axs[2].plot(noise[204:314], label="mode=0.0276", alpha=0.75)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].set_ylim(-0.15, 0.15)
    axs[1].set_ylim(-0.15, 0.15)
    axs[2].set_ylim(-0.15, 0.15)
    axs[0].set_title("Gaussian Noise")
    axs[1].set_title("Recorded Noise")
    axs[2].set_title("Rayleigh")
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

