import sqlite3 as lite
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_4_CUSTOM_LINES, NORD_LIGHT_RED, NORD_LIGHT_ORANGE, \
    NORD_LIGHT_BLUE, NORD_LIGHT_LIGHT_BLUE
from utils.db_utils import get_db_absolute_path
from utils.statistic_utils import root_mean_square


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