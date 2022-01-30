import os
import sqlite3 as lite
from typing import List, Optional, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.gridspec import GridSpec

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_5_CUSTOM_LINES, NORD_LIGHT_RED, NORD_LIGHT_ORANGE, \
    NORD_LIGHT_BLUE, NORD_LIGHT_LIGHT_BLUE, REPORT_DIR, \
    NORD_LIGHT_MPL_STYLE_2_PATH
from utils.db_utils import get_db_absolute_path
from utils.plot_utils import set_size
from utils.statistic_utils import root_mean_square, \
    signal_to_noise_ratio__sqrt_mean_std
import seaborn as sns


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
    custom_lines = NORD_LIGHT_5_CUSTOM_LINES

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
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
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
    w, h = set_size(subplots=(1, 1), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    # plt.subplots_adjust(hspace=0.5)
    plt.rcParams.update({
        "ytick.labelsize": "xx-small",
        "xtick.labelsize": "xx-small",
        "axes.titlesize": "x-small",
        "grid.alpha": "0.25",
    })

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = f"""
            select 
                * 
            from 
                trace_metadata_depth
            where
                test_dataset_id = {test_dataset_id}
                AND environment_id = {environment_id}
                AND trace_process_id = {trace_process_id}
                AND distance = {distance}
            ;
    """
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    i = 1
    for device in devices:
        ax = plt.subplot(1, len(devices), i)
        if trace_process_id == 2:
            ax.set_ylim(0.0015, 0.014)
        elif trace_process_id == 3 or 4:
            ax.set_ylim(0, 1)
        data = raw_data.copy()
        data = data[data["device"] == device]
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
                transform=ax.transAxes,
                fontsize=6,
            )
        else:
            ax.text(
                0.05, 0.1,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=6,
            )
        if i > 1:
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
        i += 1
        # ax.set_title(
        #     f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
        # )
        ax.set_title(
            f"""$D_{{{device}}}$, $\sigma$: {mean_std}\nSNR: {mean_snr}"""
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
    # plt.tight_layout()
    if save_path:
        save_path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/trace_depth_mean__test_dataset_{test_dataset_id}_trace_process_{trace_process_id}_dist_{int(distance)}__env_{environment_id}.{file_format}"
        )
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_test_trace_metadata_depth__rms__report(
        test_dataset_id: int = 1,
        distance: float = 15,
        devices: List[int] = (6, 7, 8, 9, 10),
        trace_process_id: int = 2,
        environment_id: int = 1,
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
        ylabel: str = "",
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
    w, h = set_size(subplots=(0.5, 1), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    plt.rcParams.update({
        "ytick.labelsize": "xx-small",
        "xtick.labelsize": "xx-small",
        "axes.titlesize": "x-small",
        "grid.alpha": "0.25",
    })

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = f"""
            select 
                * 
            from 
                trace_metadata_depth
            where
                test_dataset_id = {test_dataset_id}
                AND environment_id = {environment_id}
                AND trace_process_id = {trace_process_id}
                AND distance = {distance}
            ;
    """
    raw_data = pd.read_sql_query(query, con2)
    con2.close()

    i = 1
    for device in devices:
        ax = plt.subplot(1, len(devices), i)
        if trace_process_id == 2:
            ax.set_ylim(0.0015, 0.014)
        elif trace_process_id == 3 or 4:
            ax.set_ylim(0, 1)
        data = raw_data.copy()
        data = data[data["device"] == device]
        data["snr_koko"] = data["mean_val"] / data["std_val"]
        data[204:314].plot(x="data_point_index", y="rms_val", ax=ax)
        top_value = np.max(data[204:314]["rms_val"], axis=0)
        bottom_value = np.min(data[204:314]["rms_val"], axis=0)
        dyn_range = top_value - bottom_value
        scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
        mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
        mean_rms = root_mean_square(data[204:314]["rms_val"])
        mean_std = round(root_mean_square(data[204:314]["std_val"]), 5)
        mean_snr = round(root_mean_square(data[204:314]["snr_val"]), 1)
        # mean_snr = round(signal_to_noise_ratio__sqrt_mean_std(mean_mean, mean_std), 1)
        # mean_snr = round(np.mean(data[204:314]), 1)
        ax.axhline(top_value, c=NORD_LIGHT_BLUE)
        ax.axhline(bottom_value, c=NORD_LIGHT_BLUE)
        ax.axhline(mean_rms, c=NORD_LIGHT_ORANGE)
        if i == 1 and ylabel:
            ax.set_ylabel(ylabel)
        if round(top_value, 5) < 0.012:
            ax.text(
                0.02, 0.97,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=6,
            )
        else:
            ax.text(
                0.05, 0.2,
                f'Range: {round(dyn_range, 4)}\nScaling: {round(scaling_factor)}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=6,
            )
        if i > 1:
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
        i += 1
        # ax.set_title(
        #     f"""Distance: {distance}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
        # )
        # ax.set_title(
        #     f"""$D_{{{device}}}$ RMS: {round(mean_rms, 4)} \n$\sigma$: {mean_std} SNR: {mean_snr}"""
        # )
        ax.set_title(
            f"""$D_{{{device}}}$ RMS: {round(mean_rms, 4)} \n$SNR_{{{"d"}}}$: {mean_snr}"""
        )
        ax.get_legend().remove()
        ax.set_xlabel("")
    # plt.tight_layout()
    if save_path:
        save_path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/trace_depth_rms__test_dataset_{test_dataset_id}_trace_process_{trace_process_id}_dist_{int(distance)}__env_{environment_id}.{file_format}"
        )
        plt.savefig(save_path)
    if show:
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


def SNR_mean(x):
    return root_mean_square(x)


def SNR_mean_2(x):
    g = signal_to_noise_ratio__sqrt_mean_std(x.rms_val, x.std_val)
    return g


def plot_trace_termination_point(
        training_dataset_id: int = 3,
        test_dataset_id: int = 1,
        device: int = 8,
        trace_process_id: int = 9,
        environment_id: int = 1,
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
        val_type: str = "snr",
        legend_on: bool = True,
):
    """

    :param device:
    :param training_dataset_id:
    :param test_dataset_id:
    :param trace_process_id:
    :param environment_id:
    :param save_path:
    :param file_format:
    :param show:
    :return:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    w, h = set_size(subplots=(1, 1), fraction=1)
    # fig = plt.figure(constrained_layout=True, figsize=(w, h))
    # gs = GridSpec(1, 2, figure=fig)
    # ax1 = fig.add_subplot(gs[0:, 0])
    # ax2 = fig.add_subplot(gs[0:, 1])
    # fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(w, h))
    # ax1 = axs[0]
    # ax2 = axs[1]
    fig = plt.figure(figsize=(w, h), constrained_layout=True)
    ax1 = fig.gca()
    # ax1 = plt.subplot(1, 2, 1)
    # ax2 = plt.subplot(1, 2, 2)

    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    query = f"""
        SELECT 
            * 
        FROM 
            quality_table_2 
        WHERE 
            trace_process_id = 2
            AND training_dataset_id = {training_dataset_id}
            AND device = {device}
            AND environment_id = {environment_id}
            AND rank_trace_process_id = {trace_process_id}
            AND denoising_method_id IS NULL
            AND count_term_p > 99
            AND data_point_index BETWEEN 204 AND 314
        ORDER BY
            additive_noise_method_id
        ;
    """

    data = pd.read_sql_query(query, con)
    data.fillna("None", inplace=True)
    data = data.groupby(
        [
            "training_dataset_id",
            "trace_process_id",
            "test_dataset_id",
            "distance",
            "environment_id",
            "rank_trace_process_id",
            "device",
            "rank_additive_noise_method_id",
            "avg_term_p",
        ]).agg(
        {
            "rms_val": ["mean"],
            "snr_val": [SNR_mean],
            "std_val": [SNR_mean],
        }
    )
    data = data.reset_index()
    data["rank_additive_noise_method_id"] = data[
        "rank_additive_noise_method_id"].replace(
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0],
        ["$\sigma=0.03$", "$\sigma=0.04$", "$\sigma=0.05$", "scale=25",
         "scale=50", "scale=75",
         "mode=0.0138", "mode=0.0276"]
    )

    if val_type == "rms":
        s = sns.lineplot(
            x=data["rms_val"]["mean"],
            y=data["avg_term_p"],
            hue=data["rank_additive_noise_method_id"],
            ax=ax1,
            marker='o'
        )
        try:
            corr = data.corr()
            corr_val = round(corr["avg_term_p"]["rms_val"][0], 3)
            # corr_val_rms_snr = round(corr["rms_val"]["mean"]["snr_val"][0], 3)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax1.text(
                0.5, 0.97,
                f"Correlation\nTermination Point $\propto RMS$: {corr_val}",
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax1.transAxes,
                fontsize=6,
                bbox=props,
            )
        except:
            pass

        s.set_ylabel("Average termination point")
        s.set_xlabel("Average $RMS_{sbox}$")
    elif val_type == "snr":
        s = sns.lineplot(
            x=data["snr_val"]["SNR_mean"],
            y=data["avg_term_p"],
            hue=data["rank_additive_noise_method_id"],
            ax=ax1,
            marker='o'
        )
        try:
            corr = data.corr()
            corr_val = round(corr["avg_term_p"]["snr_val"][0], 3)
            corr_val_rms_snr = round(corr["rms_val"]["mean"]["snr_val"][0], 3)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax1.text(
                0.5, 0.97,
                f"Correlation\nTermination Point $\propto SNR_d$: {corr_val}\n$RMS\propto SNR$: {corr_val_rms_snr}",
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax1.transAxes,
                fontsize=6,
                bbox=props,
            )
        except:
            pass
    elif val_type == "std":
        s = sns.lineplot(
            x=data["std_val"]["SNR_mean"],
            y=data["avg_term_p"],
            hue=data["rank_additive_noise_method_id"],
            ax=ax1,
            marker='o'
        )
        try:
            corr = data.corr()
            corr_val = round(corr["avg_term_p"]["std_val"][0], 3)
            corr_val_rms_std = round(corr["rms_val"]["mean"]["std_val"][0], 3)
            props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax1.text(
                0.5, 0.97,
                f"Correlation\nTermination Point $\propto \sigma_d$: {corr_val}\n$RMS \propto \sigma$: {corr_val_rms_std}",
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax1.transAxes,
                fontsize=6,
                bbox=props,
            )
        except:
            pass

        s.set_ylabel("Average termination point")
        s.set_xlabel("$\sigma_{d,sbox}$")

    if legend_on:
        s.legend(
            bbox_to_anchor=(0., 1, 1, 0),
            loc="lower left",
            mode="expand",
            ncol=4,
        )
    else:
        plt.legend([], [], frameon=False)

    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/quality_compare_{val_type}_device_{device}_training_{training_dataset_id}_test_{test_dataset_id}_env_{environment_id}.{file_format}"
        )
        plt.savefig(path)
    if show:
        plt.show()


def plot_trace_width__rms(
        test_dataset_id: Optional[Any] = 1,
        training_dataset_id: Optional[Any] = 3,
        distance: float = 15,
        device: int = 6,
        environment_id: int = 1,
        trace_process_id: int = 1,
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
):
    """

    :param test_dataset_id:
    :param training_dataset_id:
    :param distance:
    :param device:
    :param trace_process_id:
    :return:
    """
    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    w, h = set_size(subplots=(0.4, 1), fraction=1)
    fig = plt.figure(figsize=(w, h), constrained_layout=True)
    ax1 = fig.gca()
    # ax2 = plt.subplot(1, 2, 2)

    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = f"""
        SELECT 
            * 
        FROM 
            trace_metadata_width
        WHERE
            distance = {distance}
            AND environment_id = {environment_id}
            AND device = {device}
            AND trace_process_id = {trace_process_id}
        ;
    """
    data = pd.read_sql_query(query, con2)
    con2.close()

    data.fillna("None", inplace=True)
    data = data[data["test_dataset_id"] == test_dataset_id]
    data = data[data["training_dataset_id"] == training_dataset_id]
    data["dyn_range"] = data["max_val"] - data["min_val"]
    mean_rms = np.sqrt(np.sum(data["rms_val"] ** 2) / (len(data["rms_val"])))
    ax1.plot(data["trace_index"], data["rms_val"])
    ax1.axhline(mean_rms, color=NORD_LIGHT_ORANGE)
    ax1.set_ylim(mean_rms*0.8, mean_rms*1.2)
    ax1.set_ylabel(f"$A_{{{'RMS'}}}, D_{{{device}}}$")
    # ax1.set_xlabel("$Recorded traces_{{{'index'}}}$")
    ax1.set_xlabel("")
    # plt.title(
    #     f"Device: {device}, Distance: {distance}\nMean $A_{{{'RMS'}}}$: {round(mean_rms, 4)}"
    # )
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/trace_width_device_{device}_training_{training_dataset_id}_test_{test_dataset_id}_env_{environment_id}_dist_{distance}.{file_format}"
        )
        plt.savefig(path)
    if show:
        plt.show()
