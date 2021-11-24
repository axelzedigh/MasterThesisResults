"""Functions for plotting things."""
from typing import Optional

import numpy as np
import sqlite3 as lite
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os

# Setup sqlite connection
from utils.db_utils import get_db_absolute_path, get_training_model_file_path
from utils.statistic_utils import root_mean_square

database = get_db_absolute_path("main.db")
con = lite.connect(database)


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

    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='r', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='orange', lw=4)]

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
    plt.axhline(mean_mean, c="r")
    plt.axhline(mean_rms, c="g")
    labels = ["Mean", "Mean Mean", "Mean RMS"]
    plt.legend(custom_lines, labels)
    plt.show()
    return


def plot_trace_metadata_depth__big_plots():
    set1 = (1, [6, 7, 8, 9, 10], 15)
    set2 = (2, [9, 10], 2)
    set3 = (2, [8, 9, 10], 5)
    set4 = (2, [8, 9, 10], 10)
    sets = [set1, set2, set3, set4]
    for subset in sets:
        for device in subset[1]:
            test_dataset_id = subset[0]
            distance = subset[2]
            plot_trace_metadata_depth__one(test_dataset_id, distance, device, 2)


def plot_test_trace_metadata_depth__several(sets, trace_process_id):
    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)

    for subset in sets:
        plt.figure(figsize=(20, 5))
        plt.subplots_adjust(hspace=0.5)
        # plt.suptitle("Daily closing prices", fontsize=18, y=0.95)
        i = 1
        for device in subset[1]:
            ax = plt.subplot(1, 5, i)
            if trace_process_id == 2:
                ax.set_ylim(0.0015, 0.014)
            elif trace_process_id == 3:
                ax.set_ylim(0, 1)
            data = raw_data.copy()
            data = data[data["distance"] == subset[2]]
            data = data[data["device"] == device]
            data = data[data["trace_process_id"] == trace_process_id]
            data = data[data["test_dataset_id"] == subset[0]]
            data[204:314].plot(x="data_point_index", y="mean_val", ax=ax)

            top_value = np.max(data[204:314]["mean_val"], axis=0)
            bottom_value = np.min(data[204:314]["mean_val"], axis=0)
            dyn_range = top_value - bottom_value
            scaling_factor = 1 / (max(data["mean_val"]) - min(data["mean_val"]))
            mean_mean = np.mean(data[204:314]["mean_val"], axis=0)
            mean_rms = np.mean(data[204:314]["rms_val"], axis=0)
            mean_std = round(np.mean(data[204:314]["std_val"], axis=0), 5)
            mean_snr = round(np.mean(data[204:314]["snr_val"], axis=0), 1)
            ax.axhline(top_value, c="b")
            ax.axhline(bottom_value, c="b")
            ax.axhline(mean_mean, c="r")
            ax.axhline(mean_rms, c="g")
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
                f"""Distance: {subset[2]}m, Device: {device}\nStd: {mean_std}, SNR: {mean_snr}"""
            )
            ax.get_legend().remove()
            ax.set_xlabel("")
        plt.show()

    con2.close()


def plot_training_trace_metadata_depth__several(range_start=204, range_end=314):
    """

    :param range_start:
    :param range_end:
    """
    # This function is based on legacy data
    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
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

        ax.axhline(top_value, c="b")
        ax.axhline(bottom_value, c="b")
        ax.axhline(mean_mean, c="r")
        ax.axhline(mean_rms, c="g")
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

    con2.close()


def plot_training_trace_metadata_depth__training_set_used():
    database = get_db_absolute_path("main.db")
    con2 = lite.connect(database)
    query = "select * from trace_metadata_depth;"
    raw_data = pd.read_sql_query(query, con2)
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

    ax.axhline(top_value, c="b")
    ax.axhline(bottom_value, c="b")
    ax.axhline(mean_mean, c="r")
    ax.axhline(mean_rms, c="g")

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

    con2.close()


def plot_all_trace_metadata_depth():
    """
    Plots all depth metadata.
    """
    set1 = (1, [6, 7, 8, 9, 10], 15)
    set2 = (2, [9, 10], 2)
    set3 = (2, [8, 9, 10], 5)
    set4 = (2, [8, 9, 10], 10)
    sets = [set1, set2, set3, set4]

    plot_test_trace_metadata_depth__several(sets, 2)
    plot_test_trace_metadata_depth__several(sets, 3)
    plot_training_trace_metadata_depth__training_set_used()
    plot_training_trace_metadata_depth__several(130, 240)


def plot_history_log(
        trace_process_id: int,
        keybyte: int,
        additive_noise_method_id: Optional[int],
        denoising_method_id: Optional[int],
) -> None:
    training_file_path = get_training_model_file_path(
        database="main.db",
        training_model_id=1,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        epoch=1,
        keybyte=keybyte,
        trace_process_id=trace_process_id,
    )
    training_path = os.path.dirname(training_file_path)
    history_log_file_path = os.path.join(training_path, "history_log.npy")
    history_log_fig_file_path = os.path.join(training_path, "history_log.png")
    history_log_npy = np.load(history_log_file_path, allow_pickle=True)
    history_log = history_log_npy.tolist()
    plt.subplot(1, 2, 1)
    plt.plot(history_log["accuracy"])
    plt.plot(history_log["val_accuracy"])
    plt.subplot(1, 2, 2)
    plt.plot(history_log["loss"])
    plt.plot(history_log["val_loss"])
    plt.savefig(fname=history_log_fig_file_path)
    plt.show()
