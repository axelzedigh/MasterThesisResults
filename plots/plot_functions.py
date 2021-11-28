"""Functions for plotting things."""
from typing import Optional, Tuple
import numpy as np
import sqlite3 as lite
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os

from utils.db_utils import get_db_absolute_path, get_training_model_file_path
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
    plt.axhline(mean_mean, c="r")
    plt.axhline(mean_rms, c="g")
    labels = ["Mean", "Mean Mean", "Mean RMS"]
    plt.legend(custom_lines, labels)
    plt.show()
    con.close()
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

    con.close()


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
        save: bool = False,
) -> None:
    """
    Plot the history function. Accuracy on the left, loss on the right.
    """
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
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle("Accuracy & Loss", fontsize=18, y=0.95)

    # Subplot 1 - Accuracy
    ax1 = fig.add_axes((0.1, 0.1, 0.35, 0.8))
    ax1.plot(history_log["accuracy"], solid_capstyle="round", linewidth=2)
    ax1.plot(history_log["val_accuracy"], solid_capstyle="round")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    # Subplot 2 - Loss
    ax2 = fig.add_axes((0.55, 0.1, 0.35, 0.8))
    ax2.plot(history_log["loss"], linewidth=2)
    ax2.plot(history_log["val_loss"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    if save:
        plt.savefig(fname=history_log_fig_file_path)
    plt.show()


def plot_best_additive_noise_methods(
        trace_process_id: int = 3,
        gaussian_value: float = 0.04,
        collected_value: float = 25,
        rayleigh_value: float = 0.0138,
        save: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param trace_process_id: The parameter 1 value.
    :param gaussian_value:  The parameter 1 value.
    :param collected_value: The parameter 1 value.
    :param rayleigh_value:  The parameter 1 value.
    :param save: To save figure or not.
    """
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='orange', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='r', lw=4)]

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
    full_rank_test__zedigh.fillna("None", inplace=True)
    ylim_bottom = 0
    ylim_top = 1600
    labels = [
        "None",
        f"Collected - factor={collected_value}",
        f"Gaussian - ∂={gaussian_value}",
        f"Rayleigh - mode={rayleigh_value}"
    ]
    sns.set_theme(rc={"figure.figsize": (15, 7)})
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2)
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
    axs[0].text(x=-0.2, y=(ylim_top-100), s="Wang 2021", fontsize=16)
    axs[1].set_ylim(ylim_bottom, ylim_top)
    axs[1].set_ylabel("Termination point")
    axs[1].set_xlabel("Device")
    axs[1].text(x=-0.2, y=(ylim_top-100), s="Zedigh 2021", fontsize=16)
    plt.tight_layout()
    axs[0].legend(custom_lines, labels)
    axs[1].legend(custom_lines, labels)
    if save:
        plt.savefig("../docs/figs/Additive_noise_comparison_Wang_Zedigh.png")
    plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_all_of_an_additive_noise(
        additive_noise_method: str = "Gaussian",
        trace_process_id: int = 3,
        epoch: int = 65,
        distance: float = 15,
        save: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param additive_noise_method:
    :param trace_process_id:
    :param epoch:
    :param distance:
    :param save:
    """
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
        AND test_dataset = 'Zedigh_2021'
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
    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test__wang = pd.read_sql_query(query_wang, con)
    full_rank_test__zedigh = pd.read_sql_query(query_zedigh, con)
    full_rank_test__wang.fillna("None", inplace=True)
    full_rank_test__zedigh.fillna("None", inplace=True)
    con.close()
    ylim_bottom = 0
    ylim_top = 1600
    sns.set(rc={"figure.figsize": (15, 7)})
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2)
    sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["additive_noise_method_parameter_1_value"],
        ax=axs[0]
    )
    sns.barplot(
        x=full_rank_test__zedigh["device"],
        y=full_rank_test__zedigh["termination_point"],
        hue=full_rank_test__zedigh["additive_noise_method_parameter_1_value"],
        ax=axs[1]
    )
    plt.suptitle(
        f"{additive_noise_method.capitalize()} additive noise, {distance}m, trace process {trace_process_id}",
        fontsize=18,
        y=0.95
    )
    axs[0].set_ylim(ylim_bottom, ylim_top)
    axs[0].set_ylabel("Termination point")
    axs[0].set_xlabel("Device")
    axs[0].text(x=-0.2, y=(ylim_top-100), s="Wang 2021", fontsize=16)
    axs[1].set_ylim(ylim_bottom, ylim_top)
    axs[1].set_ylabel("Termination point")
    axs[1].set_xlabel("Device")
    axs[1].text(x=-0.2, y=(ylim_top-100), s="Zedigh 2021", fontsize=16)
    plt.tight_layout()
    plt.tight_layout()
    if save:
        plt.savefig(f"../docs/figs/{additive_noise_method.replace(' ', '_')}_comparison.png")
    plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_all_of_denoising(
        denoising_method: str = "Moving Average Filter",
        trace_process_id: int = 3,
        epoch: int = 65,
        distance: float = 15,
        save: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param denoising_method:
    :param trace_process_id:
    :param epoch:
    :param distance:
    :param save:
    """
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
    full_rank_test__zedigh.fillna("None", inplace=True)
    con.close()
    ylim_bottom = 0
    ylim_top = 1600
    sns.set(rc={"figure.figsize": (15, 7)})
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1, 2)
    sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["denoising_method_parameter_1_value"],
        ax=axs[0]
    )
    sns.barplot(
        x=full_rank_test__zedigh["device"],
        y=full_rank_test__zedigh["termination_point"],
        hue=full_rank_test__zedigh["denoising_method_parameter_1_value"],
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
    axs[0].text(x=-0.2, y=(ylim_top-100), s="Wang 2021", fontsize=16)
    axs[1].set_ylim(ylim_bottom, ylim_top)
    axs[1].set_ylabel("Termination point")
    axs[1].set_xlabel("Device")
    axs[1].text(x=-0.2, y=(ylim_top-100), s="Zedigh 2021", fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(f"../docs/figs/{denoising_method.replace(' ', '_')}_comparison.png")
    plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_epoch_comparison(
        test_dataset: str = "Wang_2021",
        device: int = 6,
        distance: float = 15,
        additive_noise_method: str = "Gaussian",
        additive_noise_method_parameter_1_value: float = 0.04,
        save: bool = False,
) -> pd.DataFrame:
    """
    :param test_dataset:
    :param device:
    :param distance:
    :param additive_noise_method:
    :param additive_noise_method_parameter_1_value:
    :param save:
    :return: Pandas DataFrame
    """

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
    ylim_bottom = 100
    ylim_top = 800
    sns.set(rc={"figure.figsize": (15, 7)})
    sns.set_style("whitegrid")
    sns.barplot(x=full_rank_test["epoch"],
                y=full_rank_test["termination_point"],
                hue=full_rank_test["additive_noise_method_parameter_1_value"],
                capsize=0.3,)
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
    pass
