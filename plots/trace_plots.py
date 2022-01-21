import os
import sqlite3 as lite
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_5_CUSTOM_LINES, NORD_LIGHT_LIGHT_BLUE, REPORT_DIR, \
    NORD_LIGHT_MPL_STYLE_2_PATH, NORD_LIGHT_YELLOW, NORD_LIGHT_RED
from utils.db_utils import get_db_absolute_path, get_test_trace_path, \
    get_training_model_file_path
from utils.plot_utils import set_size
from utils.trace_utils import get_training_trace_path
from utils.training_utils import additive_noise__gaussian, \
    additive_noise__collected_noise__office_corridor, additive_noise__rayleigh, \
    cut_trace_set__column_range__randomized, cut_trace_set__column_range, \
    denoising_of_trace_set


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
    custom_lines = NORD_LIGHT_5_CUSTOM_LINES

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


def plot_overview(
        test_dataset_id: int = 1,
        environment_id: int = 1,
        distance: float = 15,
        device: int = 6,
        trace_index: int = 1,
        trace_process_id: int = 3,
        save_path: str = REPORT_DIR,
        format: str = "png",
        show: bool = True,
):
    """
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param trace_process_id:
    :param trace_index: The index of the trace in the trace set.
    :param save_path:
    :param format:
    :param show:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    plt.rcParams.update({
        "lines.linewidth": 1,
    })
    w, h = set_size(subplots=(2, 2), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(1, 8, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0:4])
    ax2 = fig.add_subplot(gs[0:, 4:6])
    ax3 = fig.add_subplot(gs[0:, 6:8])

    # Load a trace set
    trace_set_path = get_test_trace_path(
        database="main.db",
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
    )
    if trace_process_id == 2:
        file_path = os.path.join(trace_set_path, "traces.npy")
    elif trace_process_id == 3:
        file_path = os.path.join(trace_set_path, "nor_traces_maxmin.npy")
    elif trace_process_id == 4 or 5:
        file_path = os.path.join(
            trace_set_path,
            "nor_traces_maxmin__sbox_range_204_314.npy"
        )
    else:
        raise "Incorrect trace_processing_id"
    trace_set = np.load(file_path)

    # Max/min
    if trace_process_id == 2 or 3:
        ex_trace = trace_set[trace_index]
        ex_max = float(max(ex_trace))
        ex_min = float(min(ex_trace))
        ex_scaling = round(float(1 / (ex_max - ex_min)), 2)
        arg_max = np.argmax(ex_trace)
        arg_min = np.argmin(ex_trace)
    elif trace_process_id == 4 or 5:
        ex_trace = trace_set[trace_index]
        ex_max = float(max(ex_trace[204:314]))
        ex_min = float(min(ex_trace[204:314]))
        ex_scaling = round(float(1 / (ex_max - ex_min)), 2)
        arg_max = np.argmax(ex_trace)
        arg_min = np.argmin(ex_trace)
    else:
        raise "Incorrect trace_process_id."

    ax1.plot(ex_trace)
    ax1.axhline(ex_max)
    ax1.axhline(ex_min)
    ax1.axvline(x=204, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    ax1.axvline(x=314, color=NORD_LIGHT_LIGHT_BLUE, linestyle="--")
    # plt.suptitle(
    #     f"Example trace (index {trace_index}) for trace process id "
    #     f"{trace_processing_id}. Test dataset id: {test_dataset_id}, "
    #     f"Environment: {environment_id}, Distance: {distance}m, "
    #     f"Device: {device}\nMax: {round(ex_max, 4)}, "
    #     f"Min: {round(ex_min, 4)}, ArgMax: {arg_max}, ArgMin: {arg_min}, "
    #     f"Scaling: {ex_scaling}",
    #     fontsize=14,
    #     y=0.95
    # )

    # History plot
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}",
            f"trace_and_history.{format}",
        )
        plt.savefig(path)
    if show:
        plt.show()


def plot_example_test_traces_with_max_min(
        test_dataset_id: int = 1,
        environment_id: int = 1,
        distance: float = 15,
        device: int = 6,
        trace_processing_id: int = 2,
        trace_index: int = 1,
        save_path: str = REPORT_DIR,
        format: str = "png",
        show: bool = True,
):
    """
    :param test_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param trace_processing_id:
    :param trace_index: The index of the trace in the trace set.
    :param save_path:
    :param show:
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
    if save_path:
        path = os.path.join(save_path, f"figures/example_noise_traces.{format}")
        plt.savefig(path)
    if show:
        plt.show()


def plot_additive_noises_examples(
        save_path: Optional[str] = None,
        format: str = "png",
        show: bool = False,
):
    """
    Plots the additive noises.
    """
    test_dataset_id = 1
    environment_id = 1
    distance = 15
    device = 6
    trace_processing_id = 3

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    fig, axs = plt.subplots(3, 1, figsize=set_size(fraction=1, subplots=(1, 1)))

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
    axs[0].plot(noise[204:314], label="$\sigma$=0.01", alpha=0.75, lw=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.02)
    # axs[0].plot(noise[204:314], label="∂=0.02", alpha=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.03)
    # axs[0].plot(noise[204:314], label="∂=0.03", alpha=0.75)
    _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.04)
    axs[0].plot(noise[204:314], label="$\sigma$=0.04", alpha=0.75, lw=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.05)
    # axs[0].plot(noise[204:314], label="∂=0.05", alpha=0.75)

    # Collected
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=25, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=25", alpha=0.75, lw=0.75)
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=50, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=50", alpha=0.75, lw=0.75)
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=75, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=75", alpha=0.75, lw=0.75)
    _, noise = additive_noise__collected_noise__office_corridor(
        trace_set=trace_set, scaling_factor=105, mean_adjust=False
    )
    axs[1].plot(noise[204:314], label="scaling=105", alpha=0.75, lw=0.75)

    # Rayleigh
    _, noise = additive_noise__rayleigh(trace_set=trace_set, mode=0.0138)
    axs[2].plot(noise[204:314], label="mode=0.0138", alpha=0.75, lw=0.75)
    _, noise = additive_noise__rayleigh(trace_set=trace_set, mode=0.0276)
    axs[2].plot(noise[204:314], label="mode=0.0276", alpha=0.75, lw=0.75)

    leg = axs[0].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    leg = axs[1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    leg = axs[2].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

    axs[0].set_ylim(-0.15, 0.15)
    axs[1].set_ylim(-0.15, 0.15)
    axs[2].set_ylim(-0.15, 0.15)
    axs[0].set_ylabel("Gaussian")
    axs[1].set_ylabel("Recorded")
    axs[2].set_ylabel("Rayleigh")
    plt.tight_layout()

    if save_path:
        path = os.path.join(save_path, f"figures/example_noise_traces.{format}")
        plt.savefig(path)
    if show:
        plt.show()


def plot_randomized_trace_cut():
    """Test to plot and see if randomized cut works as expected."""
    training_set_path = get_training_trace_path(training_dataset_id=2)
    trace_set_file_path = os.path.join(
        training_set_path, "trace_process_8-standardization_sbox.npy"
    )
    training_trace_set = np.load(trace_set_file_path)
    trace_set = cut_trace_set__column_range__randomized(
        trace_set=training_trace_set,
        range_start=130,
        range_end=240,
        randomize=1
    )

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(trace_set[0])
    ax.plot(trace_set[1])
    ax.plot(trace_set[2])
    ax.plot(trace_set[3])
    ax.plot(trace_set[4])
    plt.show()


def plot_example_normalized_training_trace(
        training_dataset_id: int = 3,
        trace_process_id: int = 3,
        save_path: str = REPORT_DIR,
        file_format: str = "png",
        show: bool = False,
        denoising_method_id: Optional[int] = None,
):
    """

    :param training_dataset_id: 
    :param trace_process_id: 
    :param save_path: 
    :param file_format: 
    :param show: 
    :return: 
    """
    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    plt.rcParams.update({
        "lines.linewidth": 1,
    })
    w, h = set_size(subplots=(1, 1), fraction=1)
    # fig, axs = plt.figure(constrained_layout=True, figsize=(w, h))
    fig, axs = plt.subplots(2, 1, figsize=(w, h))
    ax1 = axs[0]
    ax2 = axs[1]
    # gs = GridSpec(2, 8, figure=fig)
    # ax1 = fig.add_subplot(gs[0:, 0:8])
    # ax2 = fig.add_subplot(gs[1:, 0:8])

    # Get training traces path.
    training_set_path = get_training_trace_path(training_dataset_id)

    # Get training traces (based on trace process)
    trace_set_file_path_not_normalized = os.path.join(
        training_set_path, "traces.npy"
    )
    training_trace_set_not_normalized = np.load(trace_set_file_path_not_normalized)
    if trace_process_id in [3, 13]:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id in [4, 5]:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 6:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_6-max_avg(before_sbox).npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 7:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_7-max_avg(sbox).npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id in [8, 11, 12]:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_8-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 9:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_9-maxmin_[-1_1]_[0_400].npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 10:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_10-maxmin_[-1_1]_[204_314].npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 14:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_14-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    else:
        return "Trace_process_id is wrong!"

    if denoising_method_id is not None:
        training_trace_set, start, end, clean_trace = denoising_of_trace_set(
            trace_set=training_trace_set,
            denoising_method_id=denoising_method_id,
            training_dataset_id=training_dataset_id,
        )

    if trace_process_id == 11:
        training_trace_set -= np.mean(training_trace_set, axis=0)
        # training_trace_set *= 20

    ax1.plot(training_trace_set_not_normalized[0])
    ax1.axhline(np.max(training_trace_set_not_normalized[0]))
    ax1.axhline(np.min(training_trace_set_not_normalized[0]))
    if training_dataset_id == 1:
        ax1.axvline(x=204, color=NORD_LIGHT_RED, linestyle="--")
        ax1.axvline(x=314, color=NORD_LIGHT_RED, linestyle="--")
    else:
        ax1.axvline(x=130, color=NORD_LIGHT_RED, linestyle="--")
        ax1.axvline(x=240, color=NORD_LIGHT_RED, linestyle="--")

    if training_dataset_id == 1:
        ax2.plot(training_trace_set[0][204:314])
        ax2.axhline(np.max(training_trace_set[0][204:314]))
        ax2.axhline(np.min(training_trace_set[0][204:314]))
    else:
        if trace_process_id in [12, 13, 14]:
            training_trace_set = cut_trace_set__column_range__randomized(
                trace_set=training_trace_set,
                range_start=130,
                range_end=240,
                randomize=1,
            )
        else:
            training_trace_set = cut_trace_set__column_range(
                trace_set=training_trace_set,
                range_start=130,
                range_end=240,
            )
        ax2.plot(training_trace_set[0])
        if trace_process_id in [12, 13, 14]:
            ax2.plot(training_trace_set[1])
            ax2.plot(training_trace_set[2])
        ax2.axhline(np.max(training_trace_set[0]))
        ax2.axhline(np.min(training_trace_set[0]))

    if trace_process_id in [3, 4, 5, 6, 7]:
        ax2.set_ylim(0, 1)
    elif trace_process_id in [8, 12, 14]:
        ax2.set_ylim(-2.5, 2.5)
    elif trace_process_id in [11]:
        # ax2.set_ylim(-2.5, 2.5)
        pass
    elif trace_process_id in [9, 10]:
        ax2.set_ylim(-1, 1)

    # if training_dataset_id == 1:
    #     ax2.axvline(x=204, color=NORD_LIGHT_RED, linestyle="--")
    #     ax2.axvline(x=314, color=NORD_LIGHT_RED, linestyle="--")
    # else:
    #     ax2.axvline(x=130, color=NORD_LIGHT_RED, linestyle="--")
    #     ax2.axvline(x=240, color=NORD_LIGHT_RED, linestyle="--")
    #
    plt.tight_layout()

    if save_path:
        if denoising_method_id:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}",
                f"example_normalized_training_trace_denoising_{denoising_method_id}.{file_format}",
            )
            plt.savefig(path)
        else:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}",
                f"example_normalized_training_trace.{file_format}",
            )
            plt.savefig(path)
    if show:
        plt.show()


def plot_example_normalized_training_trace_1_row(
        training_dataset_id: int = 3,
        trace_process_id: int = 3,
        save_path: str = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
        denoising_method_id: Optional[int] = None,
):
    """

    :param training_dataset_id:
    :param trace_process_id:
    :param denoising_method_id:
    :param save_path:
    :param file_format:
    :param show:
    :return:
    """
    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    plt.rcParams.update({
        "lines.linewidth": 1,
    })
    w, h = set_size(subplots=(0.5, 1), fraction=1)
    fig, ax1 = plt.subplots(1, 1, figsize=(w, h))

    # Get training traces path.
    training_set_path = get_training_trace_path(training_dataset_id)

    # Get training traces (based on trace process)
    if trace_process_id in [3, 13]:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
        if trace_process_id == 3:
            label = "MaxMin [0, 1]\nWhole trace"
        elif trace_process_id == 13:
            label = "MaxMin [0, 1]\nWhole trace\nTranslation ±1"
    elif trace_process_id in [4, 5]:
        trace_set_file_path = os.path.join(
            training_set_path, "nor_traces_maxmin__sbox_range_204_314.npy"
        )
        label = "MaxMin [0, 1]\nSbox"
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 6:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_6-max_avg(before_sbox).npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 7:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_7-max_avg(sbox).npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id in [8, 11, 12]:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_8-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
        if trace_process_id == 8:
            label = "Standardization Sbox"
        elif trace_process_id == 11:
            label = "Standardization Sbox\nMean difference * 40"
        elif trace_process_id == 12:
            label = "Standardization Sbox\nTranslation ±1"
    elif trace_process_id == 9:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_9-maxmin_[-1_1]_[0_400].npy"
        )
        training_trace_set = np.load(trace_set_file_path)
        label = "MaxMin [-1, 1]\nWhole trace"
    elif trace_process_id == 10:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_10-maxmin_[-1_1]_[204_314].npy"
        )
        label = "MaxMin [-1, 1]\nSbox"
        training_trace_set = np.load(trace_set_file_path)
    elif trace_process_id == 14:
        trace_set_file_path = os.path.join(
            training_set_path, "trace_process_14-standardization_sbox.npy"
        )
        training_trace_set = np.load(trace_set_file_path)
    else:
        return "Trace_process_id is wrong!"

    if denoising_method_id is not None:
        training_trace_set, start, end, clean_trace = denoising_of_trace_set(
            trace_set=training_trace_set,
            denoising_method_id=denoising_method_id,
            training_dataset_id=training_dataset_id,
        )

    if trace_process_id == 11:
        training_trace_set -= np.mean(training_trace_set, axis=0)
        training_trace_set *= 40

    if training_dataset_id == 1:
        ax1.plot(training_trace_set[0][204:314])
        ax1.axhline(np.max(training_trace_set[0][204:314]))
        ax1.axhline(np.min(training_trace_set[0][204:314]))
    else:
        if trace_process_id in [12, 13, 14]:
            training_trace_set = cut_trace_set__column_range__randomized(
                trace_set=training_trace_set,
                range_start=130,
                range_end=240,
                randomize=1,
            )
        else:
            training_trace_set = cut_trace_set__column_range(
                trace_set=training_trace_set,
                range_start=130,
                range_end=240,
            )
        ax1.plot(training_trace_set[0])
        ax1.plot(training_trace_set[111111])
        ax1.plot(training_trace_set[-1])
        # ax1.axhline(np.max(training_trace_set[0]))
        # ax1.axhline(np.min(training_trace_set[0]))

    ax1.set_ylabel(label)

    if trace_process_id in [3, 4, 5, 6, 7]:
        ax1.set_ylim(0, 1)
    elif trace_process_id in [8, 12, 14]:
        ax1.set_ylim(-2.5, 2.5)
    elif trace_process_id in [11]:
        # ax2.set_ylim(-2.5, 2.5)
        pass
    elif trace_process_id in [9, 10]:
        ax1.set_ylim(-1, 1)
    plt.tight_layout()

    if save_path:
        if denoising_method_id:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}",
                f"example_normalized_training_trace_denoising_{denoising_method_id}__sbox.{file_format}",
            )
            plt.savefig(path)
        else:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}",
                f"example_normalized_training_trace__sbox.{file_format}",
            )
            plt.savefig(path)
    if show:
        plt.show()
