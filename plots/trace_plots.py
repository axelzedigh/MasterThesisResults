import os
import sqlite3 as lite
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_4_CUSTOM_LINES, NORD_LIGHT_LIGHT_BLUE
from utils.db_utils import get_db_absolute_path, get_test_trace_path
from utils.plot_utils import set_size
from utils.trace_utils import get_training_trace_path
from utils.training_utils import additive_noise__gaussian, \
    additive_noise__collected_noise__office_corridor, additive_noise__rayleigh, \
    cut_trace_set__column_range__randomized


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
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False  # don't setup fonts from rc parameters
    })

    fig, axs = plt.subplots(3, 1, figsize=set_size(fraction=1, subplots=(3, 1)))

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
    axs[0].plot(noise[204:314], label="$∂$=0.01", alpha=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.02)
    # axs[0].plot(noise[204:314], label="∂=0.02", alpha=0.75)
    # _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.03)
    # axs[0].plot(noise[204:314], label="∂=0.03", alpha=0.75)
    _, noise = additive_noise__gaussian(trace_set=trace_set, mean=0, std=0.04)
    axs[0].plot(noise[204:314], label="$∂$=0.04", alpha=0.75)
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