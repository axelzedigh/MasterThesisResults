import os
import sqlite3 as lite
from typing import Tuple, Optional, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_4_CUSTOM_LINES, NORD_LIGHT_MPL_STYLE_2_PATH, REPORT_DIR, \
    NORD_LIGHT_BLUE, NORD_LIGHT_RED, NORD_LIGHT_LIGHT_BLUE
from utils.db_utils import get_db_absolute_path
from utils.plot_utils import set_size


def plot_best_additive_noise_methods(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        trace_process_id: int = 3,
        gaussian_value: float = 0.04,
        collected_value: float = 25,
        rayleigh_value: float = 0.0138,
        save_path: Optional[str] = None,
        format: str = "png",
        show: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param show:
    :param save_path:
    :param training_dataset:
    :param trace_process_id: The parameter 1 value.
    :param gaussian_value:  The parameter 1 value.
    :param collected_value: The parameter 1 value.
    :param rayleigh_value:  The parameter 1 value.
    :param format: Format of fig.
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    w, h = set_size(subplots=(2, 1))
    fig, axs = plt.subplots(2, 1, figsize=(w, h))

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
        termination_point,
        date_added
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
        additive_noise_method,
        date_added
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
        termination_point,
        date_added
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
        additive_noise_method,
        date_added
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
        f"Gaussian: std = {gaussian_value}",
        f"Rayleigh: Mode = {rayleigh_value}"
    ]
    sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["additive_noise_method"],
        ax=axs[0],
        errwidth=2,
    )
    sns.barplot(
        x=full_rank_test__zedigh["device"],
        y=full_rank_test__zedigh["termination_point"],
        hue=full_rank_test__zedigh["additive_noise_method"],
        ax=axs[1],
        errwidth=2,
    )
    # plt.suptitle(
    #     f"Best additive noise, 15m, trace process {trace_process_id}",
    #     fontsize=18,
    #     y=0.95
    # )
    axs[0].set_ylim(ylim_bottom, ylim_top)
    axs[0].set_ylabel("Termination point")
    axs[0].set_xlabel("Device")
    # axs[0].text(x=-0.1, y=(ylim_top - 200), s="Wang 2021", fontsize=16)
    axs[1].set_ylim(ylim_bottom, ylim_top)
    axs[1].set_ylabel("Termination point")
    axs[1].set_xlabel("Device")
    # axs[1].text(x=-0.1, y=(ylim_top - 200), s="Zedigh 2021", fontsize=16)
    # plt.tight_layout()
    # axs[0].legend(custom_lines, labels)
    # axs[1].legend(custom_lines, labels)
    axs[0].legend(
        handles=custom_lines,
        labels=labels,
        bbox_to_anchor=(0, 1, 1, 0),
        loc="lower left",
        mode="expand",
        ncol=2
    )
    axs[1].legend([], [], frameon=False)
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/Additive_noise_comparison_Wang_Zedigh.{format}"
        )
        plt.savefig(path)
    if show:
        plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_best_additive_noise_methods_2(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 500k traces',
        epoch_none: int = 12,
        epoch_gaussian: int = 12,
        epoch_collected: int = 12,
        epoch_rayleigh: int = 12,
        distance: float = 15,
        trace_process_id: int = 3,
        gaussian_value: float = 0.04,
        collected_value: float = 25,
        rayleigh_value: float = 0.0138,
        save_path: Optional[str] = None,
        file_format: str = "pgf",
        show: bool = False,
        y_bottom: int = 0,
        y_top: int = 1200,
        row_size: float = 2,
        col_size: float = 2,
        x_label: bool = True,
        y_label_subtext: Optional[str] = None,
        custom_labels: bool = False,
        training_dataset_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param training_dataset:
    :param trace_process_id: The parameter 1 value.
    :param gaussian_value:  The parameter 1 value.
    :param collected_value: The parameter 1 value.
    :param rayleigh_value:  The parameter 1 value.
    :param epoch_none:
    :param epoch_gaussian:
    :param epoch_collected:
    :param epoch_rayleigh:
    :param distance:
    :param show:
    :param save_path:
    :param file_format: Format of fig.
    :param y_bottom:
    :param y_top:
    :param row_size:
    :param col_size:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    w, h = set_size(subplots=(row_size, col_size), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(1, 7, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0:5])
    ax2 = fig.add_subplot(gs[0:, 5:7])
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
        AND distance = {distance}
        AND training_dataset = '{training_dataset}'
        AND environment = 'office_corridor'
        AND test_dataset = 'Wang_2021'
        AND distance = {distance}
        AND denoising_method IS NULL
        AND (
            (additive_noise_method_parameter_1_value IS NULL AND epoch = {epoch_none})
            OR (additive_noise_method_parameter_1_value = {gaussian_value} AND epoch = {epoch_gaussian})
            OR (additive_noise_method_parameter_1_value = {collected_value} AND epoch = {epoch_collected})
            OR (additive_noise_method_parameter_1_value = {rayleigh_value} AND epoch = {epoch_rayleigh})
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
        AND device != 9
        AND distance = {distance}
        AND denoising_method IS NULL
        AND (
            (additive_noise_method_parameter_1_value IS NULL AND epoch = {epoch_none})
            OR (additive_noise_method_parameter_1_value = {gaussian_value} AND epoch = {epoch_gaussian})
            OR (additive_noise_method_parameter_1_value = {collected_value} AND epoch = {epoch_collected})
            OR (additive_noise_method_parameter_1_value = {rayleigh_value} AND epoch = {epoch_rayleigh})
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
    labels = [
        f"None: $e_{{{epoch_none}}}$",
        f"Re: {collected_value} $e_{{{epoch_collected}}}$",
        f"G: {gaussian_value} $e_{{{epoch_gaussian}}}$",
        f"Ra: {rayleigh_value} $e_{{{epoch_rayleigh}}}$",
    ]
    sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["additive_noise_method"],
        ax=ax1,
        errwidth=1.5,
    )
    ax1.set_ylim(y_bottom, y_top)
    if y_label_subtext is not None:
        ax1.set_ylabel(f"{y_label_subtext}\nTermination point")
    else:
        ax1.set_ylabel("Termination point")
    if x_label:
        ax1.set_xlabel("Device\n(Wang 2021)")
    if custom_labels:
        ax1.legend(
            handles=custom_lines,
            labels=labels,
            bbox_to_anchor=(0., 1, 1.4, 0),
            loc="lower left",
            mode="expand",
            ncol=5
        )
    else:
        ax1.legend(
            # handles=custom_lines,
            # labels=labels,
            bbox_to_anchor=(0., 1, 1.4, 0),
            loc="lower left",
            mode="expand",
            ncol=5
        )

    try:
        sns2 = sns.barplot(
            x=full_rank_test__zedigh["device"],
            y=full_rank_test__zedigh["termination_point"],
            hue=full_rank_test__zedigh["additive_noise_method"],
            ax=ax2,
            errwidth=1.5,
        )
        sns2.set(ylabel=None, yticklabels=[])
        sns2.tick_params(left=False)
        ax2.set_ylim(y_bottom, y_top)
        if x_label:
            ax2.set_xlabel("Device\n(Zedigh 2021)")
        else:
            ax2.set_xlabel("")
        ax2.legend([], [], frameon=False)
    except:
        pass
    if save_path:
        if training_dataset_id:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}/additive_noise_comparison_wang_zedigh__{training_dataset_id}.{file_format}"
            )
        else:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}/additive_noise_comparison_wang_zedigh.{file_format}"
            )
        plt.savefig(path)
    if show:
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
    if save:
        plt.savefig(
            f"../docs/figs/{additive_noise_method.replace(' ', '_')}_comparison.png")
    plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_all_of_an_additive_noise__report(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        additive_noise_method: str = "Gaussian",
        trace_process_id: int = 3,
        epoch: int = 65,
        distance: float = 15,
        environment: str = "office_corridor",
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
        x_label: bool = True,
        y_label_subtext: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param training_dataset:
    :param additive_noise_method:
    :param trace_process_id:
    :param epoch:
    :param distance:
    :param environment:
    :param save_path:
    :param file_format:
    :param show:
    :param x_label:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    # custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    w, h = set_size(subplots=(2, 2), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(1, 7, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0:5])
    ax2 = fig.add_subplot(gs[0:, 5:7])

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
        AND device != 9
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
    ylim_top = 1500
    sns1 = sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["Additive parameter 1"],
        ax=ax1
    )
    ax1.set_ylim(ylim_bottom, ylim_top)
    if y_label_subtext is not None:
        ax1.set_ylabel(f"Termination point\n({y_label_subtext})")
    else:
        ax1.set_ylabel("Termination point")
    if x_label:
        ax1.set_xlabel("Device\n(Wang 2021)")
    ax1.legend(
        # handles=custom_lines,
        # labels=labels,
        bbox_to_anchor=(0., 1, 1.4, 0),
        loc="lower left",
        mode="expand",
        ncol=5
    )

    try:
        sns2 = sns.barplot(
            x=full_rank_test__zedigh["device"],
            y=full_rank_test__zedigh["termination_point"],
            hue=full_rank_test__zedigh["Additive parameter 1"],
            ax=ax2
        )
        sns2.set(ylabel=None, yticklabels=[])
        sns2.tick_params(left=False)
        ax2.set_ylim(ylim_bottom, ylim_top)
        if x_label:
            ax2.set_xlabel("Device\n(Zedigh 2021)")
        ax2.legend([], [], frameon=False)
    except:
        pass
    # plt.tight_layout()

    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/{additive_noise_method}_comparison.{file_format}"
        )
        plt.savefig(path)
    if show:
        plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_all_of_an_additive_noise__report__2(
        training_model: str = 'cnn_110',
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 500k traces',
        additive_noise_method: str = "Gaussian",
        parameter_1_value_1: float = 0.01,
        parameter_1_value_2: float = 0.03,
        parameter_1_value_3: float = 0.04,
        parameter_1_value_4: float = 0.05,
        trace_process_id: int = 3,
        epoch_none: int = 12,
        epoch_1: int = 12,
        epoch_2: int = 12,
        epoch_3: int = 12,
        epoch_4: int = 12,
        distance: float = 15,
        environment: str = "office_corridor",
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
        y_bottom: int = 0,
        y_top: int = 1200,
        row_size: float = 2,
        col_size: float = 2,
        x_label: bool = True,
        y_label_subtext: Optional[str] = None,
        labels: Optional[List] = None,
        training_dataset_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param training_model:
    :param parameter_1_value_4:
    :param parameter_1_value_3:
    :param parameter_1_value_2:
    :param parameter_1_value_1:
    :param training_dataset:
    :param additive_noise_method:
    :param trace_process_id:
    :param epoch_none:
    :param epoch_1:
    :param epoch_2:
    :param epoch_3:
    :param epoch_4:
    :param distance:
    :param environment:
    :param save_path:
    :param file_format:
    :param show:
    :param y_bottom:
    :param y_top:
    :param row_size:
    :param col_size:
    :param x_label:
    :param y_label_subtext:
    :param labels:
    :param training_dataset_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    w, h = set_size(subplots=(row_size, col_size), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(1, 7, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0:5])
    ax2 = fig.add_subplot(gs[0:, 5:7])

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
            AND training_model = '{training_model}'
            AND training_dataset = '{training_dataset}'
            AND test_dataset = 'Wang_2021'
            AND environment = '{environment}'
            AND distance = {distance}
            AND denoising_method IS NULL
            AND (additive_noise_method IS NULL OR additive_noise_method = '{additive_noise_method}')
            AND (
                (additive_noise_method IS NULL AND epoch = {epoch_none})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_1} AND epoch = {epoch_1})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_2} AND epoch = {epoch_2})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_3} AND epoch = {epoch_3})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_4} AND epoch = {epoch_4})
            )
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
            AND training_model = '{training_model}'
            AND training_dataset = '{training_dataset}'
            AND test_dataset = 'Zedigh_2021'
            AND environment = '{environment}'
            AND distance = {distance}
            AND device != 9
            AND denoising_method IS NULL
            AND (additive_noise_method IS NULL OR additive_noise_method = '{additive_noise_method}')
            AND (
                (additive_noise_method IS NULL AND epoch = {epoch_none})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_1} AND epoch = {epoch_1})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_2} AND epoch = {epoch_2})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_3} AND epoch = {epoch_3})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_4} AND epoch = {epoch_4})
            )
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
    sns1 = sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["Additive parameter 1"],
        ax=ax1
    )
    ax1.set_ylim(y_bottom, y_top)
    if y_label_subtext is not None:
        ax1.set_ylabel(f"{y_label_subtext}\nTermination point")
    else:
        ax1.set_ylabel("Termination point")
    if x_label:
        ax1.set_xlabel("Device\n(Wang 2021)")
    else:
        ax1.set_xlabel("")
    if labels:
        ax1.legend(
            handles=custom_lines,
            labels=labels,
            bbox_to_anchor=(0., 1, 1.4, 0),
            loc="lower left",
            mode="expand",
            ncol=5
        )
    else:
        ax1.legend(
            # handles=custom_lines,
            # labels=labels,
            bbox_to_anchor=(0., 1, 1.4, 0),
            loc="lower left",
            mode="expand",
            ncol=5
        )

    try:
        sns2 = sns.barplot(
            x=full_rank_test__zedigh["device"],
            y=full_rank_test__zedigh["termination_point"],
            hue=full_rank_test__zedigh["Additive parameter 1"],
            ax=ax2
        )
        sns2.set(ylabel=None, yticklabels=[])
        sns2.tick_params(left=False)
        ax2.set_ylim(y_bottom, y_top)
        if x_label:
            ax2.set_xlabel("Device\n(Zedigh 2021)")
        else:
            ax2.set_xlabel("")
        ax2.legend([], [], frameon=False)
    except:
        pass
    # plt.tight_layout()

    if save_path:
        if training_dataset_id:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}/{additive_noise_method}_comparison__diff_epochs__{training_dataset_id}.{file_format}"
            )
        else:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}/{additive_noise_method}_comparison__diff_epochs.{file_format}"
            )
        plt.savefig(path)
    if show:
        plt.show()
    return full_rank_test__wang, full_rank_test__zedigh


def plot_all_of_an_additive_noise__report__zedigh_distances(
        training_model: str = 'cnn_110',
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 500k traces',
        additive_noise_method: str = "Gaussian",
        parameter_1_value_1: float = 0.01,
        parameter_1_value_2: float = 0.03,
        parameter_1_value_3: float = 0.04,
        parameter_1_value_4: float = 0.05,
        trace_process_id: int = 3,
        epoch_none: int = 12,
        epoch_1: int = 12,
        epoch_2: int = 12,
        epoch_3: int = 12,
        epoch_4: int = 12,
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
        y_bottom: int = 0,
        y_top: int = 1200,
        row_size: float = 2,
        col_size: float = 2,
        x_label: bool = True,
        y_label_subtext: Optional[str] = None,
        labels: Optional[List] = None,
        training_dataset_id: Optional[int] = None,
        big_hall: bool = False,
) -> Tuple[pd.DataFrame]:
    """
    :param big_hall:
    :param training_model:
    :param parameter_1_value_4:
    :param parameter_1_value_3:
    :param parameter_1_value_2:
    :param parameter_1_value_1:
    :param training_dataset:
    :param additive_noise_method:
    :param trace_process_id:
    :param epoch_none:
    :param epoch_1:
    :param epoch_2:
    :param epoch_3:
    :param epoch_4:
    :param save_path:
    :param file_format:
    :param show:
    :param y_bottom:
    :param y_top:
    :param row_size:
    :param col_size:
    :param x_label:
    :param y_label_subtext:
    :param labels:
    :param training_dataset_id:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    w, h = set_size(subplots=(row_size, col_size), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    if big_hall:
        gs = GridSpec(1, 4, figure=fig)
    else:
        gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0:1])
    ax2 = fig.add_subplot(gs[0:, 1:2])
    ax3 = fig.add_subplot(gs[0:, 2:3])
    if big_hall:
        ax4 = fig.add_subplot(gs[0:, 3:4])

    query_zedigh = f"""
        select
            environment,
            device,
            epoch,
            distance,
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
            AND training_model = '{training_model}'
            AND training_dataset = '{training_dataset}'
            AND test_dataset = 'Zedigh_2021'
            AND device != 9
            AND denoising_method IS NULL
            AND (additive_noise_method IS NULL OR additive_noise_method = '{additive_noise_method}')
            AND (
                (additive_noise_method IS NULL AND epoch = {epoch_none})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_1} AND epoch = {epoch_1})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_2} AND epoch = {epoch_2})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_3} AND epoch = {epoch_3})
                OR (additive_noise_method_parameter_1_value = {parameter_1_value_4} AND epoch = {epoch_4})
            )
        order by
            additive_noise_method_parameter_1_value
            ;
    """

    database = get_db_absolute_path("main.db")
    con = lite.connect(database)
    full_rank_test__zedigh = pd.read_sql_query(query_zedigh, con)
    con.close()

    full_rank_test__zedigh.fillna("None", inplace=True)
    full_rank_test__zedigh.rename(columns={
        "additive_noise_method_parameter_1_value": "Additive parameter 1"},
        inplace=True)

    # distance 15m, office corridor
    data_15m_office = full_rank_test__zedigh.copy()
    data_15m_office = data_15m_office[data_15m_office["distance"] == 15]
    data_15m_office = data_15m_office[
        data_15m_office["environment"] == "office_corridor"]

    sns1 = sns.barplot(
        x=data_15m_office["device"],
        y=data_15m_office["termination_point"],
        hue=data_15m_office["Additive parameter 1"],
        ax=ax1
    )
    # Y
    ax1.set_ylim(y_bottom, y_top)
    if y_label_subtext is not None:
        ax1.set_ylabel(f"{y_label_subtext}\nTermination point")
    else:
        ax1.set_ylabel("Termination point")
    # X
    if x_label:
        ax1.set_xlabel("Device (15m)\nOffice Corridor")
    else:
        ax1.set_xlabel("")
    # Legend
    if labels:
        if big_hall:
            ax1.legend(
                handles=custom_lines,
                labels=labels,
                bbox_to_anchor=(0., 1, 4, 0),
                loc="lower left",
                mode="expand",
                ncol=5
            )
        else:
            ax1.legend(
                handles=custom_lines,
                labels=labels,
                bbox_to_anchor=(0., 1, 3, 0),
                loc="lower left",
                mode="expand",
                ncol=5
            )
    else:
        ax1.legend(
            # handles=custom_lines,
            # labels=labels,
            bbox_to_anchor=(0., 1, 1.4, 0),
            loc="lower left",
            mode="expand",
            ncol=5
        )
    del data_15m_office

    # distance 10m, office corridor
    data_10m_office = full_rank_test__zedigh.copy()
    data_10m_office = data_10m_office[data_10m_office["distance"] == 10]
    data_10m_office = data_10m_office[
        data_10m_office["environment"] == "office_corridor"]

    sns2 = sns.barplot(
        x=data_10m_office["device"],
        y=data_10m_office["termination_point"],
        hue=data_10m_office["Additive parameter 1"],
        ax=ax2
    )
    # Y
    ax2.set_ylim(y_bottom, y_top)
    sns2.set(ylabel=None, yticklabels=[])
    sns2.tick_params(left=False)
    # X
    if x_label:
        ax2.set_xlabel("Device (10m)\nOffice Corridor")
    else:
        ax2.set_xlabel("")
    # Legend
    ax2.legend([], [], frameon=False)
    # Rm
    del data_10m_office

    # distance 5m, office corridor
    data_5m_office = full_rank_test__zedigh.copy()
    data_5m_office = data_5m_office[data_5m_office["distance"] == 5]
    data_5m_office = data_5m_office[
        data_5m_office["environment"] == "office_corridor"]
    sns3 = sns.barplot(
        x=data_5m_office["device"],
        y=data_5m_office["termination_point"],
        hue=data_5m_office["Additive parameter 1"],
        ax=ax3
    )
    # Y
    ax3.set_ylim(y_bottom, y_top)
    sns3.set(ylabel=None, yticklabels=[])
    sns3.tick_params(left=False)
    # X
    if x_label:
        ax3.set_xlabel("Device (5m)\nOffice Corridor")
    else:
        ax3.set_xlabel("")
    # Legend
    ax3.legend([], [], frameon=False)
    # Rm
    del data_5m_office

    # distance 5m, big hall
    if big_hall:
        try:
            data_5m_hall = full_rank_test__zedigh.copy()
            data_5m_hall = data_5m_hall[data_5m_hall["distance"] == 5]
            data_5m_hall = data_5m_hall[data_5m_hall["environment"] == "big_hall"]
            sns4 = sns.barplot(
                x=data_5m_hall["device"],
                y=data_5m_hall["termination_point"],
                hue=data_5m_hall["Additive parameter 1"],
                ax=ax4,
                palette=[NORD_LIGHT_LIGHT_BLUE, NORD_LIGHT_RED]
            )
            # Y
            ax4.set_ylim(0, 3000)
            sns4.set(ylabel=None)  #, yticklabels=[])
            # sns4.tick_params(left=False)
            if x_label:
                ax4.set_xlabel("Device (5m)\nBig Hall")
            else:
                ax4.set_xlabel("")
            # Legend
            ax4.legend([], [], frameon=False)
            # Rm
            del data_5m_office
        except:
            pass

    if save_path:
        if training_dataset_id:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}/{additive_noise_method}_comparison__diff_epochs__{training_dataset_id}__zedigh_traces.{file_format}"
            )
        else:
            path = os.path.join(
                save_path,
                f"figures/{trace_process_id}/{additive_noise_method}_comparison__diff_epochs__zedigh_traces.{file_format}"
            )
        plt.savefig(path)
    if show:
        plt.show()
    return full_rank_test__zedigh


def plot_all_of_a_denoising_method__report(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 200k traces',
        denoising_method: str = "Moving Average Filter",
        trace_process_id: int = 3,
        epoch: int = 65,
        distance: float = 15,
        environment: str = "office_corridor",
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param denoising_method:
    :param training_dataset:
    :param trace_process_id:
    :param epoch:
    :param distance:
    :param environment:
    :param save_path:
    :param file_format:
    :param show:
    """
    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    # custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    w, h = set_size(subplots=(2, 2), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(1, 7, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0:5])
    ax2 = fig.add_subplot(gs[0:, 5:7])

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
        AND environment = '{environment}'
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
        AND environment = '{environment}'
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
    ylim_top = 1500
    sns1 = sns.barplot(
        x=full_rank_test__wang["device"],
        y=full_rank_test__wang["termination_point"],
        hue=full_rank_test__wang["Denoising parameter 1"],
        ax=ax1
    )
    ax1.set_ylim(ylim_bottom, ylim_top)
    ax1.set_ylabel("Termination point")
    ax1.set_xlabel("Device\n(Wang 2021)")
    ax1.legend(
        # handles=custom_lines,
        # labels=labels,
        bbox_to_anchor=(0., 1, 1.4, 0),
        loc="lower left",
        mode="expand",
        ncol=5
    )

    try:
        sns2 = sns.barplot(
            x=full_rank_test__zedigh["device"],
            y=full_rank_test__zedigh["termination_point"],
            hue=full_rank_test__zedigh["Denoising parameter 1"],
            ax=ax2
        )
        sns2.set(ylabel=None, yticklabels=[])
        sns2.tick_params(left=False)
        ax2.set_ylim(ylim_bottom, ylim_top)
        ax2.set_xlabel("Device\n(Zedigh 2021)")
        ax2.legend([], [], frameon=False)
    except:
        pass
    # plt.tight_layout()

    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/{denoising_method}_comparison.{file_format}"
        )
        plt.savefig(path)
    if show:
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


def plot_epoch_comparison_report(
        training_model_id: int = 1,
        training_dataset_id: int = 1,
        test_dataset_id: int = 1,
        trace_process_id: int = 3,
        environment_id: int = 1,
        distance: float = 15,
        device: int = 6,
        additive_noise_method_id: Optional[int] = None,
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "pgf",
        show: bool = False,
        y_bottom: int = 0,
        y_top: int = 1000,
) -> pd.DataFrame:
    """
    :param training_model_id:
    :param training_dataset_id:
    :param test_dataset_id:
    :param trace_process_id:
    :param environment_id:
    :param device:
    :param distance:
    :param additive_noise_method_id:
    :param save_path:
    :param file_format:
    :param show:
    :param y_top:
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    w, h = set_size(subplots=(1, 2), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(1, 8, figure=fig)
    ax1 = fig.add_subplot(gs[0:, 0:8])

    database = get_db_absolute_path("main.db")
    con = lite.connect(database)

    if additive_noise_method_id is None:
        query1 = f"""
        select
            device, 
            epoch, 
            termination_point
        from
            rank_test
        where
            training_model_id = {training_model_id}
            AND test_dataset_id = {test_dataset_id}
            AND training_dataset_id = {training_dataset_id}
            AND environment_id = {environment_id}
            AND device = {device}
            AND distance = {distance}
            AND trace_process_id = {trace_process_id}
            AND additive_noise_method_id IS NULL
            AND denoising_method_id IS NULL
        order by 
            epoch;
        """
    else:
        query1 = f"""
        select
            device, 
            epoch, 
            additive_noise_method_id,
            termination_point
        from
            rank_test
        where
            training_model_id = {training_model_id}
            AND test_dataset_id = {test_dataset_id}
            AND training_dataset_id = {training_dataset_id}
            AND environment_id = {environment_id}
            AND device = {device}
            AND distance = {distance}
            AND trace_process_id = {trace_process_id}
            AND additive_noise_method_id = {additive_noise_method_id}
            AND denoising_method_id IS NULL
        order by 
            epoch;
        """

    rank_test_data = pd.read_sql_query(query1, con)
    con.close()
    rank_test_data.fillna("None", inplace=True)
    sns1 = sns.barplot(
        x=rank_test_data["epoch"],
        y=rank_test_data["termination_point"],
        hue=rank_test_data["device"],
        ax=ax1,
        errwidth=2,
    )
    ax1.set_ylim(y_bottom, y_top)
    ax1.set_ylabel("Termination point")
    ax1.set_xlabel("Epoch")
    ax1.legend([], [], frameon=False)
    # plt.tight_layout()
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}/epoch_comparison_additive_method_{additive_noise_method_id}.{file_format}"
        )
        plt.savefig(path)
    if show:
        plt.show()
    return rank_test_data
