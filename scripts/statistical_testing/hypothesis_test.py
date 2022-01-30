"""Hypothesis tests."""
import os

from configs.variables import NORD_LIGHT_MPL_STYLE_2_PATH, REPORT_DIR
from utils.db_utils import get_db_absolute_path
import sqlite3 as lite
import pandas as pd

import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from bioinfokit.analys import stat
import warnings

from utils.plot_utils import set_size

warnings.filterwarnings("ignore")


def interval_upper(x):
    a = 1.96
    mean = x.mean()
    sem = x.sem()
    return mean + a * sem


def interval_lower(x):
    a = 1.96
    mean = x.mean()
    sem = x.sem()
    return mean - a * sem


def hypothesis_test_1(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 500k traces',
        test_dataset: str = "Wang_2021",
        epoch_none: int = 12,
        epoch_gaussian: int = 12,
        epoch_collected: int = 12,
        epoch_rayleigh: int = 12,
        distance: float = 15,
        trace_process_id: int = 3,
        gaussian_value: float = 0.04,
        collected_value: float = 25,
        rayleigh_value: float = 0.0138,
        histogram: bool = False,
        file_name: str = f"stats",
        label: str = f"tbl:stats",
        header: str = "",
):
    """

    :param training_dataset:
    :param epoch_none:
    :param epoch_gaussian:
    :param epoch_collected:
    :param epoch_rayleigh:
    :param distance:
    :param trace_process_id:
    :param gaussian_value:
    :param collected_value:
    :param rayleigh_value:
    :param histogram:
    :param file_name:
    :param label:
    :return:
    """

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
        AND test_dataset = '{test_dataset}'
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
    wang = pd.read_sql_query(query1, con)
    con.close()

    wang.fillna("None", inplace=True)
    wang = wang.replace({"Collected": "Recorded"})
    wang = wang.rename(
        columns={
            # "termination_point": "Termination point",
            "additive_noise_method": "noise_method",
            "device": "Device",
            "epoch": "Epoch",
            "additive_noise_method_parameter_1_value": "parameter_value",
        },
    )
    g = wang.groupby(["noise_method", "Device"])
    # wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(
    wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    # wang = pd.DataFrame(wang)

    # ANOVA one way
    no_treatment = wang["termination_point"].where(
        wang["noise_method"] == "None"
    ).dropna().reset_index(drop=True)
    gaussian = wang["termination_point"].where(
        wang["noise_method"] == "Gaussian"
    ).dropna().reset_index(drop=True)
    recorded = wang["termination_point"].where(
        wang["noise_method"] == "Recorded"
    ).dropna().reset_index(drop=True)
    rayleigh = wang["termination_point"].where(
        wang["noise_method"] == "Rayleigh"
    ).dropna().reset_index(drop=True)
    # print(f_oneway(no_treatment, gaussian, recorded, rayleigh))
    new_data = {
        "None": no_treatment,
        "Gaussian": gaussian,
        "Recorded": recorded,
        "Rayleigh": rayleigh,
    }

    new_df = pd.DataFrame(data=new_data)
    print(new_df)

    # ANOVA, 4 sample
    """
    - Check sample sizes: equal number of observation in each group
    - Calculate Mean Square for each group (MS) (SS of group/level-1); level-1 is a degrees of freedom (df) for a group
    - Calculate Mean Square error (MSE) (SS error/df of residuals)
    - Calculate F value (MS of group/MSE)
    - Calculate p value based on F value and degrees of freedom (df)
    """
    assert no_treatment.size == gaussian.size == recorded.size == rayleigh.size

    # Multiple pairwise comparison (post-hoc test)
    res = stat()
    res.tukey_hsd(
        df=wang,
        res_var='termination_point',
        xfac_var='noise_method',
        anova_model='termination_point~C(noise_method)+C(parameter_value)+C(noise_method):C(parameter_value)'
    )
    # print(res.tukey_summary.to_latex())
    latex = res.tukey_summary.to_latex(
        # header=["M", "D", "M"],
        sparsify=True,
        float_format="%.3f",
        label=label + "_wang",
        escape=False,
        caption=f"Tukeys test table - {header}.",
        position="H",
    )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables/{trace_process_id}",
        file_name + f"_wang.tex",
    )
    file = open(file_path, "w")
    file.write(latex)
    file.close()

    if histogram:
        # MPL styling
        plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
        # plt.rcParams.update({
        #     "ytick.labelsize": "xx-small",
        #     "xtick.labelsize": "xx-small",
        #     "axes.titlesize": "x-small",
        # })

        # Create 4 x 4 grid
        w, h = set_size(subplots=(2, 2), fraction=1)
        plt.figure(figsize=(w, h))
        plt.subplots_adjust(hspace=0.7, wspace=0.5)
        ax = plt.subplot(2, 2, 1)
        ax.hist(no_treatment, bins=10)
        ax.set_xlabel("None")
        ax = plt.subplot(2, 2, 2)
        ax.hist(gaussian, bins=10)
        ax.set_xlabel("Gaussian")
        ax = plt.subplot(2, 2, 3)
        ax.hist(recorded, bins=10)
        ax.set_xlabel("Recorded")
        ax = plt.subplot(2, 2, 4)
        ax.hist(rayleigh, bins=10)
        ax.set_xlabel("Rayleigh")
        plt.show()

    return wang


def hypothesis_test_best(
        training_dataset: str = 'Wang_2021 - Cable, 5 devices, 500k traces',
        test_dataset: str = "Wang_2021",
        epoch_none: int = 12,
        process_none: int = 12,
        gaussian_value: float = 0.04,
        epoch_gaussian: int = 11,
        process_gaussian: int = 9,
        collected_value: float = 50,
        epoch_collected: int = 4,
        process_collected: int = 10,
        rayleigh_value: float = 0.0276,
        epoch_rayleigh: int = 15,
        process_rayleigh: int = 9,
        distance: float = 15,
        histogram: bool = False,
        file_name: str = f"stats_best",
        label: str = f"tbl:stats_best",
        header: str = "",
):
    """

    :param training_dataset:
    :param epoch_none:
    :param epoch_gaussian:
    :param epoch_collected:
    :param epoch_rayleigh:
    :param distance:
    :param trace_process_id:
    :param gaussian_value:
    :param collected_value:
    :param rayleigh_value:
    :param histogram:
    :param file_name:
    :param label:
    :return:
    """

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
            distance = {distance}
        AND training_dataset = '{training_dataset}'
        AND environment = 'office_corridor'
        AND test_dataset = '{test_dataset}'
        AND distance = {distance}
        AND denoising_method IS NULL
        AND (
            (additive_noise_method_parameter_1_value IS NULL AND epoch = {epoch_none} AND trace_process_id = {process_none})
            OR (additive_noise_method_parameter_1_value = {gaussian_value} AND epoch = {epoch_gaussian} AND trace_process_id = {process_gaussian})
            OR (additive_noise_method_parameter_1_value = {collected_value} AND epoch = {epoch_collected} AND trace_process_id = {process_collected})
            OR (additive_noise_method_parameter_1_value = {rayleigh_value} AND epoch = {epoch_rayleigh} AND trace_process_id = {process_rayleigh})
        )
    order by
        additive_noise_method
        ;
    """
    wang = pd.read_sql_query(query1, con)
    con.close()

    wang.fillna("None", inplace=True)
    wang = wang.replace({"Collected": "Recorded"})
    wang = wang.rename(
        columns={
            # "termination_point": "Termination point",
            "additive_noise_method": "noise_method",
            "device": "Device",
            "epoch": "Epoch",
            "additive_noise_method_parameter_1_value": "parameter_value",
        },
    )
    g = wang.groupby(["noise_method", "Device"])
    # wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(
    wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    # wang = pd.DataFrame(wang)

    # ANOVA one way
    no_treatment = wang["termination_point"].where(
        wang["noise_method"] == "None"
    ).dropna().reset_index(drop=True)
    gaussian = wang["termination_point"].where(
        wang["noise_method"] == "Gaussian"
    ).dropna().reset_index(drop=True)
    recorded = wang["termination_point"].where(
        wang["noise_method"] == "Recorded"
    ).dropna().reset_index(drop=True)
    rayleigh = wang["termination_point"].where(
        wang["noise_method"] == "Rayleigh"
    ).dropna().reset_index(drop=True)
    # print(f_oneway(no_treatment, gaussian, recorded, rayleigh))
    new_data = {
        "None": no_treatment,
        "Gaussian": gaussian,
        "Recorded": recorded,
        "Rayleigh": rayleigh,
    }

    new_df = pd.DataFrame(data=new_data)
    print(new_df)

    # ANOVA, 4 sample
    """
    - Check sample sizes: equal number of observation in each group
    - Calculate Mean Square for each group (MS) (SS of group/level-1); level-1 is a degrees of freedom (df) for a group
    - Calculate Mean Square error (MSE) (SS error/df of residuals)
    - Calculate F value (MS of group/MSE)
    - Calculate p value based on F value and degrees of freedom (df)
    """
    assert no_treatment.size == gaussian.size == recorded.size == rayleigh.size

    # Multiple pairwise comparison (post-hoc test)
    res = stat()
    res.tukey_hsd(
        df=wang,
        res_var='termination_point',
        xfac_var='noise_method',
        anova_model='termination_point~C(noise_method)+C(parameter_value)+C(noise_method):C(parameter_value)'
    )
    # print(res.tukey_summary.to_latex())
    latex = res.tukey_summary.to_latex(
        # header=["M", "D", "M"],
        sparsify=True,
        float_format="%.3f",
        label=label + "_wang",
        escape=False,
        caption=f"Tukeys test table - {header}.",
        position="H",
    )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables",
        file_name + f"_wang.tex",
        )
    file = open(file_path, "w")
    file.write(latex)
    file.close()

    if histogram:
        # MPL styling
        plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
        # plt.rcParams.update({
        #     "ytick.labelsize": "xx-small",
        #     "xtick.labelsize": "xx-small",
        #     "axes.titlesize": "x-small",
        # })

        # Create 4 x 4 grid
        w, h = set_size(subplots=(2, 2), fraction=1)
        plt.figure(figsize=(w, h))
        plt.subplots_adjust(hspace=0.7, wspace=0.5)
        ax = plt.subplot(2, 2, 1)
        ax.hist(no_treatment, bins=10)
        ax.set_xlabel("None")
        ax = plt.subplot(2, 2, 2)
        ax.hist(gaussian, bins=10)
        ax.set_xlabel("Gaussian")
        ax = plt.subplot(2, 2, 3)
        ax.hist(recorded, bins=10)
        ax.set_xlabel("Recorded")
        ax = plt.subplot(2, 2, 4)
        ax.hist(rayleigh, bins=10)
        ax.set_xlabel("Rayleigh")
        plt.show()

    return wang


if __name__ == '__main__':

    # Scaling 1 - training set 500k
    header = "MaxMin (0, 1) over whole trace"
    trace_process_id = 3
    epoch_none = 10
    training_dataset = 'Wang_2021 - Cable, 5 devices, 500k traces'
    hypothesis_test_1(
        training_dataset=training_dataset,
        trace_process_id=trace_process_id,
        gaussian_value=0.04,
        collected_value=50,
        rayleigh_value=0.0276,
        epoch_none=epoch_none,
        epoch_gaussian=13,
        epoch_collected=18,
        epoch_rayleigh=17,
        # histogram=True
        file_name=f"stats_{trace_process_id}_training_500k",
        label=f"tbl:stats_{trace_process_id}",
        header=header,
    )
    #
    # # Scaling 1 - training set 200k
    # header = "MaxMin (0, 1) over whole trace"
    # trace_process_id = 3
    # epoch_none = 65
    # training_dataset = 'Wang_2021 - Cable, 5 devices, 200k traces'
    # hypothesis_test_1(
    #     training_dataset=training_dataset,
    #     trace_process_id=trace_process_id,
    #     gaussian_value=0.04,
    #     collected_value=25,
    #     rayleigh_value=0.0138,
    #     epoch_none=epoch_none,
    #     epoch_gaussian=65,
    #     epoch_collected=65,
    #     epoch_rayleigh=65,
    #     # histogram=True
    #     file_name=f"stats_{trace_process_id}_training_200k",
    #     label=f"tbl:stats_{trace_process_id}",
    #     header=header,
    # )

    # # Scaling 2
    # header = "MaxMin (0, 1) over Sbox"
    # trace_process_id = 4
    # epoch_none = 7
    # training_dataset = 'Wang_2021 - Cable, 5 devices, 500k traces'
    # hypothesis_test_1(
    #     training_dataset=training_dataset,
    #     trace_process_id=trace_process_id,
    #     epoch_none=epoch_none,
    #     gaussian_value=0.05,
    #     epoch_gaussian=16,
    #     collected_value=50,
    #     epoch_collected=7,
    #     rayleigh_value=0.0138,
    #     epoch_rayleigh=20,
    #     # histogram=True
    #     file_name=f"stats_{trace_process_id}",
    #     label=f"tbl:stats_{trace_process_id}",
    #     header=header,
    # )
    #
    # # Scaling 3
    # header = "MaxMin (-1, 1) over whole trace"
    # trace_process_id = 9
    # epoch_none = 6
    # hypothesis_test_1(
    #     trace_process_id=trace_process_id,
    #     gaussian_value=0.04,
    #     collected_value=25,
    #     rayleigh_value=0.0276,
    #     epoch_none=epoch_none,
    #     epoch_gaussian=11,
    #     epoch_collected=6,
    #     epoch_rayleigh=15,
    #     # histogram=True
    #     file_name=f"stats_{trace_process_id}",
    #     label=f"tbl:stats_{trace_process_id}",
    #     header=header,
    # )
    #
    # # Scaling 4
    # header = "MaxMin (-1, 1) over Sbox"
    # trace_process_id = 10
    # epoch_none = 6
    # training_dataset = 'Wang_2021 - Cable, 5 devices, 500k traces'
    # hypothesis_test_1(
    #     training_dataset=training_dataset,
    #     trace_process_id=trace_process_id,
    #     epoch_none=epoch_none,
    #     gaussian_value=0.05,
    #     epoch_gaussian=19,
    #     collected_value=50,
    #     epoch_collected=4,
    #     rayleigh_value=0.0138,
    #     epoch_rayleigh=6,
    #     # histogram=True
    #     file_name=f"stats_{trace_process_id}",
    #     label=f"tbl:stats_{trace_process_id}",
    #     header=header,
    # )
    #
    # # Scaling 5
    # header = "Standardization over Sbox"
    # trace_process_id = 8
    # epoch_none = 16
    # training_dataset = 'Wang_2021 - Cable, 5 devices, 500k traces'
    # hypothesis_test_1(
    #     training_dataset=training_dataset,
    #     trace_process_id=trace_process_id,
    #     epoch_none=epoch_none,
    #     gaussian_value=0.05,
    #     epoch_gaussian=17,
    #     collected_value=50,
    #     epoch_collected=8,
    #     rayleigh_value=0.0138,
    #     epoch_rayleigh=6,
    #     # histogram=True
    #     file_name=f"stats_{trace_process_id}",
    #     label=f"tbl:stats_{trace_process_id}",
    #     header=header,
    # )
    #
    # # Scaling 6
    # header = "Standardization over Sbox and Translation ±1"
    # trace_process_id = 12
    # epoch_none = 12
    # training_dataset = 'Wang_2021 - Cable, 5 devices, 500k traces'
    # hypothesis_test_1(
    #     training_dataset=training_dataset,
    #     trace_process_id=trace_process_id,
    #     epoch_none=epoch_none,
    #     gaussian_value=0.05,
    #     epoch_gaussian=13,
    #     collected_value=25,
    #     epoch_collected=12,
    #     rayleigh_value=0.0276,
    #     epoch_rayleigh=7,
    #     # histogram=True
    #     file_name=f"stats_{trace_process_id}",
    #     label=f"tbl:stats_{trace_process_id}",
    #     header=header,
    # )
    #
    # # Scaling 7
    # header = "Standardization over Sbox and Translation ±1"
    # trace_process_id = 13
    # epoch_none = 12
    # training_dataset = 'Wang_2021 - Cable, 5 devices, 500k traces'
    # hypothesis_test_1(
    #     training_dataset=training_dataset,
    #     trace_process_id=trace_process_id,
    #     epoch_none=epoch_none,
    #     gaussian_value=0.03,
    #     epoch_gaussian=17,
    #     collected_value=50,
    #     epoch_collected=11,
    #     rayleigh_value=0.0276,
    #     epoch_rayleigh=17,
    #     # histogram=True
    #     file_name=f"stats_{trace_process_id}",
    #     label=f"tbl:stats_{trace_process_id}",
    #     header=header,
    # )
    # hypothesis_test_best(
    #     header="Best additive noise methods",
    #     label=f"tbl:best_stats",
    # )
