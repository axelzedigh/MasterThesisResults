import os

import pandas as pd

from configs.variables import REPORT_TEXT_WIDTH, REPORT_DIR


def set_size(width_pt=REPORT_TEXT_WIDTH, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def interval_upper(x):
    a = 1.96
    mean = x.mean()
    sem = x.sem()
    return mean + a*sem


def interval_lower(x):
    a = 1.96
    mean = x.mean()
    sem = x.sem()
    return mean - a*sem


def df_to_latex__additive(
        wang: pd.DataFrame,
        zedigh: pd.DataFrame,
        trace_process_id: int,
        file_name: str,
        label: str,
        table_type: str = "per_device",
        header: str = "",
):
    """Store dataframe to latex table"""
    # Wang
    wang = wang.rename(
        columns={
            "termination_point": "Termination point",
            "additive_noise_method": "Noise method",
            "device": "Device",
            "epoch": "Epoch",
            "Additive parameter 1": "Parameter Value",
        }
    )

    if table_type == "per_device":
        wang = wang.groupby(["Noise method", "Device"]) \
            .agg(
            # .head(100).reset_index(drop=True).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower, interval_upper]})
        wang = wang.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        wang = wang.reset_index()
        latex = wang.to_latex(
            # header=["M", "D", "M"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_wang",
            escape=False,
            caption=f"Table for Wang 2021 dataset - {header}",
            position="H",
        )
    elif table_type == "per_additive_method":
        g = wang.groupby(["Noise method", "Device"])
        # wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(
        wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
        wang = wang.groupby(["Noise method", "Epoch", "Parameter Value"]).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower,
                                   interval_upper]})
        wang = wang.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        latex = wang.to_latex(
            # header=["M", "D", "M"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_wang",
            escape=False,
            caption=f"Table for Wang 2021 dataset - {header}",
            position="H",
        )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables/{trace_process_id}",
        file_name + f"_{table_type}_wang.tex",
    )
    file = open(file_path, "w")
    file.write(latex)
    file.close()

    # Zedigh
    zedigh = zedigh.rename(
        columns={
            "termination_point": "Termination point",
            "additive_noise_method": "Noise method",
            "device": "Device",
            "epoch": "Epoch",
            "Additive parameter 1": "Parameter Value",
        }
    )
    if table_type == "per_device":
        zedigh = zedigh.groupby(["Noise method", "Device"]) \
            .agg(
            # .head(100).reset_index(drop=True).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower, interval_upper]})
        zedigh = zedigh.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        latex = zedigh.to_latex(
            # header=["M", "D", "M"],
            # columns=["Noise method", "Device", "mean", "std", "count"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_zedigh",
            escape=False,
            caption=f"Table for Zedigh 2021 dataset - {header}",
            position="H",
        )
    elif table_type == "per_additive_method":
        g = zedigh.groupby(["Noise method", "Device"])
        zedigh = g.apply(lambda x: x.sample(g.size().min())).reset_index(
            drop=True)
        zedigh = zedigh.groupby(["Noise method", "Epoch", "Parameter Value"]).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower,
                                   interval_upper]})
        zedigh = zedigh.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        zedigh = zedigh.reset_index()
        latex = zedigh.to_latex(
            # header=["M", "D", "M"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_wang",
            escape=False,
            caption=f"Table for Zedigh 2021 dataset - {header}",
            position="H",
        )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables/{trace_process_id}",
        file_name + f"_{table_type}_zedigh.tex",
        )
    file = open(file_path, "w")
    file.write(latex)
    file.close()


def df_to_latex__denoising(
        wang: pd.DataFrame,
        zedigh: pd.DataFrame,
        trace_process_id: int,
        file_name: str,
        label: str,
        table_type: str = "per_device",
):
    """Store dataframe to latex table"""
    # Wang
    wang = wang.rename(
        columns={
            "termination_point": "Termination point",
            "denoising_method": "Denoising method",
            "device": "Device",
            "epoch": "Epoch",
            "Denoising parameter 1": "Parameter Value",
        }
    )

    if table_type == "per_device":
        wang = wang.groupby(["Denoising method", "Device"]) \
            .agg(
            # .head(100).reset_index(drop=True).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower, interval_upper]})
        wang = wang.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        wang = wang.reset_index()
        latex = wang.to_latex(
            # header=["M", "D", "M"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_wang",
            escape=False,
            caption=f"Table for Wang 2021 dataset (trace process {trace_process_id})",
            position="H",
        )
    elif table_type == "per_denoising":
        g = wang.groupby(["Denoising method", "Device"])
        # wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(
        wang = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
        wang = wang.groupby(["Denoising method", "Epoch", "Parameter Value"]).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower,
                                   interval_upper]})
        wang = wang.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        latex = wang.to_latex(
            # header=["M", "D", "M"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_wang",
            escape=False,
            caption=f"Table for Wang 2021 dataset (trace process {trace_process_id})",
            position="H",
        )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables/{trace_process_id}",
        file_name + f"_{table_type}_wang.tex",
        )
    file = open(file_path, "w")
    file.write(latex)
    file.close()

    # Zedigh
    zedigh = zedigh.rename(
        columns={
            "termination_point": "Termination point",
            "denoising_method": "Denoising method",
            "device": "Device",
            "epoch": "Epoch",
            "Denoising parameter 1": "Parameter Value",
        }
    )
    if table_type == "per_device":
        zedigh = zedigh.groupby(["Noise method", "Device"]) \
            .agg(
            # .head(100).reset_index(drop=True).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower, interval_upper]})
        zedigh = zedigh.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        latex = zedigh.to_latex(
            # header=["M", "D", "M"],
            # columns=["Noise method", "Device", "mean", "std", "count"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_zedigh",
            escape=False,
            caption=f"Table for Zedigh 2021 dataset (trace process {trace_process_id})",
            position="H",
        )
    elif table_type == "per_denoising":
        g = zedigh.groupby(["Denoising method", "Device"])
        zedigh = g.apply(lambda x: x.sample(g.size().min())).reset_index(
            drop=True)
        zedigh = zedigh.groupby(["Denoising method", "Epoch", "Parameter Value"]).agg(
            {'Termination point': ['mean', 'std', 'count', interval_lower,
                                   interval_upper]})
        zedigh = zedigh.rename(
            columns={
                "mean": "Mean",
                "std": "$\sigma$",
                "count": "Count",
                "interval_lower": "$CI_{-}$",
                "interval_upper": "$CI_{+}$",
            }
        )
        zedigh = zedigh.reset_index()
        latex = zedigh.to_latex(
            # header=["M", "D", "M"],
            sparsify=True,
            float_format="%.0f",
            label=label + "_wang",
            escape=False,
            caption=f"Table for Wang 2021 dataset (trace process {trace_process_id})",
            position="H",
        )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables/{trace_process_id}",
        file_name + f"_{table_type}_zedigh.tex",
        )
    file = open(file_path, "w")
    file.write(latex)
    file.close()
