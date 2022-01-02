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


def df_to_latex__additive(
        wang: pd.DataFrame,
        zedigh: pd.DataFrame,
        trace_process_id: int,
        file_name: str,
        label: str,
):
    """Store dataframe to latex table"""
    # Wang
    wang = wang.rename(
        columns={
            "termination_point": "Termination point",
            "additive_noise_method": "Noise method",
            "device": "Device",
        }
    )
    wang = wang.groupby(["Noise method", "Device"]).agg(
        {'Termination point': ['mean', 'std', 'count']})
    latex = wang.to_latex(
        # header=["M", "D", "M"],
        sparsify=True,
        float_format="%.0f",
        label=label + "_wang",
        caption=f"Table for Wang 2021 dataset (trace process {trace_process_id})"
    )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables/{trace_process_id}",
        file_name + "_wang.tex",
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
        }
    )
    zedigh = zedigh.groupby(["Noise method", "Device"]).agg(
        {'Termination point': ['mean', 'std', 'count']})
    latex = zedigh.to_latex(
        # header=["M", "D", "M"],
        # columns=["Noise method", "Device", "mean", "std", "count"],
        sparsify=True,
        float_format="%.0f",
        label=label + "_zedigh",
        caption=f"Table for Zedigh 2021 dataset (trace process {trace_process_id})"
    )

    file_path = os.path.join(
        REPORT_DIR,
        f"tables/{trace_process_id}",
        file_name + "_zedigh.tex",
        )
    file = open(file_path, "w")
    file.write(latex)
    file.close()
