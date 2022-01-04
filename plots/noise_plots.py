"""Plots of the additive noises."""
import matplotlib.pyplot as plt
import numpy as np
import os

from configs.variables import NORD_LIGHT_MPL_STYLE_2_PATH, NORD_LIGHT_RED, \
    REPORT_DIR
from utils.plot_utils import set_size


def plot_recorded_noise(
        save_path: str = REPORT_DIR,
        file_format: str = "png",
        show: bool = False,
):
    """
    Plots the recorded noise from the office hall.
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    plt.rcParams.update({
        "ytick.labelsize": "xx-small",
        "xtick.labelsize": "xx-small",
        "axes.titlesize": "x-small",
        "patch.force_edgecolor": "False",
        "legend.framealpha": "1",
    })

    # MLP fig/ax
    w, h = set_size(subplots=(1, 2), fraction=1)
    fig = plt.figure(figsize=(w, h))
    ax = fig.gca()

    # Get noise data
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    noise_dir = os.path.join(
        project_dir,
        "datasets/test_traces/Zedigh_2021/office_corridor/Noise/data"
    )
    noise_traces_file_path = os.path.join(noise_dir, "traces.npy")
    noise_traces_npy = np.load(noise_traces_file_path)

    # Fix outliers
    index_1 = int(512400 / 400)
    index_2 = int(1824800 / 400)
    traces_fixed = noise_traces_npy.copy()
    traces_fixed[index_1] = traces_fixed[index_1 - 20]
    traces_fixed[index_2] = traces_fixed[index_2 - 20]
    noise_traces_fixed = traces_fixed.flatten()

    # Rayleigh noise
    rayleigh_noise_set = []
    for i in range(4000):
        rayleigh_noise = np.random.rayleigh(0.00055, 110)
        rayleigh_noise_set.append(rayleigh_noise)
    rayleigh_noise_arr = np.array(rayleigh_noise_set)

    # Plot
    ax.hist(
        noise_traces_fixed, bins=100, density=True, histtype="stepfilled"
    )
    ax.hist(
        rayleigh_noise_arr.flatten(), bins=100, density=True, histtype="step",
        color=NORD_LIGHT_RED, rwidth=4
            )
    ax.set_xlabel("Noise sample amplitude")
    ax.set_ylabel("Count")
    ax.legend(labels=["Recorded noise", "Rayleigh, mode=0.00055"], loc=1)
    plt.tight_layout()
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/noise",
            f"recorded_and_rayleigh_noise.{file_format}",
        )
        plt.savefig(path)
    if show:
        plt.show()
