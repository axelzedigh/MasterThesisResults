"""Plots of the additive noises."""
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.gridspec import GridSpec

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
    w, h = set_size(subplots=(2, 2), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(2, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0:1, 0])
    ax2 = fig.add_subplot(gs[1:2, 0])

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
    ax1.hist(
        noise_traces_fixed, bins=100, density=True, histtype="stepfilled"
    )
    ax1.hist(
        rayleigh_noise_arr.flatten(), bins=100, density=True, histtype="step",
        color=NORD_LIGHT_RED, rwidth=4
            )
    ax1.set_xlabel("Sample data amplitude")
    ax1.set_ylabel("Frequency")
    ax1.legend(labels=["Recorded noise", "Rayleigh, mode=0.00055"], loc=1)

    ax2.psd(x=noise_traces_fixed, Fs=5e6)
    ax2.set_xlim(0, 100000)

    # plt.tight_layout()
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/noise",
            f"recorded_and_rayleigh_noise.{file_format}",
        )
        plt.savefig(path)
    if show:
        plt.show()
