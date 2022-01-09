"""Plots of the additive noises."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from scipy import signal

from configs.variables import NORD_LIGHT_MPL_STYLE_2_PATH, NORD_LIGHT_RED, \
    REPORT_DIR
from utils.plot_utils import set_size
from utils.trace_utils import get_training_trace_path
from utils.training_utils import cut_trace_set__column_range, \
    additive_noise_to_trace_set, denoising_of_trace_set


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


def plot_training_diff_psd(
        training_dataset_id: int = 3,
        additive_noise_method_id: Optional[int] = None,
        denoising_method_id: Optional[int] = None,
        save_path: Optional[str] = REPORT_DIR,
        file_format: str = "png",
        show: bool = False,
):
    """
    Plots the diff (trace process X) and it's corresponding psd.
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
    w, h = set_size(subplots=(3, 1), fraction=1)
    fig = plt.figure(constrained_layout=True, figsize=(w, h))
    gs = GridSpec(7, 1, figure=fig)
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[1, 0])
    ax1c = fig.add_subplot(gs[2, 0])
    ax1d = fig.add_subplot(gs[3, 0])
    ax1e = fig.add_subplot(gs[4, 0])
    ax2 = fig.add_subplot(gs[5, 0])
    ax3 = fig.add_subplot(gs[6, 0])

    trace_process_id = 11
    if training_dataset_id in [2, 3]:
        if trace_process_id == 14:
            start = 130 - 4
            end = 240 - 4
        else:
            start = 130
            end = 240
    else:
        if trace_process_id == 14:
            start = 200
            end = 310
        else:
            start = 204
            end = 314

    training_set_path = get_training_trace_path(training_dataset_id)
    trace_set_file_path = os.path.join(
        training_set_path, "trace_process_8-standardization_sbox.npy"
    )
    training_trace_set = np.load(trace_set_file_path)
    ax3.plot(np.mean(training_trace_set, axis=0))

    N = 5
    fc = 8e5
    fs = 5e8
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq

    # HP
    sos = signal.butter(N, fc, 'hp', fs=fs, output='sos')
    training_trace_set = signal.sosfilt(sos, training_trace_set)

    # High-pass (8e5)
    b, a = signal.butter(N, normal_cutoff, btype='high', analog=False)
    training_trace_set = signal.filtfilt(b, a, training_trace_set)

    # Low-pass (1.4e6)
    h = signal.firwin(numtaps=N, cutoff=14e5/nyq, nyq=fs/2)
    training_trace_set = signal.lfilter(h, 1.0, training_trace_set)

    ax3.plot(np.mean(training_trace_set, axis=0))

    # Apply additive noise
    if additive_noise_method_id is not None:
        training_trace_set, additive_noise_trace = additive_noise_to_trace_set(
            trace_set=training_trace_set,
            additive_noise_method_id=additive_noise_method_id
        )

    # Denoise the trace set.
    if denoising_method_id is not None:
        training_trace_set, start, end, clean_trace = denoising_of_trace_set(
            trace_set=training_trace_set,
            denoising_method_id=denoising_method_id,
            training_dataset_id=training_dataset_id,
        )

    training_trace_set -= np.mean(training_trace_set, axis=0)
    training_trace_set = cut_trace_set__column_range(
        trace_set=training_trace_set,
        range_start=start - 20,
        range_end=end + 20,
    )
    windowed_training_trace_set = training_trace_set * np.hanning(training_trace_set.shape[1])

    ax1a.plot(windowed_training_trace_set[0])
    ax1a.plot(windowed_training_trace_set[1])
    ax1a.plot(windowed_training_trace_set[2])
    #ax1a.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1b.plot(windowed_training_trace_set[110000])
    ax1b.plot(windowed_training_trace_set[110001])
    ax1b.plot(windowed_training_trace_set[110002])
    ax1b.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1c.plot(windowed_training_trace_set[220000])
    ax1c.plot(windowed_training_trace_set[220001])
    ax1c.plot(windowed_training_trace_set[220002])
    ax1c.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1d.plot(windowed_training_trace_set[330000])
    ax1d.plot(windowed_training_trace_set[330001])
    ax1d.plot(windowed_training_trace_set[330002])
    ax1d.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1e.plot(windowed_training_trace_set[-3])
    ax1e.plot(windowed_training_trace_set[-2])
    ax1e.plot(windowed_training_trace_set[-1])
    ax1e.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.psd(windowed_training_trace_set.flatten(), Fs=5e6)

    if save_path:
        path = os.path.join(
            save_path,
            f"figures/noise",
            f"training_trace_diff__set_{training_dataset_id}.{file_format}",
        )
        plt.savefig(path)
    if show:
        plt.show()
