import os
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH, \
    NORD_LIGHT_MPL_STYLE_2_PATH, REPORT_DIR, NORD_LIGHT_RED, NORD_LIGHT_YELLOW, \
    NORD_LIGHT_4_CUSTOM_LINES
from utils.db_utils import get_training_model_file_path
from utils.plot_utils import set_size


def plot_history_log(
        training_dataset_id: int,
        trace_process_id: int,
        keybyte: int,
        additive_noise_method_id: Optional[int],
        denoising_method_id: Optional[int],
        save: bool = False,
        show: bool = True,
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
        training_dataset_id=training_dataset_id,
    )
    training_path = os.path.dirname(training_file_path)
    history_log_file_path = os.path.join(training_path, "history_log.npy")
    history_log_fig_file_path = os.path.join(training_path, "history_log.png")
    history_log_npy = np.load(history_log_file_path, allow_pickle=True)
    history_log = history_log_npy.tolist()

    # Setup plt
    plt.style.use(NORD_LIGHT_MPL_STYLE_PATH)
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle("Accuracy & Loss", fontsize=18, y=0.95)

    # Subplot 1 - Accuracy
    ax1 = fig.add_axes((0.1, 0.1, 0.35, 0.8))
    try:
        ax1.plot(history_log["accuracy"], solid_capstyle="round", linewidth=2,
                 label="Accuracy")
    except:
        pass
    try:
        ax1.plot(history_log["categorical_accuracy"], solid_capstyle="round",
                 linewidth=2, label="Accuracy")
    except:
        pass
    try:
        ax1.plot(history_log["val_accuracy"], solid_capstyle="round",
                 label="Validation Accuracy")
    except:
        pass
    try:
        ax1.plot(history_log["val_categorical_accuracy"],
                 solid_capstyle="round", label="Validation Accuracy")
    except:
        pass
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    # Subplot 2 - Loss
    ax2 = fig.add_axes((0.55, 0.1, 0.35, 0.8))
    ax2.plot(history_log["loss"], linewidth=2, label="Loss")
    ax2.plot(history_log["val_loss"], label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax1.legend()
    ax2.legend()
    if save:
        plt.savefig(fname=history_log_fig_file_path)
    if show:
        plt.show()


def plot_history_log__overview_trace_process(
        training_dataset_id: int = 1,
        trace_process_id: int = 3,
        save_path: str = REPORT_DIR,
        file_format: str = "png",
        show: bool = False,
        last_gaussian: int = 5,
        last_collected: int = 9,
        last_rayleigh: int = 11,
        nrows: int = 4,
        ncols: int = 4,
) -> None:
    """
    Plot the history function. Accuracy on the left, loss on the right.
    """

    # MPL styling
    plt.style.use(NORD_LIGHT_MPL_STYLE_2_PATH)
    custom_lines = NORD_LIGHT_4_CUSTOM_LINES
    plt.rcParams.update({
        "ytick.labelsize": "xx-small",
        "xtick.labelsize": "xx-small",
        "axes.titlesize": "x-small",
        "grid.alpha": "0.25",
    })
    labels = [
        "Accuracy",
        f"Validation accuracy",
        f"Loss",
        f"Validation loss"
    ]

    # Create 4 x 4 grid
    w, h = set_size(subplots=(nrows, 2), fraction=1)
    fig = plt.figure(figsize=(w, h))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    additive_noise_method_ids = ["None", 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    i = ncols + 1
    j = ncols * 2 + 1
    k = ncols * 3 + 1
    tot = 0
    for additive_noise in additive_noise_method_ids:
        tot += 1
        try:
            training_file_path = get_training_model_file_path(
                database="main.db",
                training_model_id=1,
                additive_noise_method_id=additive_noise,
                denoising_method_id=None,
                epoch=1,
                keybyte=0,
                trace_process_id=trace_process_id,
                training_dataset_id=training_dataset_id,
            )
            training_path = os.path.dirname(training_file_path)
            history_log_file_path = os.path.join(training_path, "history_log.npy")
            history_log_npy = np.load(history_log_file_path, allow_pickle=True)
            history_log = history_log_npy.tolist()
            history_log = pd.DataFrame(history_log)
            if additive_noise == "None":
                ax1 = plt.subplot(nrows, ncols, 1)
                ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
                ax1.set_ylabel("No noise")
            elif additive_noise in [1, 2, 3, 4, 5]:
                ax1 = plt.subplot(nrows, ncols, i)
                ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
                if i != ncols + 1:
                    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
                else:
                    ax1.set_ylabel("GWN")
                i += 1
            elif additive_noise in [6, 7, 8, 9]:
                ax1 = plt.subplot(nrows, ncols, j)
                ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
                if j != ncols * 2 + 1:
                    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
                else:
                    ax1.set_ylabel("Collected")
                j += 1
            elif additive_noise in [10, 11]:
                ax1 = plt.subplot(nrows, ncols, k)
                ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
                if k != ncols * 3 + 1:
                    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
                else:
                    ax1.set_ylabel("Rayleigh")
                k += 1
            try:
                history_log["accuracy"] = history_log["accuracy"].apply(lambda x: x*100)
                ax1.plot(history_log["accuracy"], solid_capstyle="round",
                        label="Accuracy")
            except:
                pass
            try:
                history_log["categorical_accuracy"] = history_log["categorical_accuracy"].apply(lambda x: x*100)
                ax1.plot(history_log["categorical_accuracy"],
                         solid_capstyle="round", label="Accuracy")
            except:
                pass
            try:
                history_log["val_accuracy"] = history_log["val_accuracy"].apply(lambda x: x*100)
                ax1.plot(history_log["val_accuracy"], solid_capstyle="round",
                        label="Validation Accuracy")
            except:
                pass
            try:
                history_log["val_categorical_accuracy"] = history_log["val_categorical_accuracy"].apply(lambda x: x*100)
                ax1.plot(history_log["val_categorical_accuracy"],
                        solid_capstyle="round", label="Validation Accuracy")
            except:
                pass

            ax2 = ax1.twinx()
            ax2.plot(history_log["loss"], label="Loss", color=NORD_LIGHT_RED)
            ax2.plot(history_log["val_loss"], label="Validation Loss", color=NORD_LIGHT_YELLOW)
            ax2.yaxis.set_major_formatter(ticker.NullFormatter())
            if additive_noise in ["None", last_gaussian, last_collected, last_rayleigh]:
                ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

            if additive_noise == 1:
                ax1.set_title("$\sigma$=0.01")
            elif additive_noise == 3:
                ax1.set_title("$\sigma$=0.03")
            elif additive_noise == 4:
                ax1.set_title("$\sigma$=0.04")
            elif additive_noise == 5:
                ax1.set_title("$\sigma$=0.05")
            elif additive_noise == 6:
                ax1.set_title("scaling=25")
            elif additive_noise == 7:
                ax1.set_title("scaling=50")
            elif additive_noise == 8:
                ax1.set_title("scaling=75")
            elif additive_noise == 9:
                ax1.set_title("scaling=105")
            elif additive_noise == 10:
                ax1.set_title("mode=0.0138")
            elif additive_noise == 11:
                ax1.set_title("mode=0.0276")

            ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
            ax1.set_ylim(0, 2.5)
            ax2.set_ylim(4.6, 5.6)
        except:
            pass

    fig.legend(
        handles=custom_lines,
        labels=labels,
        # bbox_to_anchor=(0.925, 0.9),
        bbox_to_anchor=(0.975, 0.975),
        loc=1,
        # mode="expand",
        ncol=1,
    )
    # plt.tight_layout()
    fig.tight_layout()
    if save_path:
        path = os.path.join(
            save_path,
            f"figures/{trace_process_id}",
            f"acc_loss__training_set_{training_dataset_id}.{file_format}",
        )
        plt.savefig(path)
    if show:
        plt.show()

