import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from configs.variables import NORD_LIGHT_MPL_STYLE_PATH
from utils.db_utils import get_training_model_file_path


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
