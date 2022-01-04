"""Global variables in this project."""

import os

from matplotlib.lines import Line2D

VIEW_RANK_TEST_INDEX = {
    "id": 1,
    "test_dataset": 1,
    "training_dataset": 2,
    "environment": 3,
    "distance": 4,
    "device": 5,
    "training_model": 6,
    "keybyte": 7,
    "epoch": 8,
    "additive_noise_method": 9,
    "additive_param_1": 10,
    "additive_param_1_value": 11,
    "additive_param_2": 12,
    "additive_param_2_value": 13,
    "denoising_method": 14,
    "denoising_param_1": 15,
    "denoising_param_1_value": 16,
    "denoising_param_2": 17,
    "denoising_param_2_value": 18,
    "termination_point": 19,
    "trace_process_id": 20,
    "date_added": 21,
}

# Paths
PROJECT_DIR = os.getenv("MASTER_THESIS_RESULTS")
RAW_DATA_DIR = os.getenv("MASTER_THESIS_RESULTS_RAW_DATA")
REPORT_DIR = os.getenv("MASTER_THESIS_REPORT_DIR")

# Report variables
REPORT_TEXT_WIDTH = 369.88583

# Matplotlib styling variables
NORD_LIGHT_MPL_STYLE_PATH = os.path.join(
    PROJECT_DIR, "configs/matplotlib_styles", "nord-light.mplstyle"
)
NORD_LIGHT_MPL_STYLE_2_PATH = os.path.join(
    PROJECT_DIR, "configs/matplotlib_styles", "nord-light_2.mplstyle"
)
NORD_DARK_MPL_STYLE_PATH = os.path.join(
    PROJECT_DIR, "configs/matplotlib_styles", "nord-dark.mplstyle"
)
NORD_LIGHT_4_CUSTOM_LINES = [Line2D([0], [0], color='#5e81ac', lw=4),
                             Line2D([0], [0], color='#88c0d0', lw=4),
                             Line2D([0], [0], color='#bf616a', lw=4),
                             Line2D([0], [0], color='#d08770', lw=4)]
NORD_LIGHT_BLUE = '#5e81ac'
NORD_LIGHT_LIGHT_BLUE = '#88c0d0'
NORD_LIGHT_RED = '#bf616a'
NORD_LIGHT_ORANGE = '#d08770'
NORD_LIGHT_YELLOW = '#ebcb8b'
