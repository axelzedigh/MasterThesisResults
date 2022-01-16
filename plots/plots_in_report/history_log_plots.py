"""Plots for report."""
from configs.variables import REPORT_DIR
from plots.history_log_plots import plot_history_log__overview_trace_process

if __name__ == '__main__':
    # History log plots
    plot_history_log__overview_trace_process(
        training_dataset_id=3, # observe 1 or 3   !!!!!!!!!!!!!TODO
        trace_process_id=3,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=4,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=5,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=6,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=7,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=8,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=9,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=10,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=11,
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )
