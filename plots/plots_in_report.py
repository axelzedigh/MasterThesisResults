from configs.variables import REPORT_DIR
from plots.histogram_plots import plot_histogram_overview
from plots.history_log_plots import plot_history_log__overview_trace_process
from plots.rank_test_plots import plot_best_additive_noise_methods_2
from plots.trace_plots import plot_overview, plot_additive_noises_examples, \
    plot_example_normalized_training_trace
from utils.plot_utils import df_to_latex__additive


if __name__ == "__main__":
    # Example additive noise plots
    # plot_additive_noises_examples(
    #     save_path=REPORT_DIR,
    #     format="pgf",
    # )

    # Best additive noise methods
    trace_process_id = 3
    wang, zedigh = plot_best_additive_noise_methods_2(
        trace_process_id=trace_process_id,
        save_path=REPORT_DIR,
        format="pgf",
        # show=True,
    )
    df_to_latex__additive(
        wang,
        zedigh,
        trace_process_id=3,
        file_name=f"best_additive_{trace_process_id}",
        label="tbl:best_additive",
    )

    # plot_example_normalized_training_trace(
    #     training_dataset_id=3,
    #     trace_process_id=8,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True
    # )

    # plot_histogram_overview(
    #     training_dataset_id=1,
    #     test_dataset_id=1,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=10,
    #     distance=15,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=8,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )
