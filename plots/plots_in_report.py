from configs.variables import REPORT_DIR
from plots.histogram_plots import plot_histogram_overview
from plots.history_log_plots import plot_history_log__overview_trace_process
from plots.noise_plots import plot_recorded_noise
from plots.rank_test_plots import plot_best_additive_noise_methods_2, \
    plot_epoch_comparison_report
from plots.trace_plots import plot_overview, plot_additive_noises_examples, \
    plot_example_normalized_training_trace
from utils.plot_utils import df_to_latex__additive

def additive_noise_example_plot():
    plot_additive_noises_examples(
        save_path=REPORT_DIR,
        format="pgf",
    )

def best_additive_plots_and_tables():
    trace_process_id = 3
    wang, zedigh = plot_best_additive_noise_methods_2(
        trace_process_id=trace_process_id,
        save_path=REPORT_DIR,
        format="pgf",
        show=True,
    )
    df_to_latex__additive(
        wang,
        zedigh,
        trace_process_id=3,
        file_name=f"best_additive_{trace_process_id}",
        label="tbl:best_additive",
    )

def normalized_trace_plots():
    plot_example_normalized_training_trace(
        training_dataset_id=3,
        trace_process_id=8,
        save_path=REPORT_DIR,
        file_format="pgf",
        show=True
    )

def histogram_plots():
    plot_histogram_overview(
        training_model_id=1,
        training_dataset_id=1,
        test_dataset_id=1,
        environment_id=1,
        trace_process_id=3,
        device=6,
        distance=15,
        epoch=65,
        save_path=REPORT_DIR,
        file_format="pgf",
        show=True,
    )

def history_log_plots():
    plot_history_log__overview_trace_process(
        training_dataset_id=1,
        trace_process_id=3,
        save_path=REPORT_DIR,
        file_format="pgf",
        show=True,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
    )

def epoch_comparison_plots():
    plot_epoch_comparison_report(
        training_model_id=1,
        training_dataset_id=1,
        test_dataset_id=1,
        trace_process_id=3,
        environment_id=1,
        distance=15,
        device=6,
        additive_noise_method_id=4,
        save_path=REPORT_DIR,
        format="pgf",
        # show=True
    )

    plot_epoch_comparison_report(
        training_model_id=1,
        training_dataset_id=1,
        test_dataset_id=1,
        trace_process_id=3,
        environment_id=1,
        distance=15,
        device=6,
        additive_noise_method_id=7,
        save_path=REPORT_DIR,
        format="pgf",
        # show=True
    )

    plot_epoch_comparison_report(
        training_model_id=1,
        training_dataset_id=1,
        test_dataset_id=1,
        trace_process_id=3,
        environment_id=1,
        distance=15,
        device=6,
        additive_noise_method_id=10,
        save_path=REPORT_DIR,
        format="pgf",
        # show=True
    )


if __name__ == "__main__":
    # additive_noise_example_plot()
    # best_additive_plots_and_tables()
    # normalized_trace_plots()
    # histogram_plots()
    # history_log_plots()
    # epoch_comparison_plots()
    plot_recorded_noise(file_format="pgf")
