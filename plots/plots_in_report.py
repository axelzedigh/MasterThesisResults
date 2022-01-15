"""Plots for report."""
from configs.variables import REPORT_DIR
from plots.histogram_plots import plot_histogram_overview
from plots.history_log_plots import plot_history_log__overview_trace_process
from plots.rank_test_plots import plot_best_additive_noise_methods_2, \
    plot_epoch_comparison_report, plot_all_of_an_additive_noise__report, \
    plot_all_of_a_denoising_method__report, \
    plot_all_of_an_additive_noise__report__2
from plots.trace_plots import plot_example_normalized_training_trace_1_row
from utils.plot_utils import df_to_latex__additive

if __name__ == "__main__":
    # Additive noise example
    # additive_noise_example_plot()
    # plot_recorded_noise(file_format="pgf", show=True)
    # plot_training_diff_psd(
    #     training_dataset_id=3,
    #     additive_noise_method_id=None,
    #     denoising_method_id=None,
    #     #save_path=REPORT_DIR,
    #     save_path=None,
    #     file_format="pgf",
    #     show=True
    # )

    # Denoising
    # plot_all_of_a_denoising_method__report(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     denoising_method="Moving Average Filter",
    #     trace_process_id=trace_process_id,
    #     epoch=epoch,
    #     distance=distance,
    #     environment="office_corridor",
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    # )
