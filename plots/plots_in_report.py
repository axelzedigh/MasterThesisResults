from configs.variables import REPORT_DIR
from plots.histogram_plots import plot_histogram_overview
from plots.history_log_plots import plot_history_log__overview_trace_process
from plots.noise_plots import plot_recorded_noise, plot_training_diff_psd
from plots.rank_test_plots import plot_best_additive_noise_methods_2, \
    plot_epoch_comparison_report, plot_all_of_an_additive_noise__report
from plots.trace_plots import plot_overview, plot_additive_noises_examples, \
    plot_example_normalized_training_trace
from utils.plot_utils import df_to_latex__additive


def additive_noise_example_plot():
    plot_additive_noises_examples(
        save_path=REPORT_DIR,
        format="pgf",
    )


def best_additive_plots_and_tables(
        trace_process_id: int = 3,
        gaussian: float = 0.04,
        collected: float = 25,
        rayleigh: float = 0.0138,
        epoch: int = 65,
        training_dataset: str ="Wang_2021 - Cable, 5 devices, 500k traces",
):
    wang, zedigh = plot_best_additive_noise_methods_2(
        training_dataset=training_dataset,
        trace_process_id=trace_process_id,
        save_path=REPORT_DIR,
        format="pgf",
        show=True,
        gaussian_value=gaussian,
        collected_value=collected,
        rayleigh_value=rayleigh,
        epoch=epoch,
    )
    df_to_latex__additive(
        wang,
        zedigh,
        trace_process_id=trace_process_id,
        file_name=f"best_additive",
        label=f"tbl:best_additive_{trace_process_id}_per_additive",
        # table_type="per_device",
        table_type="per_additive_method",
    )


def additive_noise_comparision_plot(trace_process_id: int = 3):
    plot_all_of_an_additive_noise__report(
        training_dataset='Wang_2021 - Cable, 5 devices, 200k traces',
        additive_noise_method="Gaussian",
        trace_process_id=trace_process_id,
        epoch=65,
        distance=15,
        environment="office_corridor",
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
    )

    plot_all_of_an_additive_noise__report(
        training_dataset='Wang_2021 - Cable, 5 devices, 200k traces',
        additive_noise_method="Collected",
        trace_process_id=trace_process_id,
        epoch=65,
        distance=15,
        environment="office_corridor",
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
    )

    plot_all_of_an_additive_noise__report(
        training_dataset='Wang_2021 - Cable, 5 devices, 200k traces',
        additive_noise_method="Rayleigh",
        trace_process_id=trace_process_id,
        epoch=65,
        distance=15,
        environment="office_corridor",
        save_path=REPORT_DIR,
        file_format="pgf",
        # show=True,
    )


def normalized_trace_plots(trace_process_id: int = 3):
    plot_example_normalized_training_trace(
        training_dataset_id=3,
        trace_process_id=trace_process_id,
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


def history_log_plots(
        training_dataset_id: int = 3,
        trace_process_id : int = 3
):
    plot_history_log__overview_trace_process(
        training_dataset_id=training_dataset_id,
        trace_process_id=trace_process_id,
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
        additive_noise_method_id=10,
        save_path=REPORT_DIR,
        format="pgf",
        # show=True
    )


if __name__ == "__main__":
    training_dataset_id = 3
    test_dataset_id = 1
    trace_process_id = 12
    environment_id = 1
    distance = 15
    device = 6
    additive_noise_method = 10

    # additive_noise_example_plot()
    # normalized_trace_plots(trace_process_id=trace_process_id)
    history_log_plots(training_dataset_id=3, trace_process_id=12)
    # epoch_comparison_plots()
    # plot_epoch_comparison_report(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     trace_process_id=trace_process_id,
    #     environment_id=environment_id,
    #     distance=distance,
    #     device=10,
    #     additive_noise_method_id=3,
    #     save_path=REPORT_DIR,
    #     format="pgf",
    #     show=True
    # )

    # best_additive_plots_and_tables(trace_process_id=trace_process_id, gaussian=0.04, epoch=12)
    # additive_noise_comparision_plot()
    # histogram_plots()
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
