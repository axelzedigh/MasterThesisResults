"""Plots for report."""
from configs.variables import REPORT_DIR
from plots.histogram_plots import plot_histogram_overview
from plots.rank_test_plots import plot_best_additive_noise_methods_2, \
    plot_epoch_comparison_report, plot_all_of_an_additive_noise__report__2
from plots.trace_plots import plot_example_normalized_training_trace_1_row
from utils.plot_utils import df_to_latex__additive

if __name__ == '__main__':
    # Feature scaling 4 - MaxMin(-1, 1) over Sbox
    # Trace process id 10
    trace_process_id = 13
    plot_example_normalized_training_trace_1_row(
        training_dataset_id=3, trace_process_id=trace_process_id,
        file_format="pgf", show=False, denoising_method_id=None
    )

    # plot_epoch_comparison_report(
    #     training_model_id=1,
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=trace_process_id,
    #     environment_id=1,
    #     distance=15,
    #     device=10,
    #     additive_noise_method_id=5,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    #     y_bottom=100,
    #     y_top=400,
    # )

    epoch_none = 12
    plot_all_of_an_additive_noise__report__2(
        trace_process_id=trace_process_id,
        additive_noise_method='Gaussian',
        parameter_1_value_1=0.01,
        parameter_1_value_2=0.02,
        parameter_1_value_3=0.04,
        parameter_1_value_4=0.05,
        epoch_none=epoch_none,
        epoch_1=0,
        epoch_2=12,
        epoch_3=12,
        epoch_4=0,
        # show=True,
        y_top=1000,
        row_size=1,
        x_label=False,
        y_label_subtext="Gaussian",
        labels=[
            f"None $e_{{{epoch_none}}}$",
            f"0.03 $e_{{{12}}}$",
            f"0.04 $e_{{{12}}}$",
        ],
    )

    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Collected',
    #     parameter_1_value_1=25,
    #     parameter_1_value_2=50,
    #     parameter_1_value_3=75,
    #     parameter_1_value_4=105,
    #     epoch_none=epoch_none,
    #     epoch_1=0,
    #     epoch_2=0,
    #     epoch_3=0,
    #     epoch_4=0,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    #     x_label=False,
    #     y_label_subtext="Recorded",
    #     labels=[
    #         f"None $e_{{{epoch_none}}}$",
    #         f"25 $e_{{{9}}}$",
    #         f"50 $e_{{{8}}}$",
    #     ],
    # )

    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Rayleigh',
    #     parameter_1_value_1=0.0138,
    #     parameter_1_value_2=0.0276,
    #     parameter_1_value_3=0,
    #     parameter_1_value_4=0,
    #     epoch_none=epoch_none,
    #     epoch_1=0,
    #     epoch_2=0,
    #     epoch_3=0,
    #     epoch_4=0,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    #     x_label=False,
    #     y_label_subtext="Rayleigh",
    #     labels=[
    #         f"None $e_{{{epoch_none}}}$",
    #         f"0.0138 $e_{{{6}}}$",
    #         f"0.0276 $e_{{{15}}}$",
    #     ],
    # )

    wang, zedigh = plot_best_additive_noise_methods_2(
        training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
        trace_process_id=trace_process_id,
        save_path=REPORT_DIR,
        file_format="pgf",
        gaussian_value=0.02,
        collected_value=50,
        rayleigh_value=0.0138,
        epoch_none=epoch_none,
        epoch_gaussian=12,
        epoch_collected=0,
        epoch_rayleigh=0,
        # show=True,
        y_top=1000,
        row_size=1,
        x_label=True,
        y_label_subtext="Best Additive Noise",
        custom_labels=True
    )

    df_to_latex__additive(
        wang,
        zedigh,
        trace_process_id=trace_process_id,
        file_name=f"best_additive",
        label=f"tbl:best_additive_{trace_process_id}_per_additive",
        table_type="per_additive_method",
    )

    df_to_latex__additive(
        wang,
        zedigh,
        trace_process_id=trace_process_id,
        file_name=f"best_additive",
        label=f"tbl:best_additive__per_device",
        table_type="per_device",
    )

    # plot_histogram_overview(
    #     training_model_id=1,
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=6,
    #     distance=15,
    #     epoch=65,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )
