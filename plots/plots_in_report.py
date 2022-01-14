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

    # # Feature scaling 1 - MaxMin (0, 1) over whole trace
    # trace_process_id = 3
    # training_dataset_id = 1
    # epoch = 65
    # distance = 15
    # test_dataset_id = 1
    # environment_id = 1
    # device = 10
    # additive_noise_method = 4
    # # plot_example_normalized_training_trace(
    # #     training_dataset_id=3, trace_process_id=trace_process_id,
    # #     file_format="pgf", show=False, denoising_method_id=None
    # # )
    # plot_example_normalized_training_trace_1_row(
    #     training_dataset_id=3, trace_process_id=trace_process_id,
    #     file_format="pgf", show=False, denoising_method_id=None
    # )
    # plot_history_log__overview_trace_process(
    #     training_dataset_id=training_dataset_id,
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=False,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )
    # plot_epoch_comparison_report(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     trace_process_id=trace_process_id,
    #     environment_id=environment_id,
    #     distance=distance,
    #     device=device,
    #     additive_noise_method_id=additive_noise_method,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True
    # )
    # plot_all_of_an_additive_noise__report(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     additive_noise_method="Gaussian",
    #     trace_process_id=trace_process_id,
    #     epoch=epoch,
    #     distance=distance,
    #     environment="office_corridor",
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )
    # plot_all_of_an_additive_noise__report(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     additive_noise_method="Collected",
    #     trace_process_id=trace_process_id,
    #     epoch=epoch,
    #     distance=distance,
    #     environment="office_corridor",
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )
    # plot_all_of_an_additive_noise__report(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     additive_noise_method="Rayleigh",
    #     trace_process_id=trace_process_id,
    #     epoch=epoch,
    #     distance=distance,
    #     environment="office_corridor",
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )
    # wang, zedigh = plot_best_additive_noise_methods_2(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     format="pgf",
    #     gaussian_value=0.04,
    #     collected_value=25,
    #     rayleigh_value=0.0138,
    #     epoch=epoch,
    #     # show=True,
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive_{trace_process_id}_per_additive",
    #     table_type="per_additive_method",
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive_{trace_process_id}_per_device",
    #     table_type="per_device",
    # )
    # plot_histogram_overview(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=6,
    #     distance=15,
    #     epoch=65,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )

    # # Feature scaling 2 - MaxMin (0, 1) over Sbox
    # #  Trace process 4
    trace_process_id = 4
    training_dataset_id = 3
    epoch = 65
    distance = 15
    test_dataset_id = 1
    environment_id = 1
    device = 10
    additive_noise_method = 4
    # # plot_example_normalized_training_trace(
    # #     training_dataset_id=3, trace_process_id=trace_process_id,
    # #     file_format="pgf", show=False, denoising_method_id=None
    # # )
    # plot_example_normalized_training_trace_1_row(
    #     training_dataset_id=3, trace_process_id=trace_process_id,
    #     file_format="pgf", show=False, denoising_method_id=None
    # )
    # plot_history_log__overview_trace_process(
    #     training_dataset_id=training_dataset_id,
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=False,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )
    # plot_epoch_comparison_report(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     trace_process_id=trace_process_id,
    #     environment_id=environment_id,
    #     distance=distance,
    #     device=device,
    #     additive_noise_method_id=additive_noise_method,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True
    # )
    # plot_all_of_an_additive_noise__report(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     additive_noise_method="Gaussian",
    #     trace_process_id=trace_process_id,
    #     epoch=epoch,
    #     distance=distance,
    #     environment="office_corridor",
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )
    # plot_all_of_an_additive_noise__report(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     additive_noise_method="Collected",
    #     trace_process_id=trace_process_id,
    #     epoch=epoch,
    #     distance=distance,
    #     environment="office_corridor",
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )
    # plot_all_of_an_additive_noise__report(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     additive_noise_method="Rayleigh",
    #     trace_process_id=trace_process_id,
    #     epoch=epoch,
    #     distance=distance,
    #     environment="office_corridor",
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )
    # wang, zedigh = plot_best_additive_noise_methods_2(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     format="pgf",
    #     gaussian_value=0.04,
    #     collected_value=25,
    #     rayleigh_value=0.0138,
    #     epoch=epoch,
    #     # show=True,
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive_{trace_process_id}_per_additive",
    #     table_type="per_additive_method",
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive_{trace_process_id}_per_device",
    #     table_type="per_device",
    # )
    # plot_histogram_overview(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=6,
    #     distance=15,
    #     epoch=65,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )

    # Feature scaling 3 - MaxMin(-1, 1) over Sbox
    # Trace process id 9
    trace_process_id = 9
    # plot_example_normalized_training_trace_1_row(
    #     training_dataset_id=3, trace_process_id=trace_process_id,
    #     file_format="pgf", show=False, denoising_method_id=None
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )

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
    #
    epoch_none = 11
    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Gaussian',
    #     parameter_1_value_1=0.01,
    #     parameter_1_value_2=0.03,
    #     parameter_1_value_3=0.04,
    #     parameter_1_value_4=0.05,
    #     epoch_none=epoch_none,
    #     epoch_1=20,
    #     epoch_2=7,
    #     epoch_3=7,
    #     epoch_4=19,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )

    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Collected',
    #     parameter_1_value_1=25,
    #     parameter_1_value_2=50,
    #     parameter_1_value_3=75,
    #     parameter_1_value_4=105,
    #     epoch_none=epoch_none,
    #     epoch_1=20,  # Not investigated yet
    #     epoch_2=7,
    #     epoch_3=12,  # Not investigated yet
    #     epoch_4=19,
    #     show=True,
    # )

    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Rayleigh',
    #     parameter_1_value_1=0.0138,
    #     parameter_1_value_2=0.0276,
    #     parameter_1_value_3=0,
    #     parameter_1_value_4=0,
    #     epoch_none=epoch_none,
    #     epoch_1=6,
    #     epoch_2=13,
    #     epoch_3=0,
    #     epoch_4=0,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )

    # wang, zedigh = plot_best_additive_noise_methods_2(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     gaussian_value=0.05,
    #     collected_value=50,
    #     rayleigh_value=0.0138,
    #     epoch_none=20,
    #     epoch_gaussian=19,
    #     epoch_collected=7,
    #     epoch_rayleigh=6,
    #     show=True,
    #     y_top=1000,
    #     row_size=1,
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive__per_additive",
    #     table_type="per_additive_method",
    # )

    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive__per_device",
    #     table_type="per_device",
    # )

    # plot_histogram_overview(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=6,
    #     distance=15,
    #     epoch=65,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )

    # Feature scaling 2 - MaxMin (0, 1) over Sbox
    trace_process_id = 4
    # plot_example_normalized_training_trace_1_row(
    #     training_dataset_id=3, trace_process_id=trace_process_id,
    #     file_format="pgf", show=False, denoising_method_id=None
    # )

    # plot_epoch_comparison_report(
    #     training_model_id=1,
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=trace_process_id,
    #     environment_id=1,
    #     distance=15,
    #     device=10,
    #     additive_noise_method_id=11,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    #     y_bottom=100,
    #     y_top=400,
    # )

    epoch_none = 6  # ??
    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Gaussian',
    #     parameter_1_value_1=0.01,
    #     parameter_1_value_2=0.03,
    #     parameter_1_value_3=0.04,
    #     parameter_1_value_4=0.05,
    #     epoch_none=epoch_none,
    #     epoch_1=20,  # ??
    #     epoch_2=20,
    #     epoch_3=7,  # ??
    #     epoch_4=16,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )

    plot_all_of_an_additive_noise__report__2(
        trace_process_id=trace_process_id,
        additive_noise_method='Collected',
        parameter_1_value_1=25,
        parameter_1_value_2=50,
        parameter_1_value_3=75,
        parameter_1_value_4=105,
        epoch_none=epoch_none,
        epoch_1=20,  # Not investigated yet
        epoch_2=7,
        epoch_3=12,  # Not investigated yet
        epoch_4=19,
        # show=True,
        y_top=1000,
        row_size=1,
    )

    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Rayleigh',
    #     parameter_1_value_1=0.0138,
    #     parameter_1_value_2=0.0276,
    #     parameter_1_value_3=0,
    #     parameter_1_value_4=0,
    #     epoch_none=epoch_none,
    #     epoch_1=20,
    #     epoch_2=15,
    #     epoch_3=0,
    #     epoch_4=0,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )

    # wang, zedigh = plot_best_additive_noise_methods_2(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     gaussian_value=0.05,
    #     collected_value=50,
    #     rayleigh_value=0.0138,
    #     epoch_none=epoch_none,
    #     epoch_gaussian=16,
    #     epoch_collected=7,
    #     epoch_rayleigh=20,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive_{trace_process_id}_per_additive",
    #     table_type="per_additive_method",
    # )

    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive__per_device",
    #     table_type="per_device",
    # )

    # plot_histogram_overview(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=6,
    #     distance=15,
    #     epoch=65,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )

    # Feature scaling 4 - MaxMin(-1, 1) over Sbox
    # Trace process id 10
    trace_process_id = 10
    # plot_example_normalized_training_trace_1_row(
    #     training_dataset_id=3, trace_process_id=trace_process_id,
    #     file_format="pgf", show=False, denoising_method_id=None
    # )

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
    #
    epoch_none = 6
    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Gaussian',
    #     parameter_1_value_1=0.01,
    #     parameter_1_value_2=0.03,
    #     parameter_1_value_3=0.04,
    #     parameter_1_value_4=0.05,
    #     epoch_none=epoch_none,
    #     epoch_1=20,
    #     epoch_2=7,
    #     epoch_3=7,
    #     epoch_4=19,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )
    #
    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Collected',
    #     parameter_1_value_1=25,
    #     parameter_1_value_2=50,
    #     parameter_1_value_3=75,
    #     parameter_1_value_4=105,
    #     epoch_none=epoch_none,
    #     epoch_1=20,  # Not investigated yet
    #     epoch_2=7,
    #     epoch_3=12,  # Not investigated yet
    #     epoch_4=19,
    #     show=True,
    # )

    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Rayleigh',
    #     parameter_1_value_1=0.0138,
    #     parameter_1_value_2=0.0276,
    #     parameter_1_value_3=0,
    #     parameter_1_value_4=0,
    #     epoch_none=epoch_none,
    #     epoch_1=6,
    #     epoch_2=13,
    #     epoch_3=0,
    #     epoch_4=0,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )

    # wang, zedigh = plot_best_additive_noise_methods_2(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     gaussian_value=0.05,
    #     collected_value=50,
    #     rayleigh_value=0.0138,
    #     epoch_none=epoch_none,
    #     epoch_gaussian=19,
    #     epoch_collected=7,
    #     epoch_rayleigh=6,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive_{trace_process_id}_per_additive",
    #     table_type="per_additive_method",
    # )
    #
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive__per_device",
    #     table_type="per_device",
    # )

    # plot_histogram_overview(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=6,
    #     distance=15,
    #     epoch=65,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
    # )

    # Feature scaling 5 - Standardization over Sbox range
    trace_process_id = 8
    # plot_example_normalized_training_trace_1_row(
    #     training_dataset_id=3, trace_process_id=trace_process_id,
    #     file_format="pgf", show=False, denoising_method_id=None
    # )

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

    epoch_none = 6
    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Gaussian',
    #     parameter_1_value_1=0.01,
    #     parameter_1_value_2=0.03,
    #     parameter_1_value_3=0.04,
    #     parameter_1_value_4=0.05,
    #     epoch_none=epoch_none,
    #     epoch_1=20,
    #     epoch_2=7,
    #     epoch_3=7,
    #     epoch_4=19,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )
    #
    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Collected',
    #     parameter_1_value_1=25,
    #     parameter_1_value_2=50,
    #     parameter_1_value_3=75,
    #     parameter_1_value_4=105,
    #     epoch_none=epoch_none,
    #     epoch_1=20,  # Not investigated yet
    #     epoch_2=7,
    #     epoch_3=12,  # Not investigated yet
    #     epoch_4=19,
    #     show=True,
    # )

    # plot_all_of_an_additive_noise__report__2(
    #     trace_process_id=trace_process_id,
    #     additive_noise_method='Rayleigh',
    #     parameter_1_value_1=0.0138,
    #     parameter_1_value_2=0.0276,
    #     parameter_1_value_3=0,
    #     parameter_1_value_4=0,
    #     epoch_none=epoch_none,
    #     epoch_1=6,
    #     epoch_2=13,
    #     epoch_3=0,
    #     epoch_4=0,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )

    # wang, zedigh = plot_best_additive_noise_methods_2(
    #     training_dataset='Wang_2021 - Cable, 5 devices, 500k traces',
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     gaussian_value=0.05,
    #     collected_value=50,
    #     rayleigh_value=0.0138,
    #     epoch_none=epoch_none,
    #     epoch_gaussian=19,
    #     epoch_collected=7,
    #     epoch_rayleigh=6,
    #     # show=True,
    #     y_top=1000,
    #     row_size=1,
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive_{trace_process_id}_per_additive",
    #     table_type="per_additive_method",
    # )
    #
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=trace_process_id,
    #     file_name=f"best_additive",
    #     label=f"tbl:best_additive__per_device",
    #     table_type="per_device",
    # )

    # plot_histogram_overview(
    #     training_model_id=1,
    #     training_dataset_id=training_dataset_id,
    #     test_dataset_id=test_dataset_id,
    #     environment_id=1,
    #     trace_process_id=3,
    #     device=6,
    #     distance=15,
    #     epoch=65,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     # show=True,
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

    # Appendix
    # History log plots

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3, # observe 1 or 3   !!!!!!!!!!!!!TODO
    #     trace_process_id=3,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=4,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=5,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=6,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=7,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
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

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=9,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=10,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )

    # plot_history_log__overview_trace_process(
    #     training_dataset_id=3,
    #     trace_process_id=11,
    #     save_path=REPORT_DIR,
    #     file_format="pgf",
    #     show=True,
    #     last_gaussian=5,
    #     last_collected=9,
    #     last_rayleigh=11,
    #     nrows=4,
    #     ncols=4,
    # )
