from configs.variables import REPORT_DIR
from plots.rank_test_plots import plot_best_additive_noise_methods_2
from plots.trace_plots import plot_overview, plot_additive_noises_examples
from utils.plot_utils import df_to_latex__additive


if __name__ == "__main__":
    # Example additive noise plots
    # plot_additive_noises_examples(
    #     save_path=REPORT_DIR,
    #     format="pgf",
    # )

    # Best additive noise methods
    # trace_process_id = 3
    # wang, zedigh = plot_best_additive_noise_methods_2(
    #     trace_process_id=trace_process_id,
    #     save_path=REPORT_DIR,
    #     format="pgf",
    # )
    # df_to_latex__additive(
    #     wang,
    #     zedigh,
    #     trace_process_id=3,
    #     file_name=f"best_additive_{trace_process_id}",
    #     label="tbl:best_additive",
    # )

    pass