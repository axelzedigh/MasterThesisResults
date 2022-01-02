from configs.variables import REPORT_DIR
from plots.rank_test_plots import plot_best_additive_noise_methods, \
    plot_best_additive_noise_methods_2
from plots.trace_plots import plot_additive_noises_examples

def plot_overview():
    pass


if __name__ == "__main__":
    # Example additive noise plots
    # plot_additive_noises_examples(
    #     save_path=REPORT_DIR,
    #     format="pgf",
    # )

    # Best additive noise methods
    plot_best_additive_noise_methods_2(
        trace_process_id=3,
        save_path=REPORT_DIR,
        format="pgf",
        # show=True
    )
