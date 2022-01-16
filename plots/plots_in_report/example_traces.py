"""Plots for report."""
from plots.trace_plots import plot_example_normalized_training_trace_1_row

if __name__ == '__main__':
    for trace_process in [3, 4, 8, 9, 10, 11, 12, 13]:
        plot_example_normalized_training_trace_1_row(
            training_dataset_id=3, trace_process_id=trace_process,
            file_format="pgf", show=False, denoising_method_id=None
        )
