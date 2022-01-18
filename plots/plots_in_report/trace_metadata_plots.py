"""Trace metadata plots."""
from plots.trace_metadata_plots import plot_test_trace_metadata_depth__mean, \
    plot_test_trace_metadata_depth__rms__report

if __name__ == '__main__':
    # plot_test_trace_metadata_depth__mean(
    #     test_dataset_id=1,
    #     distance=15,
    #     devices=[6, 7, 8, 9, 10],
    #     environment_id=1,
    #     trace_process_id=2,
    #     show=True,
    # )

    # 15m, wang, office corridor
    # plot_test_trace_metadata_depth__rms__report(
    #     test_dataset_id=1,
    #     distance=15,
    #     devices=[6, 7, 8, 9, 10],
    #     environment_id=1,
    #     trace_process_id=2,
    #     show=True,
    # )

    # 15m, zedigh, office corridor
    plot_test_trace_metadata_depth__rms__report(
        test_dataset_id=2,
        distance=15,
        devices=[8, 9, 10],
        environment_id=1,
        trace_process_id=2,
        show=True,
    )

    # 10m, zedigh, office corridor
    plot_test_trace_metadata_depth__rms__report(
        test_dataset_id=2,
        distance=10,
        devices=[8, 9, 10],
        environment_id=1,
        trace_process_id=2,
        show=True,
    )

    # 5m, zedigh, office corridor
    plot_test_trace_metadata_depth__rms__report(
        test_dataset_id=2,
        distance=5,
        devices=[8, 9, 10],
        environment_id=1,
        trace_process_id=2,
        show=True,
    )

    # 5m, zedigh, big hall
    plot_test_trace_metadata_depth__rms__report(
        test_dataset_id=2,
        distance=5,
        devices=[8, 9, 10],
        environment_id=2,
        trace_process_id=2,
        show=True,
    )
