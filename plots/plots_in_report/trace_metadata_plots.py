"""Trace metadata plots."""
from plots.trace_metadata_plots import plot_test_trace_metadata_depth__mean, \
    plot_test_trace_metadata_depth__rms__report, \
    plot_trace_termination_point

if __name__ == '__main__':
    # Run once!
    # insert_big_hall_traces_depth()

    # plot_test_trace_metadata_depth__mean(
    #     test_dataset_id=1,
    #     distance=15,
    #     devices=[6, 7, 8, 9, 10],
    #     environment_id=1,
    #     trace_process_id=2,
    #     show=True,
    # )

    # # 15m, wang, office corridor
    # plot_test_trace_metadata_depth__rms__report(
    #     test_dataset_id=1,
    #     distance=15,
    #     devices=[6, 7, 8, 9, 10],
    #     environment_id=1,
    #     trace_process_id=2,
    #     # show=True,
    #     ylabel="Wang2021 (15m)\n office corridor"
    # )
    #
    # # 15m, zedigh, office corridor
    # plot_test_trace_metadata_depth__rms__report(
    #     test_dataset_id=2,
    #     distance=15,
    #     devices=[8, 10],
    #     environment_id=1,
    #     trace_process_id=2,
    #     # show=True,
    #     ylabel="Zedigh2021 (15m)\n office corridor"
    # )
    #
    # # 10m, zedigh, office corridor
    # plot_test_trace_metadata_depth__rms__report(
    #     test_dataset_id=2,
    #     distance=10,
    #     devices=[8, 10],
    #     environment_id=1,
    #     trace_process_id=2,
    #     # show=True,
    #     ylabel="Zedigh2021 (10m)\n office corridor"
    # )
    #
    # # 5m, zedigh, office corridor
    # plot_test_trace_metadata_depth__rms__report(
    #     test_dataset_id=2,
    #     distance=5,
    #     devices=[8, 10],
    #     environment_id=1,
    #     trace_process_id=2,
    #     # show=True,
    #     ylabel="Zedigh2021 (5m)\noffice corridor"
    # )
    #
    # # 5m, zedigh, big hall
    # plot_test_trace_metadata_depth__rms__report(
    #     test_dataset_id=2,
    #     distance=5,
    #     devices=[8, 10],
    #     environment_id=2,
    #     trace_process_id=2,
    #     # show=True,
    #     ylabel="Zedigh2021 (5m)\nBig hall"
    # )

    # Device 8
    # trace process 3, training 3
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=3,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=3,
        environment_id=1,
        device=8,
        # show=True,
        legend_on=False,
        val_type="rms"
    )

    # trace process 3, training 1
    plot_trace_termination_point(
        training_dataset_id=1,
        test_dataset_id=1,
        trace_process_id=3,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=1,
        test_dataset_id=1,
        trace_process_id=3,
        environment_id=1,
        device=8,
        # show=True,
        legend_on=False,
        val_type="rms"
    )

    # trace process 4
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=4,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=4,
        environment_id=1,
        device=8,
        # show=True,
        legend_on=False,
        val_type="rms"
    )

    # trace process 8
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=8,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=8,
        environment_id=1,
        device=8,
        # show=True,
        legend_on=False,
        val_type="rms"
    )

    # trace process 9
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=9,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=9,
        environment_id=1,
        device=8,
        # show=True,
        legend_on=False,
        val_type="rms"
    )

    # trace process 10
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=10,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=10,
        environment_id=1,
        device=8,
        # show=True,
        legend_on=False,
        val_type="rms"
    )

    # trace process 11
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=11,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=11,
        environment_id=1,
        device=8,
        # show=True,
        legend_on=False,
        val_type="rms"
    )

    # trace process 12
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=12,
        environment_id=1,
        device=8,
        # show=True,
        val_type="snr"
    )
    plot_trace_termination_point(
        training_dataset_id=3,
        test_dataset_id=1,
        trace_process_id=12,
        environment_id=1,
        device=8,
        # show=True,
        val_type="rms",
        legend_on=False,
    )

    # # Device 10
    # # trace process 3, training 3
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=3,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=3,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     legend_on=False,
    #     val_type="rms"
    # )
    #
    # # trace process 3, training 1
    # plot_trace_termination_point(
    #     training_dataset_id=1,
    #     test_dataset_id=1,
    #     trace_process_id=3,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=1,
    #     test_dataset_id=1,
    #     trace_process_id=3,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     legend_on=False,
    #     val_type="rms"
    # )
    #
    # # trace process 4
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=4,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=4,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     legend_on=False,
    #     val_type="rms"
    # )
    #
    # # trace process 8
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=8,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=8,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     legend_on=False,
    #     val_type="rms"
    # )
    #
    # # trace process 9
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=9,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=9,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     legend_on=False,
    #     val_type="rms"
    # )
    #
    # # trace process 10
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=10,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=10,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     legend_on=False,
    #     val_type="rms"
    # )
    #
    # # trace process 11
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=11,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=11,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     legend_on=False,
    #     val_type="rms"
    # )
    #
    # # trace process 12
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=12,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="snr"
    # )
    # plot_trace_termination_point(
    #     training_dataset_id=3,
    #     test_dataset_id=1,
    #     trace_process_id=12,
    #     environment_id=1,
    #     device=10,
    #     # show=True,
    #     val_type="rms",
    #     legend_on=False,
    # )
