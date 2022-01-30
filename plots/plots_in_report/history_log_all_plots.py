from plots.history_log_plots import plot_history_log__overview_trace_process

if __name__ == '__main__':
    add_axv_dict = {
        "None": 65,
        1: 65, 2: 65, 3: 65, 4: 65, 5: 65,
        6: 65, 7: 65, 8: 65, 9: 65,
        10: 65, 11: 65,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=1,
        trace_process_id=3,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        "None": 10,
        3: 17, 4: 14, 5: 15,
        6: 5, 7: 18,
        11: 17,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=3,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        "None": 7,
        6: 12, 7: 7, 8: 65, 9: 65,
        3: 20, 4: 9, 5: 16,
        10: 20, 11: 15,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=4,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        "None": 6,
        6: 6, 7: 4, 8: 65, 9: 65,
        3: 13, 4: 11, 5: 12,
        10: 16, 11: 15,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=9,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=5,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        "None": 6,
        6: 12, 7: 4,
        3: 7, 4: 7, 5: 19,
        10: 6, 11: 13,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=10,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        "None": 16,
        6: 9, 7: 8,
        3: 10, 4: 5, 5: 17,
        10: 6, 11: 15,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=8,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=5,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        "None": 12,
        6: 12, 7: 7,
        3: 12, 4: 12, 5: 13,
        10: 12, 11: 7,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=12,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        "None": 12,
        6: 17, 7: 11, 8: 17,
        3: 17, 4: 12, 5: 15,
        11: 17,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=13,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=4,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )

    add_axv_dict = {
        11: 17,
    }
    plot_history_log__overview_trace_process(
        training_dataset_id=3,
        trace_process_id=11,
        file_format="pgf",
        show=False,
        last_gaussian=5,
        last_collected=9,
        last_rayleigh=11,
        nrows=5,
        ncols=4,
        add_axv_dict=add_axv_dict,
    )
