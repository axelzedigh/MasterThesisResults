"""Python script to run from terminal for rank test (see shell_rank_test.sh)."""

import sys

import numpy as np
from tqdm import tqdm

from scripts.RankTestScriptToDatabase import termination_point_test_setup, \
    termination_point_test__rank_test
from utils.db_utils import insert_data_to_db
from utils.denoising_utils import moving_average_filter_n3, \
    moving_average_filter_n5, wiener_filter_trace_set, moving_average_filter_n11


def termination_point_test_and_insert_to_db(
        database,
        runs,
        test_dataset_id,
        training_dataset_id,
        environment_id,
        distance,
        device,
        training_model_id,
        keybyte,
        epoch,
        additive_noise_method_id,
        denoising_method_id,
        trace_process_id,
        plot,
):
    """

    :param database:
    :param runs:
    :param test_dataset_id:
    :param training_dataset_id:
    :param environment_id:
    :param distance:
    :param device:
    :param training_model_id:
    :param keybyte:
    :param epoch:
    :param additive_noise_method_id:
    :param denoising_method_id:
    :param trace_process_id:
    """
    filter_function = None
    termination_point = None
    tp_list = []
    if denoising_method_id == 1:
        filter_function = moving_average_filter_n3
    elif denoising_method_id == 2:
        filter_function = moving_average_filter_n5
    elif denoising_method_id == 3:
        filter_function = wiener_filter_trace_set
    elif denoising_method_id == 5:
        filter_function = moving_average_filter_n11

    tqdm_iterator = tqdm(range(0, runs), desc=f"TP ep:{epoch}, Mean =  {termination_point}")

    t, p, k, c = termination_point_test_setup(
        database=database,
        filter_function=filter_function,
        test_dataset_id=test_dataset_id,
        training_dataset_id=training_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
        training_model_id=training_model_id,
        keybyte=keybyte,
        epoch=epoch,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        trace_process_id=trace_process_id,
        plot=plot,
    )

    for _ in tqdm_iterator:
        termination_point = termination_point_test__rank_test(
            testing_traces=t,
            predictions=p,
            key_interest=k,
            cts_interest=c,
            plot=plot,
        )
        if termination_point is not None:
            tp_list.append(termination_point)
            tqdm_iterator.set_description(
                f"add:{additive_noise_method_id}, epoch:{epoch}, Mean = {round(np.mean(tp_list), 1)}"
            )
            insert_data_to_db(
                database=database,
                test_dataset_id=test_dataset_id,
                training_dataset_id=training_dataset_id,
                environment_id=environment_id,
                distance=distance,
                device=device,
                training_model_id=training_model_id,
                keybyte=keybyte,
                epoch=epoch,
                additive_noise_method_id=additive_noise_method_id,
                denoising_method_id=denoising_method_id,
                termination_point=termination_point,
                trace_process_id=trace_process_id,
            )


if __name__ == "__main__":
    case = 4
    if case == 1:
        if sys.argv[11].strip() == "None":
            additive_id = None
        else:
            additive_id = int(sys.argv[11].strip())

        if sys.argv[12].strip() == "None":
            denoising_id = None
        else:
            denoising_id = int(sys.argv[12].strip())

        termination_point_test_and_insert_to_db(
            database=str(sys.argv[1]),
            runs=int(sys.argv[2]),
            test_dataset_id=int(sys.argv[3]),
            training_dataset_id=int(sys.argv[4]),
            environment_id=int(sys.argv[5]),
            distance=int(sys.argv[6]),
            device=int(sys.argv[7]),
            training_model_id=int(sys.argv[8]),
            keybyte=int(sys.argv[9]),
            epoch=int(sys.argv[10]),
            additive_noise_method_id=additive_id,
            denoising_method_id=denoising_id,
            trace_process_id=int(sys.argv[13]),
            plot=False
        )
    elif case == 2:
        database = "tmp_1.db"
        runs = 100,
        test_dataset_id = 1
        training_dataset_id = 2
        environment_id = 1
        distance = 15
        device = 10
        training_model_id = 1
        epoch = 45
        additive_noise_method_id = None
        denoising_method_id = None
        trace_process_id = 8

        termination_point_test_and_insert_to_db(
            database="tmp_1.db",
            runs=runs,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            environment_id=environment_id,
            distance=distance,
            device=device,
            training_model_id=training_model_id,
            keybyte=0,
            epoch=epoch,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
            trace_process_id=trace_process_id,
            plot=False
        )
    elif case == 3:
        database = "main.db"
        # database = "tmp_1.db"
        runs = 2
        # runs = 100
        test_dataset_ids = [1]
        training_dataset_ids = [3]
        environment_ids = [1]
        distances = [15]
        # devices = [6, 7, 8, 9, 10]
        devices = [10]
        training_model_id = 1
        epochs = [x for x in range(2, 20)]
        # epochs = [17]
        # additive_noise_method_ids = [None]
        additive_noise_method_ids = [4, 10, 11]
        denoising_method_ids = [None]
        trace_process_ids = [13]
        plot = False

        for test_dataset_id in test_dataset_ids:
            for training_dataset_id in training_dataset_ids:
                for environment_id in environment_ids:
                    for distance in distances:
                        for device in devices:
                            for epoch in epochs:
                                for additive_noise_method_id in additive_noise_method_ids:
                                    for denoising_method_id in denoising_method_ids:
                                        for trace_process_id in trace_process_ids:
                                            termination_point_test_and_insert_to_db(
                                                database=database,
                                                runs=runs,
                                                test_dataset_id=test_dataset_id,
                                                training_dataset_id=training_dataset_id,
                                                environment_id=environment_id,
                                                distance=distance,
                                                device=device,
                                                training_model_id=training_model_id,
                                                keybyte=0,
                                                epoch=epoch,
                                                additive_noise_method_id=additive_noise_method_id,
                                                denoising_method_id=denoising_method_id,
                                                trace_process_id=trace_process_id,
                                                plot=plot,
                                            )
    elif case == 4:
        database = "main.db"
        # database = "tmp_1.db"
        # runs = 2
        runs = 100
        test_dataset_ids = [1]
        training_dataset_ids = [3]
        environment_ids = [1]
        distances = [15]
        devices = [6, 7, 8, 9, 10]
        training_model_id = 1
        denoising_method_ids = [None]
        trace_process_ids = [13]
        plot = False
        additive_epochs = [
            (5, 15),
            (8, 17),
            (11, 17),
        ]

        for test_dataset_id in test_dataset_ids:
            for training_dataset_id in training_dataset_ids:
                for environment_id in environment_ids:
                    for distance in distances:
                        for device in devices:
                            for denoising_method_id in denoising_method_ids:
                                for trace_process_id in trace_process_ids:
                                    for item in additive_epochs:
                                        termination_point_test_and_insert_to_db(
                                            database=database,
                                            runs=runs,
                                            test_dataset_id=test_dataset_id,
                                            training_dataset_id=training_dataset_id,
                                            environment_id=environment_id,
                                            distance=distance,
                                            device=device,
                                            training_model_id=training_model_id,
                                            keybyte=0,
                                            epoch=item[1],
                                            additive_noise_method_id=item[0],
                                            denoising_method_id=denoising_method_id,
                                            trace_process_id=trace_process_id,
                                            plot=plot,
                                        )
