"""Python script to run from terminal for rank test (see shell_rank_test.sh)."""

import sys

from tqdm import tqdm
from scripts.RankTestScriptToDatabase import termination_point_test
from utils.db_utils import insert_data_to_db
from utils.denoising_utils import moving_average_filter_n3, \
    moving_average_filter_n5


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
    if denoising_method_id == 1:
        filter_function = moving_average_filter_n3
    if denoising_method_id == 2:
        filter_function = moving_average_filter_n5
    for _ in tqdm(range(0, runs)):
        termination_point = termination_point_test(
            database,
            filter_function,
            test_dataset_id,
            environment_id,
            distance,
            device,
            training_model_id,
            keybyte,
            epoch,
            additive_noise_method_id,
            denoising_method_id,
            trace_process_id=trace_process_id,
        )
        print(termination_point)
        if termination_point is not None:
            insert_data_to_db(
                database,
                test_dataset_id,
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
    )
