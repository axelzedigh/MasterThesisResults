import os
from typing import Optional, List

from tensorflow.python.keras.models import load_model
import numpy as np
from tqdm import tqdm

from utils.db_utils import get_training_model_file_path, get_test_trace_path
from utils.trace_utils import get_normalized_test_traces


def get_model_class_prediction(
        training_model_id: int,
        training_dataset_id: int,
        trace_process_id: int,
        epoch: int,
        keybyte: int,
        test_dataset_id: int,
        environment_id: int,
        distance: float,
        device: int,
        additive_noise_method_id: Optional[int],
        denoising_method_id: Optional[int],
        trace_indexes: List[int],
):
    """Get the model class prediction."""

    # Get path to training model and load model
    training_model_file_path = get_training_model_file_path(
        database="main.db",
        training_model_id=training_model_id,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        epoch=epoch,
        keybyte=keybyte,
        trace_process_id=trace_process_id,
        training_dataset_id=training_dataset_id,
    )
    model_1 = load_model(training_model_file_path)
    training_model_file_path = get_training_model_file_path(
        database="main.db",
        training_model_id=training_model_id,
        additive_noise_method_id=4,
        denoising_method_id=denoising_method_id,
        epoch=65,
        keybyte=keybyte,
        trace_process_id=3,
        training_dataset_id=1,
    )
    model_2 = load_model(training_model_file_path)
    training_model_file_path = get_training_model_file_path(
        database="main.db",
        training_model_id=training_model_id,
        additive_noise_method_id=6,
        denoising_method_id=denoising_method_id,
        epoch=65,
        keybyte=keybyte,
        trace_process_id=3,
        training_dataset_id=1,
    )
    model_3 = load_model(training_model_file_path)

    # Get label
    path = get_test_trace_path(
        database="main.db",
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
    )
    labels = np.load(os.path.join(path, "label_lastround_Sout_0.npy"))

    # Get path test data and load trace/label.
    test_trace_set = get_normalized_test_traces(
        trace_process_id=trace_process_id,
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
        save=False,
    )
    test_trace_set = test_trace_set[:, [i for i in range(204, 314)]]

    prediction_1_list = list()
    prediction_2_list = list()
    prediction_3_list = list()
    prediction_avg_list = list()

    for trace_index in tqdm(trace_indexes):
        traces_copy = test_trace_set.copy()
        traces_copy = traces_copy.reshape(
            (traces_copy.shape[0], traces_copy.shape[1], 1)
        )

        # Select and cut trace set
        traces_copy = traces_copy[trace_index:trace_index+1]
        traces_copy = traces_copy.reshape(
            (traces_copy.shape[0], traces_copy.shape[1], 1)
        )

        # Get prediction.
        prediction_1 = model_1.predict(traces_copy)
        prediction_2 = model_2.predict(traces_copy)
        prediction_3 = model_3.predict(traces_copy)
        # avg_prediction = np.sqrt((prediction_1**2 + prediction_2**2 + prediction_3**2)/3)
        avg_prediction = (prediction_1 + prediction_2 + prediction_3)/3
        corr_label = labels[trace_index:trace_index+1].tolist()[0]

        larger_than_label_prediction_1 = [
            x for x in prediction_1[0] if x > prediction_1[0][corr_label]
        ]
        prediction_1_list.append(len(larger_than_label_prediction_1))

        larger_than_label_prediction_2 = [
            x for x in prediction_2[0] if x > prediction_2[0][corr_label]
        ]
        prediction_2_list.append(len(larger_than_label_prediction_2))

        larger_than_label_prediction_3 = [
            x for x in prediction_3[0] if x > prediction_3[0][corr_label]
        ]
        prediction_3_list.append(len(larger_than_label_prediction_3))

        larger_than_label_prediction_avg = [
            x for x in avg_prediction[0] if x > avg_prediction[0][corr_label]
        ]
        prediction_avg_list.append(len(larger_than_label_prediction_avg))

    print(f"Number of traces index: {len(trace_indexes)}")
    print(f"Prediction 1 (None): {np.mean(prediction_1_list)}")
    print(f"Prediction 2 (4): {np.mean(prediction_2_list)}")
    print(f"Prediction 3 (6): {np.mean(prediction_3_list)}")
    print(f"Prediction avg: {np.mean(prediction_avg_list)}")


if __name__ == "__main__":
    training_model_id = 1
    training_dataset_id = 3
    trace_process_id = 8
    epoch = 2
    keybyte = 0
    additive_noise_method_id = 6
    denoising_method_id = None

    test_dataset_id = 2
    environment_id = 2
    distance = 5
    device = 10

    trace_indexes = [x for x in range(100)]

    get_model_class_prediction(
        training_model_id=training_model_id,
        training_dataset_id=training_dataset_id,
        trace_process_id=trace_process_id,
        epoch=epoch,
        keybyte=keybyte,
        test_dataset_id=test_dataset_id,
        environment_id=environment_id,
        distance=distance,
        device=device,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        trace_indexes=trace_indexes,
    )
