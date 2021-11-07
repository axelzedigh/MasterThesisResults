from utils.db_utils import insert_data_to_db
import time
import sys


def main(dev):
    """

    :param dev:
    """
    database = "TerminationPoints.db"
    testing_dataset = "Wang2021"
    environment = "office"
    distance = 15
    device = dev
    training_model = "CNN 110"
    keybyte = 0
    epoch = 65
    additive_noise_method = "gaussian"
    additive_noise_parameter_1 = "std"
    additive_noise_parameter_1_value = 0.04
    additive_noise_parameter_2 = "Mean"
    additive_noise_parameter_2_value = 0
    denoising_method = None
    denoising_method_parameter_1 = None
    denoising_method_parameter_1_value = None
    denoising_method_parameter_2 = None
    denoising_method_parameter_2_value = None
    termination_point = 200
    insert_data_to_db(
        database,
        testing_dataset,
        environment_id=environment,
        distance=distance,
        device=device,
        training_model_id=training_model,
        keybyte=keybyte,
        epoch=epoch,
        additive_noise_method_id=additive_noise_method,
        denoising_method_id=denoising_method,
        termination_point=termination_point,
    )
    print(f"Added one entry of device {dev}!")
    time.sleep(5)
    main(sys.argv[1])


if __name__ == "__main__":
    main(sys.argv[1])
