from initialize_sqlite3_db import insert_data, fetchall
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
    insert_data(database,
                testing_dataset,
                environment,
                distance,
                device,
                training_model,
                keybyte,
                epoch,
                additive_noise_method,
                additive_noise_parameter_1,
                additive_noise_parameter_1_value,
                additive_noise_parameter_2,
                additive_noise_parameter_2_value,
                denoising_method,
                denoising_method_parameter_1,
                denoising_method_parameter_1_value,
                denoising_method_parameter_2,
                denoising_method_parameter_2_value,
                termination_point,
                )
    print(f"Added one entry of device {dev}!")
    time.sleep(5)
    main(sys.argv[1])


if __name__ == "__main__":
    main(sys.argv[1])
