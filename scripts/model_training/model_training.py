"""Main script for training models."""
from utils.training_utils import training_cnn_110

if __name__ == "__main__":
    case = 2

    if case == 1:
        training_dataset_id = 3
        keybyte = 0
        epochs = 30
        batch_size = 256
        additive_noise_method_id = None
        denoising_method_id = None
        training_model_id = 1
        trace_process_id = 8
        verbose = False
        mode = 1

        training_cnn_110(
            training_dataset_id=training_dataset_id,
            keybyte=keybyte,
            epochs=epochs,
            batch_size=batch_size,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
            trace_process_id=trace_process_id,
            verbose=verbose,
            mode=mode,
        )
    elif case == 2:
        keybyte = 0
        epochs = 20
        batch_size = 256
        verbose = False
        mode = 1

        training_model_id = 1
        training_dataset_ids = [3]
        additive_noise_method_ids = [None]
        denoising_method_ids = [None]
        trace_process_ids = [8]

        for training_dataset_id in training_dataset_ids:
            for additive_noise_method_id in additive_noise_method_ids:
                for denoising_method_id in denoising_method_ids:
                    for trace_process_id in trace_process_ids:
                        training_cnn_110(
                            training_dataset_id=training_dataset_id,
                            keybyte=keybyte,
                            epochs=epochs,
                            batch_size=batch_size,
                            additive_noise_method_id=additive_noise_method_id,
                            denoising_method_id=denoising_method_id,
                            trace_process_id=trace_process_id,
                            verbose=verbose,
                            mode=mode,
                        )
