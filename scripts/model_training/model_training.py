"""Main script for training models."""
from utils.training_utils import training_cnn_110

if __name__ == "__main__":
    keybyte = 0
    epochs = 100
    batch_size = 256
    additive_noise_method_id = None
    denoising_method_id = None
    training_model_id = 1
    trace_process_id = 6

    training_cnn_110(
        keybyte=keybyte,
        epochs=epochs,
        batch_size=batch_size,
        additive_noise_method_id=additive_noise_method_id,
        denoising_method_id=denoising_method_id,
        trace_process_id=trace_process_id,
        verbose=True,
    )
