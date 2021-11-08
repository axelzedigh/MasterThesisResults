# Setup and run this file in order to insert numpy array files to db.
import os

# from docs.update_db_docs import update_preprocessing_docs
from docs.update_db_docs import update_preprocessing_docs
from utils.db_utils import insert_legacy_rank_test_numpy_file_to_db, \
    get_db_absolute_path


def main():
    # Mostly same
    test_dataset_id = 1
    training_dataset_id = 1
    environment_id = 1
    distance = 15
    training_model_id = 1

    # Look after these
    additive_noise_method_id = 4
    denoising_method_id = None

    files = os.listdir("to_migrate")
    db_path = get_db_absolute_path()
    for file in files:
        file_path = os.path.join("to_migrate", file)
        insert_legacy_rank_test_numpy_file_to_db(
            database=db_path,
            file_path=file_path,
            test_dataset_id=test_dataset_id,
            training_dataset_id=training_dataset_id,
            environment_id=environment_id,
            distance=distance,
            training_model_id=training_model_id,
            additive_noise_method_id=additive_noise_method_id,
            denoising_method_id=denoising_method_id,
        )
    update_preprocessing_docs()


if __name__ == "__main__":
    main()
