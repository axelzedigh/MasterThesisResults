--liquibase formatted sql

--changeset axelzedigh:3
--Database: sqlite
CREATE VIEW IF NOT EXISTS full_rank_test
    AS
    SELECT
        Rank_Test.id,
        Test_Datasets.test_dataset AS test_dataset,
        Training_Datasets.training_dataset AS training_dataset,
        Environments.environment AS environment,
        Rank_Test.distance,
        Rank_Test.device,
        Training_Models.training_model AS training_model,
        Rank_Test.keybyte,
        Rank_Test.epoch,
        Additive_Noise_Methods.additive_noise_method
            AS additive_noise_method,
        Additive_Noise_Methods.additive_noise_method_parameter_1
            AS additive_noise_method_parameter_1,
        Additive_Noise_Methods.additive_noise_method_parameter_1_value
            AS additive_noise_method_parameter_1_value,
        Additive_Noise_Methods.additive_noise_method_parameter_2
            AS additive_noise_method_parameter_2,
        Additive_Noise_Methods.additive_noise_method_parameter_2_value
            AS additive_noise_method_parameter_2_value,
        Denoising_Methods.denoising_method AS denoising_method,
        Denoising_Methods.denoising_method_parameter_1
            AS denoising_method_parameter_1,
        Denoising_Methods.denoising_method_parameter_1_value
            AS denoising_method_parameter_1_value,
        Denoising_Methods.denoising_method_parameter_2
            AS denoising_method_parameter_2,
        Denoising_Methods.denoising_method_parameter_2_value
            AS denoising_method_parameter_2_value,
        Rank_Test.termination_point,
        Rank_Test.trace_process_id,
        Rank_Test.date_added
    FROM
        Rank_Test
    LEFT JOIN Test_Datasets
        ON Test_Datasets.id = Rank_Test.test_dataset_id
    LEFT JOIN Training_Datasets
        ON Training_Datasets.id = Rank_Test.training_dataset_id
    LEFT JOIN Environments
        ON Environments.id = Rank_Test.environment_id
    LEFT JOIN Training_Models
        ON Training_Models.id = Rank_Test.training_model_id
    LEFT JOIN Additive_Noise_Methods
        ON Additive_Noise_Methods.id = Rank_test.additive_noise_method_id
    LEFT JOIN Denoising_Methods
        ON Denoising_Methods.id = Rank_Test.denoising_method_id;
