--liquibase formatted sql

--changeset axelzedigh:24
--Database: sqlite
    CREATE VIEW IF NOT EXISTS rank_test__grouped
    AS
    SELECT
        test_dataset_id,
        training_dataset_id,
        environment_id,
        distance,
        device,
        training_model_id,
        trace_process_id,
        epoch,
        additive_noise_method_id,
        denoising_method_id,
        Count(termination_point)
            AS count_term_p,
        avg(termination_point)
            AS avg_term_p
    FROM
        rank_test
    GROUP BY
        test_dataset_id,
        training_dataset_id,
        trace_process_id,
        environment_id,
        distance,
        additive_noise_method_id,
        denoising_method_id,
        epoch,
        device
    ORDER BY
        avg_term_p;
