--liquibase formatted sql

--changeset axelzedigh:30
--Database: sqlite
CREATE VIEW IF NOT EXISTS quality_table_2
AS
SELECT
    rank_test__grouped.training_dataset_id,
    rank_test__grouped.test_dataset_id,
    rank_test__grouped.environment_id,
    rank_test__grouped.distance,
    rank_test__grouped.device,
    rank_test__grouped.epoch,
    rank_test__grouped.additive_noise_method_id,
    rank_test__grouped.denoising_method_id,
    rank_test__grouped.count_term_p,
    rank_test__grouped.avg_term_p,
    rank_test__grouped.trace_process_id AS rank_trace_process_id,
    trace_metadata_depth.trace_process_id,
    trace_metadata_depth.additive_noise_method_id,
    trace_metadata_depth.trace_process_id,
    trace_metadata_depth.data_point_index,
    trace_metadata_depth.max_val,
    trace_metadata_depth.min_val,
    trace_metadata_depth.mean_val,
    trace_metadata_depth.rms_val,
    trace_metadata_depth.std_val,
    trace_metadata_depth.snr_val
FROM
    rank_test__grouped
LEFT JOIN trace_metadata_depth
    ON trace_metadata_depth.test_dataset_id = rank_test__grouped.test_dataset_id
    AND trace_metadata_depth.environment_id = rank_test__grouped.environment_id
    AND trace_metadata_depth.distance = rank_test__grouped.distance
    AND trace_metadata_depth.device = rank_test__grouped.device
    ;
