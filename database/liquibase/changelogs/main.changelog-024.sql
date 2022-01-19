--liquibase formatted sql

--changeset axelzedigh:25
--Database: sqlite
CREATE VIEW IF NOT EXISTS quality_table
AS
SELECT
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
    trace_metadata_depth__grouped.trace_process_id,
    trace_metadata_depth__grouped.max_max,
    trace_metadata_depth__grouped.min_min,
    trace_metadata_depth__grouped.avg_mean,
    trace_metadata_depth__grouped.avg_rms
FROM
    rank_test__grouped
LEFT JOIN trace_metadata_depth__grouped
    ON trace_metadata_depth__grouped.test_dataset_id = rank_test__grouped.test_dataset_id
    AND trace_metadata_depth__grouped.environment_id=rank_test__grouped.environment_id
    AND trace_metadata_depth__grouped.distance=rank_test__grouped.distance
    AND trace_metadata_depth__grouped.device=rank_test__grouped.device;
