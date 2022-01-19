--liquibase formatted sql

--changeset axelzedigh:29
--Database: sqlite
drop view trace_metadata_depth__grouped;
CREATE VIEW IF NOT EXISTS trace_metadata_depth__grouped
AS
SELECT
    *,
    max(max_val) AS max_max,
    min(min_val) AS min_min,
    avg(mean_val) AS avg_mean,
    avg(rms_val) AS avg_rms,
    avg(snr_val) AS avg_snr
FROM
    Trace_Metadata_Depth
GROUP BY
    test_dataset_id,
    environment_id,
    distance,
    device,
    trace_process_id;
