--liquibase formatted sql

--changeset axelzedigh:4
--Database: sqlite
UPDATE
    trace_processes
SET trace_process = 'Normalized - MaxMin S-Box Range - Normalization after additive noise'
WHERE
    id = 4;