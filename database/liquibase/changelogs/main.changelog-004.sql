--liquibase formatted sql

--changeset axelzedigh:5
--Database: sqlite
INSERT INTO
    trace_processes
VALUES
(5, 'Normalized - MaxMin S-Box Range - No normalization after additive noise');
