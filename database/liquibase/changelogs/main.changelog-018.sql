--liquibase formatted sql

--changeset axelzedigh:19
--Database: sqlite
INSERT INTO
    trace_processes
VALUES (12, 'Standardization - SBox Range - Misalignment ± 1');
