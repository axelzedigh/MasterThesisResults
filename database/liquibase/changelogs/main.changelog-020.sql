--liquibase formatted sql

--changeset axelzedigh:20
--Database: sqlite
INSERT INTO trace_processes
VALUES (14, 'Standardization - SBox Range [200, 310] - Misalignment ± 1');
