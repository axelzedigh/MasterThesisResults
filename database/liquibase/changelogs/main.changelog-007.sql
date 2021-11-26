--liquibase formatted sql

--changeset axelzedigh:8
--Database: sqlite
INSERT INTO trace_processes
VALUES
(6,
'Normalized - MaxMin Sbox Range - No re-normalization - 100k training traces');

INSERT INTO trace_processes
VALUES
(7,
'Normalized - MaxMin Sbox Range - No re-normalization - 500k training traces');
