--liquibase formatted sql

--changeset axelzedigh:10
--Database: sqlite
UPDATE training_datasets
SET training_dataset = "Wang_2021 - Cable, 5 devices, 200k traces"
WHERE id = 1;

INSERT INTO training_datasets VALUES (2,'Wang_2021 - Cable, 5 devices, 100k traces');
INSERT INTO training_datasets VALUES (3,'Wang_2021 - Cable, 5 devices, 500k traces');
