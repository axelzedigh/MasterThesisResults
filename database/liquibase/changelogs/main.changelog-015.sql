--liquibase formatted sql

--changeset axelzedigh:16
--Database: sqlite
INSERT INTO training_models VALUES (2,'cnn_110_sgd');
