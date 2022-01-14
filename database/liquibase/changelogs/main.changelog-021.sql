--liquibase formatted sql

--changeset axelzedigh:21
--Database: sqlite
INSERT INTO training_models VALUES (5,'cnn_110_batch_normalization');
