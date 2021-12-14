--liquibase formatted sql

--changeset axelzedigh:17
--Database: sqlite
INSERT INTO training_models VALUES (3,'cnn_110_simpler');
