--liquibase formatted sql

--changeset axelzedigh:18
--Database: sqlite
INSERT INTO training_models VALUES (4,'cnn_110_more');
