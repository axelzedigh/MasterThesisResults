--liquibase formatted sql

--changeset axelzedigh:13
--Database: sqlite
INSERT INTO additive_noise_methods VALUES
(12,'Rayleigh', 'Mode', 0.0069, NULL, NULL);
