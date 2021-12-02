--liquibase formatted sql

--changeset axelzedigh:9
--Database: sqlite
INSERT INTO
    denoising_methods
VALUES (3, 'Wiener Filter', 'Noise Power', "2e-7", NULL, NULL);
