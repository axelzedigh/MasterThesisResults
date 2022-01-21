--liquibase formatted sql

--changeset axelzedigh:33
--Database: sqlite
UPDATE denoising_methods set denoising_method_parameter_1_value = NULL where id = 3;
INSERT INTO denoising_methods VALUES (4,'Wiener Filter', 'Noise Power', 0.2, NULL, NULL);
INSERT INTO denoising_methods VALUES (5,'Moving Average Filter', 'N', 11, NULL, NULL);
INSERT INTO denoising_methods VALUES (6,'Wiener Filter', 'Noise Power', 0.02, NULL, NULL);
