--liquibase formatted sql

--changeset axelzedigh:7
--Database: sqlite
CREATE TABLE IF NOT EXISTS Noise_info(
    id INTEGER PRIMARY KEY,
    environment_id INT,
    RMS FLOAT,
    device_during_capturing INT
);
