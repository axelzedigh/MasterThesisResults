--liquibase formatted sql

--changeset axelzedigh:20
--Database: sqlite
INSERT INTO
    trace_processes
VALUES (13, 'MaxMin - Whole trace - Misalignment ± 1');
