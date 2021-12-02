--liquibase formatted sql

--changeset axelzedigh:14
--Database: sqlite
INSERT INTO trace_processes
VALUES (9, 'MaxMin [-1, 1] - Whole trace [0:400]');
INSERT INTO trace_processes
VALUES (10, 'MaxMin [-1, 1] - Sbox Range [204:314]');
