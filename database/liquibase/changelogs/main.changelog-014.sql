--liquibase formatted sql

--changeset axelzedigh:15
--Database: sqlite
INSERT INTO
trace_processes
VALUES (11, 'Standardization - SBox Range - Difference with Mean');
