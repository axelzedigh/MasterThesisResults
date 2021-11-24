--liquibase formatted sql

--changeset axelzedigh:6
--Database: sqlite
UPDATE rank_test SET trace_process_id = 3;