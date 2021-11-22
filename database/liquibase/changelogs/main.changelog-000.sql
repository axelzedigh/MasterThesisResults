--liquibase formatted sql

--changeset axelzedigh:1
--Database: sqlite
ALTER TABLE rank_test RENAME COLUMN average_rank TO trace_process_id;
