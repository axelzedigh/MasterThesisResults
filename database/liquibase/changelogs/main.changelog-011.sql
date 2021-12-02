--liquibase formatted sql

--changeset axelzedigh:11
--Database: sqlite
Update trace_processes set trace_process = 'Raw (all__.npy)' where id = 1;
