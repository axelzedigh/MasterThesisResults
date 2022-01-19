--liquibase formatted sql

--changeset axelzedigh:23
--Database: sqlite
drop view quality_table;
drop view rank_test__grouped;