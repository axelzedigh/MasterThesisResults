-- *********************************************************************
-- Update Database Script
-- *********************************************************************
-- Change Log: liquibase/dbchangelog.xml
-- Ran at: 2021-12-13 16:13
-- Against: null@jdbc:sqlite:main.db
-- Liquibase version: 4.6.1
-- *********************************************************************

-- Lock Database
UPDATE DATABASECHANGELOGLOCK SET LOCKED = 1, LOCKEDBY = '192.168.1.5 (192.168.1.5)', LOCKGRANTED = '2021-12-13 16:13:56.786' WHERE ID = 1 AND LOCKED = 0;

-- Release Database Lock
UPDATE DATABASECHANGELOGLOCK SET LOCKED = 0, LOCKEDBY = NULL, LOCKGRANTED = NULL WHERE ID = 1;

