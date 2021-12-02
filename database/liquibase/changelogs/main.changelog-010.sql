--liquibase formatted sql

--changeset axelzedigh:12
--Database: sqlite
DELETE FROM trace_processes;

INSERT INTO trace_processes VALUES (1, 'Raw (all__.npy');

INSERT INTO trace_processes VALUES (2, 'Randomized order (traces.npy)');

INSERT INTO trace_processes VALUES (3, 'MaxMin - Whole trace [0:400]');

INSERT INTO trace_processes VALUES (4, 'MaxMin - SBox Range [204:314]');

INSERT INTO trace_processes VALUES
(5, 'MaxMin - SBox Range [204:314] - Normalization after additive noise');

INSERT INTO trace_processes VALUES
(6, 'MaxMin - Min in SBox Range [204:314] - Max: Avg([74:174]) * 2.2');

INSERT INTO trace_processes VALUES
(7, 'MaxMin - Min in SBox Range [204:314] - Max: Avg([204:314]) * X');

INSERT INTO trace_processes VALUES
(8, 'Standardization - SBox Range [204:314]');
