QUERY_CREATE_TABLE_ENVIRONMENTS = """
        CREATE TABLE IF NOT EXISTS Environments(
        id INTEGER PRIMARY KEY,
        environment TEXT NOT NULL
        );
        """

QUERY_CREATE_TABLE_TEST_DATASETS = """
        CREATE TABLE IF NOT EXISTS Test_Datasets(
        id INTEGER PRIMARY KEY,
        test_dataset TEXT
        );
        """

QUERY_CREATE_TABLE_TRAINING_DATASETS = """
        CREATE TABLE IF NOT EXISTS Training_Datasets(
        id INTEGER PRIMARY KEY,
        training_dataset TEXT
        );
        """

QUERY_CREATE_TABLE_TRAINING_MODELS = """
        CREATE TABLE IF NOT EXISTS Training_Models(
        id INTEGER PRIMARY KEY,
        training_model TEXT
        );
        """

QUERY_CREATE_TABLE_ADDITIVE_NOISE_METHODS = """
        CREATE TABLE IF NOT EXISTS Additive_Noise_Methods(
        id INTEGER PRIMARY KEY,
        additive_noise_method TEXT,
        additive_noise_method_parameter_1 TEXT,
        additive_noise_method_parameter_1_value FLOAT,
        additive_noise_method_parameter_2 TEXT,
        additive_noise_method_parameter_2_value FLOAT
        );
        """

QUERY_CREATE_TABLE_DENOISING_METHODS = """
        CREATE TABLE IF NOT EXISTS Denoising_Methods(
        id INTEGER PRIMARY KEY,
        denoising_method TEXT,
        denoising_method_parameter_1 TEXT,
        denoising_method_parameter_1_value FLOAT,
        denoising_method_parameter_2 TEXT,
        denoising_method_parameter_2_value FLOAT
        );
        """

QUERY_CREATE_TABLE_RANK_TEST = """
        CREATE TABLE IF NOT EXISTS Rank_Test(
        id INTEGER PRIMARY KEY,
        test_dataset_id INTEGER NOT NULL,
        training_dataset_id INTEGER NOT NULL,
        environment_id INTEGER NOT NULL,
        distance FLOAT NOT NULL,
        device INT NOT NULL, 
        training_model_id INTEGER,
        keybyte INT NOT NULL,
        epoch INT NOT NULL,
        additive_noise_method_id INTEGER,
        denoising_method_id INTEGER,
        termination_point INT NOT NULL,
        trace_process_id INT,
        date_added REAL NOT NULL,

        FOREIGN KEY(test_dataset_id) 
            REFERENCES Test_Datasets(test_dataset_id),
        FOREIGN KEY(training_dataset_id) 
            REFERENCES Training_Datasets(training_dataset_id),
        FOREIGN KEY(environment_id) 
            REFERENCES Environments(environments_id),
        FOREIGN KEY(training_model_id) 
            REFERENCES Training_Model(training_model_id),
        FOREIGN KEY(additive_noise_method_id) 
            REFERENCES Additive_Noise_Methods(additive_noise_method_id),
        FOREIGN KEY(denoising_method_id) 
            REFERENCES Denoising_Method(denoising_method_id)
        )
        """

QUERY_CREATE_VIEW_FULL_RANK_TEST = """
        CREATE VIEW IF NOT EXISTS full_rank_test
        AS
        SELECT
            Rank_Test.id,
            Test_Datasets.test_dataset AS test_dataset,
            Training_Datasets.training_dataset AS training_dataset,
            Environments.environment AS environment,
            Rank_Test.distance,
            Rank_Test.device,
            Training_Models.training_model AS training_model,
            Rank_Test.keybyte,
            Rank_Test.epoch,
            Additive_Noise_Methods.additive_noise_method 
                AS additive_noise_method,
            Additive_Noise_Methods.additive_noise_method_parameter_1 
                AS additive_noise_method_parameter_1,
            Additive_Noise_Methods.additive_noise_method_parameter_1_value 
                AS additive_noise_method_parameter_1_value,
            Additive_Noise_Methods.additive_noise_method_parameter_2 
                AS additive_noise_method_parameter_2,
            Additive_Noise_Methods.additive_noise_method_parameter_2_value 
                AS additive_noise_method_parameter_2_value,
            Denoising_Methods.denoising_method AS denoising_method,
            Denoising_Methods.denoising_method_parameter_1 
                AS denoising_method_parameter_1,
            Denoising_Methods.denoising_method_parameter_1_value 
                AS denoising_method_parameter_1_value,
            Denoising_Methods.denoising_method_parameter_2 
                AS denoising_method_parameter_2,
            Denoising_Methods.denoising_method_parameter_2_value 
                AS denoising_method_parameter_2_value,
            Rank_Test.termination_point,
            Rank_Test.trace_process_id,
            Rank_Test.date_added
        FROM
            Rank_Test
        LEFT JOIN Test_Datasets 
            ON Test_Datasets.id = Rank_Test.test_dataset_id
        LEFT JOIN Training_Datasets 
            ON Training_Datasets.id = Rank_Test.training_dataset_id
        LEFT JOIN Environments 
            ON Environments.id = Rank_Test.environment_id
        LEFT JOIN Training_Models 
            ON Training_Models.id = Rank_Test.training_model_id
        LEFT JOIN Additive_Noise_Methods 
            ON Additive_Noise_Methods.id = Rank_test.additive_noise_method_id
        LEFT JOIN Denoising_Methods 
            ON Denoising_Methods.id = Rank_Test.denoising_method_id;
        """

QUERY_SELECT_ADDITIVE_NOISE_METHOD_ID = """
        SELECT
            Additive_Noise_Methods.id
        FROM Additive_Noise_Methods
        WHERE 
            additive_noise_method = ? AND
            (additive_noise_method_parameter_1 = ? 
                OR additive_noise_method_parameter_1 IS NULL) AND
            (additive_noise_method_parameter_1_value = ? 
                OR additive_noise_method_parameter_1_value IS NULL) AND
            (additive_noise_method_parameter_2 = ? 
                OR additive_noise_method_parameter_2 IS NULL) AND
            (additive_noise_method_parameter_2_value = ? 
                OR additive_noise_method_parameter_2_value IS NULL);
        """

QUERY_SELECT_DENOISING_METHOD_ID = """
        SELECT
            Denoising_Methods.id
        FROM Denoising_Methods
        WHERE 
            (denoising_method = ? 
                OR denoising_method IS NULL) AND
            (denoising_method_parameter_1 = ? 
                OR denoising_method_parameter_1 IS NULL) AND
            (denoising_method_parameter_1_value = ? 
                OR denoising_method_parameter_1_value IS NULL) AND
            (denoising_method_parameter_2 = ? 
                OR denoising_method_parameter_2 IS NULL) AND
            (denoising_method_parameter_2_value = ? 
                OR denoising_method_parameter_2_value IS NULL);
        """

QUERY_LIST_INITIALIZE_DB = [
    "INSERT INTO environments VALUES (1,'office_corridor');",
    "INSERT INTO environments VALUES (2,'big_hall');",

    "INSERT INTO test_datasets VALUES (1,'Wang_2021');",
    "INSERT INTO test_datasets VALUES (2,'Zedigh_2021');",

    ("INSERT INTO training_datasets VALUES (1,'Wang_2021 - Cable, 5 devices, "
     "200k traces');"),
    ("INSERT INTO training_datasets VALUES (2,'Wang_2021 - Cable, 5 devices, "
     "100k traces');"),
    ("INSERT INTO training_datasets VALUES (3,'Wang_2021 - Cable, 5 devices, "
     "500k traces');"),

    "INSERT INTO training_models VALUES (1,'cnn_110');",
    "INSERT INTO training_models VALUES (2,'cnn_110_sgd');",
    "INSERT INTO training_models VALUES (3,'cnn_110_simpler');",
    "INSERT INTO training_models VALUES (4,'cnn_110_more');",
    "INSERT INTO training_models VALUES (5,'cnn_110_batch_normalization');",

    "INSERT INTO additive_noise_methods VALUES (1,'Gaussian', 'Std', 0.01, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (2,'Gaussian', 'Std', 0.02, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (3,'Gaussian', 'Std', 0.03, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (4,'Gaussian', 'Std', 0.04, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (5,'Gaussian', 'Std', 0.05, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (6,'Collected', 'Scale', "
    "25, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (7,'Collected', 'Scale', "
    "50, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (8,'Collected', 'Scale', "
    "75, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (9,'Collected', 'Scale', "
    "105, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (10,'Rayleigh', 'Mode', "
    "0.0138, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (11,'Rayleigh', 'Mode', "
    "0.0276, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (12,'Rayleigh', 'Mode', "
    "0.0069, NULL, NULL); ",

    "INSERT INTO denoising_methods VALUES (1,'Moving Average Filter', 'N', "
    "3, NULL, NULL); ",
    "INSERT INTO denoising_methods VALUES (2,'Moving Average Filter', 'N', "
    "5, NULL, NULL); ",
    "INSERT INTO denoising_methods VALUES (3,'Wiener Filter', 'Noise Power', "
    "NULL, NULL, NULL); ",
    "INSERT INTO denoising_methods VALUES (4,'Wiener Filter', 'Noise Power', "
    "0.2, NULL, NULL); ",
    "INSERT INTO denoising_methods VALUES (5,'Moving Average Filter', 'N', "
    "11, NULL, NULL); ",
    "INSERT INTO denoising_methods VALUES (6,'Wiener Filter', 'Noise Power', "
    "0.02, NULL, NULL); ",

    "INSERT INTO trace_processes VALUES (1, 'Raw (all__.npy');",
    "INSERT INTO trace_processes VALUES (2, 'Randomized order (traces.npy)');",
    ("INSERT INTO trace_processes VALUES (3, 'MaxMin - "
     "Whole trace [0:400]');"),
    ("INSERT INTO trace_processes VALUES (4, 'MaxMin - "
     "SBox Range [204:314]');"),
    ("INSERT INTO trace_processes VALUES (5, 'MaxMin - "
     "SBox Range [204:314] - Normalization after additive noise');"),
    ("INSERT INTO trace_processes VALUES (6, 'MaxMin - "
     "Min in SBox Range [204:314] - Max: Avg([74:174]) * 2.2');"),
    ("INSERT INTO trace_processes VALUES (7, 'MaxMin - "
     "Min in SBox Range [204:314] - Max: Avg([204:314]) * 1.8');"),
    ("INSERT INTO trace_processes VALUES (8, 'Standardization - "
     "SBox Range [204:314]');"),
    ("INSERT INTO trace_processes VALUES (9, 'MaxMin [-1, 1] - "
     "Whole trace [0:400]');"),
    ("INSERT INTO trace_processes VALUES (10, 'MaxMin [-1, 1] - "
     "Sbox Range [204:314]');"),
    ("INSERT INTO trace_processes VALUES (11, 'Standardization - SBox Range - "
     "Difference with Mean');"),
    ("INSERT INTO trace_processes VALUES (12, 'Standardization - SBox Range - "
     "Misalignment ?? 1');"),
    ("INSERT INTO trace_processes VALUES (13, 'MaxMin - Whole trace - "
     "Misalignment ?? 1');"),
    ("INSERT INTO trace_processes VALUES (14, 'Standardization - SBox Range "
     "[200, 310] - Misalignment ?? 1');"),
]

QUERY_RANK_TEST_GROUPED_A = """
SELECT 
    *, 
    Count(*), 
    avg(termination_point) 
FROM 
    Rank_Test 
GROUP BY
    test_dataset_id,
    training_dataset_id,
    environment_id,
    distance,
    device,
    training_model_id,
    keybyte,
    epoch,
    additive_noise_method_id,  
    denoising_method_id;
"""

QUERY_FULL_RANK_TEST_GROUPED_A = """
SELECT
    test_dataset,
    training_dataset,
    environment,
    distance,
    device,
    training_model,
    keybyte,
    epoch,
    trace_process_id,
    additive_noise_method
        AS additive_method,
    additive_noise_method_parameter_1
        AS additive_param_1,
    additive_noise_method_parameter_1_value
        AS additive_param_1_value,
    additive_noise_method_parameter_2
        AS additive_param_2,
    additive_noise_method_parameter_2_value
        AS additive_param_2_value,
    denoising_method,
    denoising_method_parameter_1
        AS denoising_param_1,
    denoising_method_parameter_1_value
        AS denoising_param_1_value,
    denoising_method_parameter_2
        AS denoising_param_2,
    denoising_method_parameter_2_value 
        AS denoising_param_2_value,
    Count(termination_point) 
        AS count_term_p,
    avg(termination_point)
        AS avg_term_p
FROM
    full_rank_test
GROUP BY
    test_dataset,
    training_dataset,
    environment,
    distance,
    trace_process_id,
    additive_method,
    additive_param_1,
    additive_param_1_value,
    additive_param_2,
    additive_param_2_value,
    denoising_method,
    denoising_param_1,
    denoising_param_1_value,
    denoising_param_2,
    denoising_param_2_value,
    epoch,
    device,
    training_model,
    keybyte
"""

QUERY_CREATE_TABLE_TRACE_PROCESSES = """
        CREATE TABLE Trace_Processes(
        id INTEGER PRIMARY KEY,
        trace_process TEXT
        );
        """

QUERY_CREATE_TABLE_TRACE_METADATA_DEPTH = """
    CREATE TABLE IF NOT EXISTS Trace_Metadata_Depth(
    id INTEGER PRIMARY KEY,
    test_dataset_id INT,
    training_dataset_id INT,
    environment_id INT,
    distance FLOAT,
    device INT,
    additive_noise_method_id INT,
    trace_process_id INT NOT NULL,
    data_point_index INT NOT NULL,
    max_val FLOAT NOT NULL,
    min_val FLOAT NOT NULL,
    mean_val FLOAT NOT NULL,
    rms_val FLOAT NOT NULL,
    std_val FLOAT NOT NULL,
    snr_val FLOAT NOT NULL
    );
"""

QUERY_CREATE_TABLE_TRACE_METADATA_WIDTH = """
 CREATE TABLE IF NOT EXISTS Trace_Metadata_Width(
    id INTEGER PRIMARY KEY,
    test_dataset_id INT,
    training_dataset_id INT,
    environment_id INT,
    distance FLOAT,
    device INT,
    additive_noise_method_id INT,
    trace_process_id INT NOT NULL,
    trace_index INT NOT NULL,
    max_val FLOAT NOT NULL,
    min_val FLOAT NOT NULL,
    mean_val FLOAT NOT NULL,
    rms_val FLOAT NOT NULL,
    std_val FLOAT NOT NULL
 );
"""

QUERY_QUALITY_TABLE = """
    CREATE VIEW IF NOT EXISTS quality_table
    AS
    SELECT
        rank_test__grouped.training_dataset_id,
        rank_test__grouped.test_dataset_id,
        rank_test__grouped.environment_id,
        rank_test__grouped.distance,
        rank_test__grouped.device,
        rank_test__grouped.epoch,
        rank_test__grouped.additive_noise_method_id,
        rank_test__grouped.denoising_method_id,
        rank_test__grouped.count_term_p,
        rank_test__grouped.avg_term_p,
        rank_test__grouped.trace_process_id AS rank_trace_process_id,
        trace_metadata_depth__grouped.trace_process_id,
        trace_metadata_depth__grouped.max_max,
        trace_metadata_depth__grouped.min_min,
        trace_metadata_depth__grouped.avg_mean,
        trace_metadata_depth__grouped.avg_rms,
        trace_metadata_depth__grouped.avg_snr,
        trace_metadata_depth__grouped.snr_val
    FROM
        rank_test__grouped
    LEFT JOIN trace_metadata_depth__grouped 
        ON trace_metadata_depth__grouped.test_dataset_id = rank_test__grouped.test_dataset_id
        AND trace_metadata_depth__grouped.environment_id=rank_test__grouped.environment_id
        AND trace_metadata_depth__grouped.distance=rank_test__grouped.distance
        AND trace_metadata_depth__grouped.device=rank_test__grouped.device;
"""

QUERY_QUALITY_TABLE_2 = """
    CREATE VIEW IF NOT EXISTS quality_table_2
    AS
    SELECT
        rank_test__grouped.training_dataset_id,
        rank_test__grouped.test_dataset_id,
        rank_test__grouped.environment_id,
        rank_test__grouped.distance,
        rank_test__grouped.device,
        rank_test__grouped.epoch,
        rank_test__grouped.additive_noise_method_id AS rank_additive_noise_method_id,
        rank_test__grouped.denoising_method_id,
        rank_test__grouped.count_term_p,
        rank_test__grouped.avg_term_p,
        rank_test__grouped.trace_process_id AS rank_trace_process_id,
        trace_metadata_depth.additive_noise_method_id,
        trace_metadata_depth.trace_process_id,
        trace_metadata_depth.data_point_index,
        trace_metadata_depth.max_val,
        trace_metadata_depth.min_val,
        trace_metadata_depth.mean_val,
        trace_metadata_depth.rms_val,
        trace_metadata_depth.std_val,
        trace_metadata_depth.snr_val
    FROM
        rank_test__grouped
    LEFT JOIN trace_metadata_depth 
        ON trace_metadata_depth.test_dataset_id = rank_test__grouped.test_dataset_id
        AND trace_metadata_depth.environment_id = rank_test__grouped.environment_id
        AND trace_metadata_depth.distance = rank_test__grouped.distance
        AND trace_metadata_depth.device = rank_test__grouped.device
        ;
"""

QUERY_CREATE_VIEW_RANK_TEST__GROUPED = """
    CREATE VIEW IF NOT EXISTS rank_test__grouped
    AS
    SELECT
        test_dataset_id,
        training_dataset_id,
        environment_id,
        distance,
        device,
        training_model_id,
        trace_process_id,
        epoch,
        additive_noise_method_id,
        denoising_method_id,
        Count(termination_point) 
            AS count_term_p,
        avg(termination_point)
            AS avg_term_p
    FROM
        rank_test
    GROUP BY
        test_dataset_id,
        training_dataset_id,
        trace_process_id,
        environment_id,
        distance,
        additive_noise_method_id,
        denoising_method_id,
        epoch,
        device
    ORDER BY 
        avg_term_p
    ;
"""

QUERY_CREATE_VIEW_TRACE_METADATA_DEPTH__GROUPED = """
    CREATE VIEW IF NOT EXISTS trace_metadata_depth__grouped
    AS
    SELECT
        *,
        max(max_val) AS max_max,
        min(min_val) AS min_min,
        avg(mean_val) AS avg_mean,
        avg(rms_val) AS avg_rms,
        avg(snr_val) AS avg_snr
    FROM
        Trace_Metadata_Depth
    GROUP BY
        test_dataset_id,
        environment_id,
        distance,
        device,
        trace_process_id
    ;
"""

QUERY_CREATE_TABLE_NOISE_INFO = """
CREATE TABLE IF NOT EXISTS Noise_info(
    id INTEGER PRIMARY KEY,
    environment_id INT,
    RMS FLOAT,
    device_during_capturing INT
);
"""
