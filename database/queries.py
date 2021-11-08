QUERY_CREATE_TABLE_ENVIRONMENTS = """
        CREATE TABLE Environments(
        id INTEGER PRIMARY KEY,
        environment TEXT NOT NULL
        );
        """

QUERY_CREATE_TABLE_TEST_DATASETS = """
        CREATE TABLE Test_Datasets(
        id INTEGER PRIMARY KEY,
        test_dataset TEXT
        );
        """

QUERY_CREATE_TABLE_TRAINING_DATASETS = """
        CREATE TABLE Training_Datasets(
        id INTEGER PRIMARY KEY,
        training_dataset TEXT
        );
        """

QUERY_CREATE_TABLE_TRAINING_MODELS = """
        CREATE TABLE Training_Models(
        id INTEGER PRIMARY KEY,
        training_model TEXT
        );
        """

QUERY_CREATE_TABLE_ADDITIVE_NOISE_METHODS = """
        CREATE TABLE Additive_Noise_Methods(
        id INTEGER PRIMARY KEY,
        additive_noise_method TEXT,
        additive_noise_method_parameter_1 TEXT,
        additive_noise_method_parameter_1_value FLOAT,
        additive_noise_method_parameter_2 TEXT,
        additive_noise_method_parameter_2_value FLOAT
        );
        """

QUERY_CREATE_TABLE_DENOISING_METHODS = """
        CREATE TABLE Denoising_Methods(
        id INTEGER PRIMARY KEY,
        denoising_method TEXT,
        denoising_method_parameter_1 TEXT,
        denoising_method_parameter_1_value FLOAT,
        denoising_method_parameter_2 TEXT,
        denoising_method_parameter_2_value FLOAT
        );
        """

QUERY_CREATE_TABLE_RANK_TEST = """
        CREATE TABLE Rank_Test(
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
        average_rank INT,
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
        CREATE VIEW full_rank_test
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
            Rank_Test.average_rank,
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
    "INSERT INTO environments VALUES (NULL,'office corridor');",
    "INSERT INTO environments VALUES (NULL,'big hall');",

    "INSERT INTO test_datasets VALUES (NULL,'Wang2021');",
    "INSERT INTO test_datasets VALUES (NULL,'Zedigh2021');",

    "INSERT INTO training_datasets VALUES (NULL,'Wang2021 - Cable');",

    "INSERT INTO training_models VALUES (NULL,'CNN110');",

    "INSERT INTO additive_noise_methods VALUES (NULL,'Gaussian', 'Std', 0.01, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Gaussian', 'Std', 0.02, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Gaussian', 'Std', 0.03, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Gaussian', 'Std', 0.04, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Gaussian', 'Std', 0.05, "
    "'Mean', 0); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Collected', 'Scale', "
    "25, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Collected', 'Scale', "
    "49, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Collected', 'Scale', "
    "75, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Collected', 'Scale', "
    "105, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Rayleigh', 'Mode', "
    "0.0138, NULL, NULL); ",
    "INSERT INTO additive_noise_methods VALUES (NULL,'Rayleigh', 'Mode', "
    "0.0276, NULL, NULL); ",

    "INSERT INTO denoising_methods VALUES (NULL,'Moving Average Filter', 'N', "
    "3, NULL, NULL); ",
    "INSERT INTO denoising_methods VALUES (NULL,'Moving Average Filter', 'N', "
    "5, NULL, NULL); ",
]
