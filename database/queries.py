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
            parameter_1_id INTEGER,
            parameter_2_id INTEGER,
            
            FOREIGN KEY(parameter_1_id) REFERENCES Parameters(id),
            FOREIGN KEY(parameter_2_id) REFERENCES Parameters(id)
            );
            """
QUERY_CREATE_TABLE_DENOISING_METHODS = """
            CREATE TABLE Denoising_Methods(
            id INTEGER PRIMARY KEY,
            denoising_method TEXT,
            parameter_1_id INTEGER,
            parameter_2_id INTEGER,
            
            FOREIGN KEY(parameter_1_id) REFERENCES Parameters(id),
            FOREIGN KEY(parameter_2_id) REFERENCES Parameters(id)
            );
            """

QUERY_CREATE_TABLE_PARAMETERS = """
            CREATE TABLE Parameters(
            id INTEGER PRIMARY KEY,
            name TEXT,
            value FLOAT
            );
            """

QUERY_CREATE_TABLE_RANK_TEST = """
            CREATE TABLE Rank_Test(
                id INTEGER PRIMARY KEY,
                test_dataset_id INTEGER,
                training_dataset_id INTEGER,
                environment_id INTEGER,
                distance FLOAT NOT NULL,
                device INT NOT NULL, 
                training_model_id INTEGER,
                keybyte INT NOT NULL,
                epoch INT NOT NULL,
                additive_noise_method_id INTEGER,
                denoising_method_id INTEGER,
                termination_point INT NOT NULL,
                average_rank INT NOT NULL,
                date_added TEXT NOT NULL,

                FOREIGN KEY(test_dataset_id) REFERENCES Test_Datasets(test_dataset_id),
                FOREIGN KEY(training_dataset_id) REFERENCES Training_Datasets(training_dataset_id),
                FOREIGN KEY(environment_id) REFERENCES Environments(environments_id),
                FOREIGN KEY(training_model_id) REFERENCES Training_Model(training_model_id),
                FOREIGN KEY(additive_noise_method_id) REFERENCES Additive_Noise_Methods(additive_noise_method_id),
                FOREIGN KEY(denoising_method_id) REFERENCES Denoising_Method(denoising_method_id)
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
                full_additive_noise_methods.additive_noise_method AS additive_noise_method,
                full_additive_noise_methods.parameter_1 AS additive_noise_method_parameter_1,
                full_additive_noise_methods.parameter_1_value AS additive_noise_method_parameter_1_value,
                full_additive_noise_methods.parameter_2 AS additive_noise_method_parameter_2,
                full_additive_noise_methods.parameter_2_value AS additive_noise_method_parameter_2_value,
                full_denoising_methods.denoising_method AS denoising_method,
                full_denoising_methods.parameter_1 AS denoising_method_parameter_1,
                full_denoising_methods.parameter_1_value AS denoising_method_parameter_1_value,
                full_denoising_methods.parameter_2 AS denoising_method_parameter_2,
                full_denoising_methods.parameter_2_value AS denoising_method_parameter_2_value,
                Rank_Test.termination_point,
                Rank_Test.average_rank,
                Rank_Test.date_added
            FROM
                Rank_Test
            INNER JOIN Test_Datasets ON Test_Datasets.id = Rank_Test.id
            INNER JOIN Training_Datasets ON Training_Datasets.id = Rank_Test.id
            INNER JOIN Environments ON Environments.id = Rank_TEst.id
            INNER JOIN Training_Models on Training_Models.id = Rank_Test.id
            LEFT JOIN full_additive_noise_methods ON full_additive_noise_methods.id = Rank_test.id
            LEFT JOIN full_denoising_methods ON full_denoising_methods.id = Rank_Test.id;
            """

QUERY_CREATE_VIEW_FULL_ADDITIVE_NOISE_METHODS = """
            CREATE VIEW full_additive_noise_methods
            AS
            SELECT
                AddNoise.id,
                AddNoise.additive_noise_method,
                Parameter_1.name AS parameter_1,
                Parameter_1.value AS parameter_1_value,
                Parameter_2.name AS parameter_2,
                Parameter_2.value AS parameter_2_value
            FROM 
                Additive_Noise_Methods AddNoise
            JOIN Parameters Parameter_1 
            ON AddNoise.parameter_1_id = Parameter_1.id
            JOIN Parameters Parameter_2
            ON AddNoise.parameter_2_id = Parameter_2.id
            """

QUERY_CREATE_VIEW_FULL_DENOISING_METHODS = """
            CREATE VIEW full_denoising_methods
            AS
            SELECT
                Denoising.id,
                Denoising.denoising_method,
                Parameter_1.name AS parameter_1,
                Parameter_1.value AS parameter_1_value,
                Parameter_2.name AS parameter_2,
                Parameter_2.value AS parameter_2_value
            FROM 
                Denoising_Methods Denoising
            INNER JOIN Parameters Parameter_1 
            ON Denoising.parameter_1_id = Parameter_1.id 
            INNER JOIN Parameters Parameter_2
            ON Denoising.parameter_2_id = Parameter_2.id 
            """
