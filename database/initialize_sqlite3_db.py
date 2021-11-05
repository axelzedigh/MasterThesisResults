import sqlite3 as lite
import os
import datetime


def initialise_sqlite3_db(database="TerminationPoints.db") -> None:
    """

    :return:
    """
    dirs = os.listdir()
    if database in dirs:
        print("Database already exists!")
        return
    else:
        con = lite.connect("TerminationPoints.db")
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE RankTest(
            id INTEGER PRIMARY KEY,
            testing_dataset TEXT NOT NULL,
            environment TEXT NOT NULL,
            distance FLOAT NOT NULL,
            device INT NOT NULL, 
            training_model TEXT NOT NULL,
            keybyte INT NOT NULL,
            epoch INT NOT NULL,
            additive_noise_method TEXT,
            additive_noise_parameter_1 TEXT,
            additive_noise_parameter_1_value FLOAT,
            additive_noise_parameter_2 TEXT,
            additive_noise_parameter_2_value FLOAT,
            denoising_method TEXT,
            denoising_method_parameter_1 TEXT,
            denoising_method_parameter_1_value FLOAT,
            denoising_method_parameter_2 TEXT,
            denoising_method_parameter_2_value FLOAT,
            termination_point INT NOT NULL,
            date_added TEXT NOT NULL
            )
            """
        )
        con.commit()
        con.close()
    return


def insert_data(
        database="TerminationPoints.db",
        testing_dataset: str = "",
        environment: str = "office",
        distance: float = 15,
        device: int = 8,
        training_model: str = "CNN 110",
        keybyte: int = 0,
        epoch: int = 100,
        additive_noise_method: str = None,
        additive_noise_parameter_1: str = None,
        additive_noise_parameter_1_value: float = None,
        additive_noise_parameter_2: str = None,
        additive_noise_parameter_2_value: float = None,
        denoising_method: str = None,
        denoising_method_parameter_1: str = None,
        denoising_method_parameter_1_value: float = None,
        denoising_method_parameter_2: str = None,
        denoising_method_parameter_2_value: float = None,
        termination_point: int = 9999,
) -> None:
    """

    :param database: The database-file to write to. Standard is "TerminationPoints.db".
    :param testing_dataset: The dataset (either 'Wang2021' or 'Zedigh2021')
    :param environment:
    The environment the testing traces were collected. Either office or KTH Kista Hall.
    :param distance: Distance between device under test and antenna.
    :param device: Device under test.
    :param training_model: The deep learning model used. Usually 'CNN 110'.
    :param keybyte: The keybyte trained and tested. Between 0-15.
    :param epoch: The epoch of the DL model. Between 1-100.
    :param additive_noise_method:
    The additive noise method used during model training. E.g. Gaussian, Collected, Rayleigh etc.
    :param additive_noise_parameter_1:
    First parameter of the additive noise method. E.g Mean, Scale, Mode etc.
    :param additive_noise_parameter_1_value:
    Value of the first additive noise parameter.
    :param additive_noise_parameter_2:
    Second parameter of the additive noise method. E.g. Std, Translation etc.
    :param additive_noise_parameter_2_value:
    Value of the second additive noise parameter.
    :param denoising_method:
    Denoising method used. E.g. Moving average filter, CDAE, Weiner filter.
    :param denoising_method_parameter_1:
    First parameter of the denoising parameter. E.g. N, etc.
    :param denoising_method_parameter_1_value: Value of the first denoising method.
    :param denoising_method_parameter_2: Second parameter of the denoising method.
    :param denoising_method_parameter_2_value: Value of the second denoising parameter.
    :param termination_point: Termination point from rank test. Dependent variable!
    """
    initialise_sqlite3_db(database)
    con = lite.connect(database)
    cur = con.cursor()
    date_added = str(datetime.datetime.today())

    cur.execute("INSERT INTO RankTest VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (testing_dataset,
                 environment,
                 distance,
                 device,
                 training_model,
                 keybyte,
                 epoch,
                 additive_noise_method,
                 additive_noise_parameter_1,
                 additive_noise_parameter_1_value,
                 additive_noise_parameter_2,
                 additive_noise_parameter_2_value,
                 denoising_method,
                 denoising_method_parameter_1,
                 denoising_method_parameter_1_value,
                 denoising_method_parameter_2,
                 denoising_method_parameter_2_value,
                 termination_point,
                 date_added)
                )
    con.commit()
    con.close()


def fetchall(database="TerminationPoints.db"):
    """

    :param database:
    :return:
    """
    con = lite.connect(database)
    cur = con.cursor()
    all_data = cur.execute("SELECT * FROM RankTest;").fetchall()
    con.close()
    return all_data


if __name__ == "__main__":
    initialise_sqlite3_db()
