import sqlite3 as lite
import os
import numpy as np
import datetime
import sys


def initialise_sqlite3_db():
    """

    :return:
    """
    dirs = os.listdir()
    if "TerminationPoints.db" in dirs:
        print("Database already exists!")
        return
    else:
        con = lite.connect("TerminationPoints.db")
        con.execute(
            """
            CREATE TABLE RankTest(
            id INTEGER PRIMARY KEY,
            testing_dataset TEXT,
            environment TEXT,
            distance FLOAT,
            device INT, 
            training_model TEXT,
            keybyte INT,
            epoch INT,
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
            termination_point INT,
            date_added TEXT
            )
            """
        )
        con.commit()
        con.close()
    return


if __name__ == "__main__":
    initialise_sqlite3_db()
