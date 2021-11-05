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


def insert_data(
        testing_dataset,
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
        date_added,
):
    """

    :param testing_dataset:
    :param environment:
    :param distance:
    :param device:
    :param training_model:
    :param keybyte:
    :param epoch:
    :param additive_noise_method:
    :param additive_noise_parameter_1:
    :param additive_noise_parameter_1_value:
    :param additive_noise_parameter_2:
    :param additive_noise_parameter_2_value:
    :param denoising_method:
    :param denoising_method_parameter_1:
    :param denoising_method_parameter_1_value:
    :param denoising_method_parameter_2:
    :param denoising_method_parameter_2_value:
    :param termination_point:
    :param date_added:
    """
    con = lite.connect("TerminationPoints.db")
    if testing_dataset:


    insert_string = f"""
    INSERT INTO RankTest VALUES(
    NULL,
    '{testing_dataset}',
    '{environment}',
    {distance},
    {device},
    '{training_model}',
    {keybyte},
    {epoch},
    '{additive_noise_method}',
    '{additive_noise_parameter_1}',
    {additive_noise_parameter_1_value},
    '{additive_noise_parameter_2}',
    {additive_noise_parameter_2_value},
    '{denoising_method}',
    '{denoising_method_parameter_1}',
    {denoising_method_parameter_1_value},
    '{denoising_method_parameter_2}',
    {denoising_method_parameter_2_value},
    {termination_point},
    '{date_added}'
    )
    """

    insert_string.replace("\n", "")
    con.execute(
        insert_string
    )


if __name__ == "__main__":
    initialise_sqlite3_db()
