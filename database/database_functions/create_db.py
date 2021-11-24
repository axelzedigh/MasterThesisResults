""""Run to create database (if no previous exists)"""

from utils.db_utils import create_db_with_tables, initialize_table_data


def main(database="main.db"):
    create_db_with_tables(database)
    initialize_table_data(database)


if __name__ == "__main__":
    pass
    # database = sys.argv[1]
    # database = "main_2.db"
    # main(database)
    # database = "main_3.db"
    # main(database)
    # database = "main_4.db"
    # main(database)
