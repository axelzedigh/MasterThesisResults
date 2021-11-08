# Run to create database (if no previous exists)
from utils.db_utils import create_db_with_tables, initialize_table_data


def main():
    create_db_with_tables("main.db")
    initialize_table_data("main.db")


if __name__ == "__main__":
    main()
