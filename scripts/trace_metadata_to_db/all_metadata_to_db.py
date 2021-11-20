"""Run to update all metadata."""
from utils.db_utils import create_db_with_tables, initialize_table_data
from utils.trace_utils import insert_all_trace_metadata_depth_to_db, insert_all_trace_metadata_width_to_db


def update_trace_metadata_db(new_db=False):
    """
    Run to update all trace metadata.
    """
    database = "5m.db"
    if new_db:
        create_db_with_tables(database)
        initialize_table_data(database)
    insert_all_trace_metadata_depth_to_db(database)
    # insert_all_trace_metadata_width_to_db(database)


if __name__ == "__main__":
    update_trace_metadata_db(False)
