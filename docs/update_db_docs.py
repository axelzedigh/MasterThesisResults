# Run to update help docs based on current status of db.

import os
from utils.db_utils import create_pre_processing_table_info_md, \
    create_rank_test_table_info_md, get_db_absolute_path


def update_helper_docs():
    path = "docs"
    database = get_db_absolute_path()
    create_pre_processing_table_info_md(database, path)
    create_rank_test_table_info_md(database, path)


if __name__ == "__main__":
    update_helper_docs()
