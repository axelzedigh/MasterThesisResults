# Run to update help docs based on current status of db.

import os
from utils.db_utils import create_pre_processing_table_info_md, \
    create_rank_test_table_info_md


def update_preprocessing_docs():
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    path = "docs"
    database = os.path.join(project_dir, "database", "main.db")
    create_pre_processing_table_info_md(database, path)
    create_rank_test_table_info_md(database, path)


if __name__ == "__main__":
    update_preprocessing_docs()
