# Run to update help docs from the current db.
import os
from utils.db_utils import create_pre_processing_table_info_file


def update_preprocessing_docs():
    project_dir = os.getenv("MASTER_THESIS_RESULTS")
    database = os.path.join(project_dir, "database", "main.db")
    create_pre_processing_table_info_file(database, "docs")


if __name__ == "__main__":
    update_preprocessing_docs()
