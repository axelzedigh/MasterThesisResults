from docs.update_db_docs import update_helper_docs
from utils.db_utils import copy_rank_test_from_db1_to_db2

if __name__ == "__main__":
    database_from = "db_to_extract/main_4.db"
    database_to = "main.db"
    copy_rank_test_from_db1_to_db2(
        database_from=database_from,
        database_to=database_to,
    )
    update_helper_docs()
