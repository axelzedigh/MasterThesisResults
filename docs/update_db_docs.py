# Run to update help docs based on current status of db.

from utils.db_utils import create_md__option_tables, \
    create_md__rank_test_tbl__meta_info, get_db_absolute_path, \
    create_md__full_rank_test__grouped


def update_helper_docs():
    path = "docs"
    database = get_db_absolute_path()
    create_md__option_tables(database, path)
    create_md__rank_test_tbl__meta_info(database, path)
    create_md__full_rank_test__grouped(database, path)


if __name__ == "__main__":
    update_helper_docs()
