from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_db_package import utils
import requests
import json
import sys
from pathlib import Path
import os
import logging
import time
import pandas as pd
import numpy as np
import psycopg
import argparse

from vector_db_package.database_utils import (
    get_config,
    get_connection,
    setup_logging
)

from vector_db_package.schema_utils import (
    get_advisors,
    advisor_exists,
    create_new_advisor,
    check_table_exists
)

def main(config_file: str):
    try:
        postgres_info, table_info, logging_info, sentence_transformer = get_config(config_file)

        postgres_schema = postgres_info.get("schema")
        advisors_table = table_info.get("advisors")
        documents_table = table_info.get("documents")
        chunks_table = table_info.get("chunks")
        advisors_documents_table = table_info.get("advisor_documents")
        etl_history_table = table_info.get("etl_history")

        conn, cur = get_connection(postgres_info)    
        model = SentenceTransformer(sentence_transformer.get("model"))
        log_file = setup_logging(logging_info)

        logging.info("Program Started")
        logging.info("Log file: %s", log_file)

        is_ended = False

        while not is_ended:
            print() 

    except Exception as e:
        print()
    finally:
        logging.info("Terminating Program")
        conn.close()
        cur.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", required=True, help="Configuration File Path")

    args = parser.parse_args()

    main(args.config_file)
