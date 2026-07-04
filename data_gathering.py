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
import datetime

from vector_db_package.database_utils import (
    get_config,
    get_connection,
    setup_logging,
    ETL_History,
    insert_url_to_database
)

from vector_db_package.schema_utils import (
    get_advisors,
    advisor_exists,
    create_new_advisor,
    check_table_exists
)

def main(config_file: str):

    run_status = "Failure"
    is_ended = False
    start_time = str(datetime.now())
    num_input = 0
    URLs = []
    error_message = None
    job_type = "ingestion"

    try:
        postgres_info, table_info, logging_info, sentence_transformer = get_config(config_file)
        conn, cur = get_connection(postgres_info)    
        model = SentenceTransformer(sentence_transformer.get("model"))
        log_file = setup_logging(logging_info)

        logging.info("Program Started")
        logging.info("Log file: %s", log_file)

        while not is_ended:
            url = input("Enter URL or type 'quit' to terminate program: ").strip()

            if url.lower() in {"quit", "q", "exit"}:
                break

            try:
                print("Avaliable Advisors: ")
                existing_advisors = get_advisors(cur)
                for idx, advisor in enumerate(existing_advisors, start=1):
                    print(f"[{idx}]: {advisor}")

                selected_advisors = input("Enter advisor numbers, comma-separated, or 'all': ").strip()
                selected_advisors_list = selected_advisors.split(",")

                # Remove invalid selected advisors
                invalid_advisors = [x for x in selected_advisors if x not in existing_advisors]

                if invalid_advisors:
                    print(f"Dropping invalid Advisors: {invalid_advisors}")
                    for invalid in invalid_advisors:
                        selected_advisors.remove(invalid)

                if not selected_advisors:
                    print(f"Selected advisors are invalid, loop reseting.")
                    raise Exception("No valid advisors selected")

                print(f"Starting Ingestion for {url}")
                try:
                    # Do Insertion
                    insert_url_to_database(conn, cur, url)
                    
                except Exception as e:
                    logging.error(f"Attempt to insert {url} failed: {e}")
                    raise

                num_input += 1
                URLs.append(url)       
                conn.commit()

            except Exception as e:
                conn.rollback()
                continue
            
        run_status = "Success"

    except Exception as e:
        logging.error(f"ETL Run Failed: {e}")
        error_message = str(e)
        run_status = "Failure"

    finally:
        end_time = str(datetime.now())

        if conn is not None and cur is not None:
            try:
                ETL_History(conn, 
                            cur, 
                            job_type,
                            run_status,
                            num_input,
                            URLs,
                            start_time,
                            end_time,
                            error_message,
                            log_file)
            except Exception as e:
                logging.error(f"Failed to write to ETL History: {e}")

            logging.info("Terminating Program")
            conn.close()
            cur.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Configuration File Path")
    args = parser.parse_args()

    main(args.config_file)
