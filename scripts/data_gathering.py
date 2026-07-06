from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
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
    create_new_advisor,
    check_table_exists
)

def main(config_file: str):

    run_status = "Failure"
    is_ended = False
    start_time = str(datetime.datetime.now())
    num_URLs = 0
    num_chunks = 0
    URLs = []
    error_message = None
    job_type = "ingestion"

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

        while not is_ended:
            url = input("Enter URL or type 'quit' to terminate program: ").strip()

            if url.lower() in {"quit", "q", "exit"}:
                break

            try:
                print("Available Advisors: ")
                existing_advisors = get_advisors(cur, postgres_schema, advisors_table)
                for idx, (advisor_id, advisor_name) in enumerate(existing_advisors, start=1):
                    print(f"[{idx}]: {advisor_name}")

                selected_advisors = input("Enter advisor numbers, comma-separated, or 'all': ").strip()

                if selected_advisors.lower() == "all":
                    selected_advisors_list = [advisor_id for advisor_id, _ in existing_advisors]
                else:
                    selected_advisors_list = []
                    # Remove invalid selected advisors
                    for item in selected_advisors.split(","):
                        item = item.strip()

                        if not item.isdigit():
                            print(f"Dropping invalid advisor selection: {item}")
                            continue

                        idx = int(item)

                        if 1 <= idx <= len(existing_advisors):
                            advisor_id = existing_advisors[idx - 1][0]
                            selected_advisors_list.append(advisor_id)
                        else:
                            print(f"Dropping out-of-range advisor selection: {item}")

                    if not selected_advisors_list:
                        print(f"Selected advisors are invalid, loop reseting.")
                        raise Exception("No valid advisors selected")

                print(f"Starting Ingestion for {url}")
                try:
                    # Do Insertion
                    CHUNK_SIZE = 250
                    OVERLAP_SIZE = 40
                    num_chunks += insert_url_to_database(cur, 
                                                         postgres_schema, 
                                                         table_info, 
                                                         model, 
                                                         selected_advisors_list, 
                                                         url, 
                                                         CHUNK_SIZE, 
                                                         OVERLAP_SIZE)
                    
                except Exception as e:
                    logging.error(f"Attempt to insert {url} failed: {e}")
                    raise

                num_URLs += 1
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
        end_time = str(datetime.datetime.now())

        if conn is not None and cur is not None:
            try:
                ETL_History(conn, 
                            cur, 
                            postgres_schema,
                            etl_history_table,
                            job_type,
                            run_status,
                            num_URLs,
                            URLs,
                            start_time,
                            end_time,
                            error_message,
                            log_file)
                
            except Exception as e:
                logging.error(f"Failed to write to ETL History: {e}")

            logging.info("Terminating Program")
            cur.close()
            conn.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Configuration File Path")
    args = parser.parse_args()

    main(args.config_file)
