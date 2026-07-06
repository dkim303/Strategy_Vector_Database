import sqlalchemy as sa
import json
import pandas as pd
import numpy as np
import pyodbc
import yaml
import requests
import psycopg
import io
from pypdf import PdfReader
from psycopg import sql
from bs4 import BeautifulSoup
import logging
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import hashlib
import ollama
import argparse

from vector_db_package.database_utils import (
    get_config,
    get_connection,
    setup_logging
)

from vector_db_package.schema_utils import (
    get_advisors,
    create_new_advisor,
    check_table_exists
)

from vector_db_package.advising_utils import (
    run_advisor_query,
    get_advisor_desc
)

from vector_db_package.ollama_utils import (
    check_ollama_connection,
)

def main(config_file_name: str):
    try:
        print("Start of Query Program:")
        postgres_info, table_info, logging_info, sentence_transformer, ollama_info = get_config(config_file_name)

        postgres_schema = postgres_info.get("schema")
        advisors_table = table_info.get("advisors")
        documents_table = table_info.get("documents")
        chunks_table = table_info.get("chunks")
        advisors_documents_table = table_info.get("advisor_documents")
        advisors_table = table_info.get("advisors")

        conn, cur = get_connection(postgres_info)    
        ST_model = SentenceTransformer(sentence_transformer.get("model"))
        ollama_model = ollama_info.get("model")

        K = int(input("Enter K value (Reccomended 30): "))
        advisor_id = int(input("Select Advisor: "))
        advisor_desc = get_advisor_desc(cur,
                                        postgres_schema,
                                        advisors_table,
                                        advisor_id)


        query_text = input("Enter Prompt: ").strip()

        top_K_df = run_advisor_query(cur,
                            postgres_info,
                            table_info,
                            ST_model,
                            advisor_id,
                            query_text,
                            K)
                            
        """
            Transform DF into usable strings
        """


        chat_message = [
                {"role": "system", "content": f"{advisor_desc}"},
                {"role": "user", "content": f"{query_text}"}
            ]
        
        check_ollama_connection()
        response = ollama.chat(
            model=ollama_model,
            messages=chat_message
        )

        answer = response["message"]["content"]
        print(answer)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print("Terminating Program")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Configuration File Path")
    args = parser.parse_args()

    main(args.config_file)
