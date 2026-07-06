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
    run_advisor_query
)

def main(config_file: str):
    try:
        print("Start of News Overview Program:")
        postgres_info, table_info, logging_info, sentence_transformer = get_config(config_file)

        postgres_schema = postgres_info.get("schema")
        advisors_table = table_info.get("advisors")
        documents_table = table_info.get("documents")
        chunks_table = table_info.get("chunks")
        advisors_documents_table = table_info.get("advisor_documents")

        conn, cur = get_connection(postgres_info)    
        model = SentenceTransformer(sentence_transformer.get("model"))

    except Exception as e:
        print(f"Error: {e}")

    finally:
        print("Terminating Program")


if __name__ == "__main__":
    main()