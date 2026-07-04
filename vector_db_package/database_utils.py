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
from schema_utils import check_table_exists

def get_config(config_file_path: str) -> tuple[dict, dict, dict, dict]:
    config_path = Path(config_file_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as config:
        config_info = yaml.safe_load(config)

    if config_info is None:
        raise ValueError(f"Config file is empty: {config_path}")


    # Read YAML file for information:
    with open(config_file_path, "r") as config:
        config_info = yaml.safe_load(config)

        postgres_info = config_info.get("postgres")
        table_info = config_info.get("tables")
        logging_info = config_info.get("logging")
        sentence_transformer = config_info.get("sentence_transformer")

        host = postgres_info.get("host")
        port = postgres_info.get("port")
        database = postgres_info.get("database")
        schema = postgres_info.get("schema")
        username = postgres_info.get("username")
        password = postgres_info.get("password")

        advisors = table_info.get("advisors")
        documents = table_info.get("documents")
        chunks = table_info.get("chunks")
        advisor_documents = table_info.get("advisor_documents")

        logging_location = logging_info.get("logging_location")

        DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        model = sentence_transformer.get("model")

        # Ensure all information is valid
        if any(x is None for x in (host, port, database, schema, username, password, advisors, documents, chunks, advisor_documents, logging_location)):
            raise ValueError("Critical Error: invalid config file information")
        
        if model is None:
            sentence_transformer["model"] = DEFAULT_EMBEDDING_MODEL

    return postgres_info, table_info, logging_info, sentence_transformer


def get_connection(postgres_info: dict) -> tuple[psycopg.Connection, psycopg.Cursor]:
    host = postgres_info.get("host")
    port = postgres_info.get("port")
    dbname = postgres_info.get("database")
    user = postgres_info.get("username")
    password = postgres_info.get("password")

    conn = psycopg.connect(
            host = host,
            port = port,
            dbname = dbname,
            user = user,
            password = password)
    
    cur = conn.cursor()
    return conn, cur


def setup_logging(logging_info: dict, log_name_prefix: str = "Database") -> str:
    # logging_info is already the logging section from the config
    log_location = logging_info.get("logging_location", "logs")
    log_level_name = logging_info.get("log_level", "INFO")

    log_level = getattr(logging, log_level_name.upper(), logging.INFO)

    log_dir = Path(log_location).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{log_name_prefix}_{timestamp}.log"

    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        force=True,
    )

    return str(log_file)


def ETL_History(conn: psycopg.Connection, cur: psycopg.Cursor, job_type: str, run_status, num_entries: int, urls: list[str], start_time: str, end_time: str, error_message: str, log_file: str) -> None:
    query = """
        INSERT INTO project.etl_runs (
            job_type,
            run_status,
            num_entries,
            urls,
            start_time,
            end_time,
            error_message,
            log_file
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    cur.execute(
        query,
        (
            job_type,
            run_status,
            num_entries,
            urls,
            start_time,
            end_time,
            error_message,
            log_file,
        ),
    )
    conn.commit()


def insert_HTML(cur: psycopg.Cursor, resp: requests.models.Response, model: SentenceTransformer, chunk_size: int, overlap_size: int) -> None:
    content_type = "HTML"
    soup = BeautifulSoup(resp.text, "html.parser")


def insert_PDF(cur: psycopg.Cursor, resp: requests.models.Response, model: SentenceTransformer, chunk_size: int, overlap_size: int) -> None:
    content_type = "PDF"

    pdf_bytes = resp.content
    pdf_file = io.BytesIO(pdf_bytes)

    reader = PdfReader(pdf_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""


def insert_Plain_Text(cur: psycopg.Cursor, resp: requests.models.Response, model: SentenceTransformer, chunk_size: int, overlap_size: int) -> None:
    content_type = "Plain_Text"

    text = resp.text
    words = text.split()
    chunks = []
    
    step = chunk_size - overlap_size

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))


    # Insert each chunk into Postgre
    for chunk in chunks:

        # Vectorize text
        embedding = model.encode(chunk).tolist()

        query = """
            INSERT INTO project.chunks (
                document_id,
                chunk_index,
                chunk_text,
                token_count,
                embedding,
                embedding model
            )
            VALUES (%s, %s, %s, %s, %s, %s);
        """
        cur.execute(
            query,
            (
                
            ),
        )
        


def insert_url_to_database(cur: psycopg.Cursor, model: SentenceTransformer, url: str, chunk_size: int = 250, overlap_size: int = 40) -> None:
    try:
        resp = requests.get(url, timeout=10)
        content_type = resp.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            insert_HTML(cur, resp, model, chunk_size, overlap_size)

        elif "application/pdf" in content_type:
            insert_PDF(cur, resp, model, chunk_size, overlap_size)

        elif "text/plain" in content_type:
            insert_Plain_Text(cur, resp, model, chunk_size, overlap_size)

        else:
            raise Exception("URL is not a supported type (HTML, PDF, Plain Text)")
        
    except Exception as e:
        raise Exception("URL insertion failed")