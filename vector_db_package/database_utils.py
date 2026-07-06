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
        etl_history = table_info.get("etl_history")

        logging_location = logging_info.get("logging_location")

        DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        model = sentence_transformer.get("model")

        # Ensure all information is valid
        if any(x is None for x in (host, port, database, schema, username, password, advisors, documents, chunks, advisor_documents, etl_history, logging_location)):
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


def ETL_History(conn: psycopg.Connection, 
                cur: psycopg.Cursor,
                schema: str,
                table: str,
                job_type: str, 
                run_status: str, 
                num_entries: int, 
                urls: list[str], 
                start_time: str, 
                end_time: str, 
                error_message: str, 
                log_file: str) -> None:
    
    query = sql.SQL("""
        INSERT INTO {}.{} (
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
    """).format(
        sql.Identifier(schema),
        sql.Identifier(table)
    )
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


def get_content_hash(text: str) -> str:
    normalized_text = text.strip()
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def insert_HTML(cur: psycopg.Cursor, 
                schema: str,
                documents_table: str,
                chunks_table: str,
                advisor_documents_table: str, 
                resp: requests.models.Response, 
                model: SentenceTransformer, 
                selected_advisors_list: list[str], 
                chunk_size: int, 
                overlap_size: int, 
                url: str) -> int:
    
    content_type = "HTML"
    
    content_hash = get_content_hash(text)
    model_name = model.model_card_data.base_model
    chunks = []

    soup = BeautifulSoup(resp.text, "html.parser")


def insert_PDF(cur: psycopg.Cursor,
               schema: str,
               documents_table: str,
               chunks_table: str,
               advisor_documents_table: str, 
               resp: requests.models.Response, 
               model: SentenceTransformer, 
               selected_advisors_list: list[str], 
               chunk_size: int, 
               overlap_size: int, 
               url: str) -> int:
    
    content_type = "PDF"
    model_name = model.model_card_data.base_model
    chunks = []

    pdf_bytes = resp.content
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    content_hash = get_content_hash(text)
    words  = text.split()

    # Part 1: Update Documents table to have new URL
    query = sql.SQL("""
            INSERT INTO {}.{} (
                source_type,
                url,
                content_hash
            )
            VALUES (%s, %s, %s)
            RETURNING document_id;
            """).format(
                sql.Identifier(schema),
                sql.Identifier(documents_table)
            )

    cur.execute(query,(content_type, url, content_hash))
    document_id = cur.fetchone()[0]

    step = chunk_size - overlap_size

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    # Part 2: Insert chunks from URL into Chunks table
    chunk_index = 0
    for chunk in chunks:
        # Vectorize text
        embedding = model.encode(chunk).tolist()
        word_count = len(chunk.split())

        query = sql.SQL("""
            INSERT INTO {}.{} (
                document_id,
                chunk_index,
                chunk_text,
                word_count,
                embedding,
                embedding_model
            )
            VALUES (%s, %s, %s, %s, %s, %s);
        """).format(
            sql.Identifier(schema),
            sql.Identifier(chunks_table)
        )

        cur.execute(
            query,
            (
                document_id,
                chunk_index,
                chunk,
                word_count,
                embedding,
                model_name
            )
        )

        chunk_index += 1

    # Part 3: Update Advisor Documents table to show access
    for advisor in selected_advisors_list:
        query = sql.SQL("""
                INSERT INTO {}.{} (
                    advisor_id,
                    document_id
                )
                VALUES (%s, %s);
                """).format(
                    sql.Identifier(schema),
                    sql.Identifier(advisor_documents_table)
                )
        cur.execute(
            query,
            (
                int(advisor),
                document_id
            )
        )
    
    return len(chunks)



def insert_Plain_Text(cur: psycopg.Cursor, 
                      schema: str,
                      documents_table: str,
                      chunks_table: str,
                      advisor_documents_table: str, 
                      resp: requests.models.Response, 
                      model: SentenceTransformer, 
                      selected_advisors_list: list[str], 
                      chunk_size: int, 
                      overlap_size: int, url: str) -> int:
    
    content_type = "Plain_Text"
    text = resp.text
    words = text.split()
    chunks = []
    content_hash = get_content_hash(text)
    model_name = model.model_card_data.base_model
    
    # Part 1: Update Documents table to have new URL
    query = sql.SQL("""
            INSERT INTO {}.{} (
                source_type,
                url,
                content_hash
            )
            VALUES (%s, %s, %s)
            RETURNING document_id;
            """).format(
                sql.Identifier(schema),
                sql.Identifier(documents_table)
            )
    cur.execute(query,(content_type, url, content_hash))
    document_id = cur.fetchone()[0]

    step = chunk_size - overlap_size

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    # Part 2: Insert chunks from URL into Chunks table
    chunk_index = 0
    for chunk in chunks:
        # Vectorize text
        embedding = model.encode(chunk).tolist()
        word_count = len(chunk.split())

        query = sql.SQL("""
            INSERT INTO {}.{} (
                document_id,
                chunk_index,
                chunk_text,
                word_count,
                embedding,
                embedding_model
            )
            VALUES (%s, %s, %s, %s, %s, %s);
        """).format(
            sql.Identifier(schema),
            sql.Identifier(chunks_table)
        )
        cur.execute(
            query,
            (
                document_id,
                chunk_index,
                chunk,
                word_count,
                embedding,
                model_name
            )
        )

        chunk_index += 1

    # Part 3: Update Advisor Documents table to show access
    for advisor in selected_advisors_list:
        query = sql.SQL("""
                INSERT INTO {}.{} (
                    advisor_id,
                    document_id
                )
                VALUES (%s, %s);
                """).format(
                    sql.Identifier(schema),
                    sql.Identifier(advisor_documents_table)
                )
        cur.execute(
            query,
            (
                int(advisor),
                document_id
            )
        )
    
    return len(chunks)
        


def insert_url_to_database(cur: psycopg.Cursor, 
                           schema: str,
                           table_info: dict,
                           model: SentenceTransformer, 
                           selected_advisors_list: list[str], 
                           url: str, 
                           chunk_size: int = 250, 
                           overlap_size: int = 40) -> int:
    
    try:
        resp = requests.get(url, timeout=10)
        content_type = resp.headers.get("Content-Type", "").lower()

        advisors_table = table_info.get("advisors")
        documents_table = table_info.get("documents")
        chunks_table = table_info.get("chunks")
        advisors_documents_table = table_info.get("advisor_documents")

        if "text/html" in content_type:
            return insert_HTML(
                cur,
                schema,
                documents_table,
                chunks_table,
                advisors_documents_table,
                resp,
                model,
                selected_advisors_list,
                chunk_size,
                overlap_size,
                url
            )

        elif "application/pdf" in content_type:
            return insert_PDF(
                cur,
                schema,
                documents_table,
                chunks_table,
                advisors_documents_table,
                resp,
                model,
                selected_advisors_list,
                chunk_size,
                overlap_size,
                url
            )

        elif "text/plain" in content_type:
            return insert_Plain_Text(
                cur,
                schema,
                documents_table,
                chunks_table,
                advisors_documents_table,
                resp,
                model,
                selected_advisors_list,
                chunk_size,
                overlap_size,
                url
            )

        else:
            raise Exception("URL is not a supported type (HTML, PDF, Plain Text)")
        
    except Exception as e:
        raise Exception(f"URL insertion failed: {e}")