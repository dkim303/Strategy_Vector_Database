import sqlalchemy as sa
import json
import pandas as pd
import numpy as np
import pyodbc
import yaml
import requests
import psycopg
import io
from psycopg import sql
import logging
from sentence_transformers import SentenceTransformer

def check_table_exists(cur: psycopg.Cursor, schema: str, table: str) -> bool:
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        );
    """

    cur.execute(query, (schema, table))
    return cur.fetchone()[0]


def get_table_columns(cur: psycopg.Cursor, schema: str, table: str) -> list[str]:
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
        ORDER BY ordinal_position;
    """

    cur.execute(query, (schema, table))
    rows = cur.fetchall()

    return [row[0] for row in rows]


def add_new_col(cur: psycopg.Cursor, table: str, col: str, type: str) -> None:
    query = """
        ALTER TABLE %s
    """
    cur.execute(query, (table))


def ensure_valid_table(cur: psycopg.Cursor, postgres_schema: str, table: str, table_type: str, present_cols: list[str]) -> None:
    expected_columns = []
    missing_cols = []

    match table_type:
        case "advisor":
            expected_columns = {
                "advisor_id": "SERIAL PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "description": "TEXT",
                "config": "JSONB",
            }
        
        case "documents":
            expected_columns = {
                "document_id": "SERIAL PRIMARY KEY",
                "url": "TEXT",
                "source_type": "TEXT",
                "content_hash": "TEXT UNIQUE",
            }
        
        case "advisor_documents":
            expected_columns = {
                "advisor_id": "INTEGER",
                "document_id": "INTEGER",
                "weight": "DOUBLE PRECISION DEFAULT 1.0",
                "relevance_note": "TEXT",
            }

        case "chunks":
            expected_columns = {
                "chunk_id": "SERIAL PRIMARY KEY",
                "document_id": "INTEGER",
                "chunk_index": "INTEGER",
                "chunk_text": "TEXT",
                "token_count": "INTEGER",
                "embedding": "DOUBLE PRECISION[]",
                "embedding_model": "TEXT",
            }
        
        case "etl_history":
            expected_columns = {
                "job_type": "TEXT",
                "run_status": "TEXT",
                "num_entries": "INTEGER",
                "urls": "TEXT[]",
                "start_time": "TIMESTAMP",
                "end_time": "TIMESTAMP",
                "error_message": "TEXT",
                "log_file": "TEXT",
            }

        case _:
            logging.error(f"Invalid table_type {table_type} input for missing columns check")
            raise Exception(f"Invalid table_type {table_type} input for missing columns check")
        
    present_cols_set = set(present_cols)
    missing_cols = [
        col for col in expected_columns
        if col not in present_cols_set
        ]

    if len(missing_cols) == 0:
        return
    else:
        logging.info(f"Creating missing columns for {table}: {missing_cols}")
        for missing_col in missing_cols:
            datatype = expected_columns[missing_col]
            add_new_col(cur, table, missing_col, datatype)
        


def create_table(cur: psycopg.Cursor, schema: str, table: str, table_type: str) -> None:
    query = """
        
    """


def get_advisors(
    cur: psycopg.Cursor,
    schema: str,
    advisor_table: str
) -> list[tuple[str, str]]:
    query = sql.SQL("""
        SELECT advisor_id, name
        FROM {}.{}
        ORDER BY name;
    """).format(
        sql.Identifier(schema),
        sql.Identifier(advisor_table)
    )

    cur.execute(query)
    rows = cur.fetchall()

    return [(row[0], row[1]) for row in rows]


def create_new_advisor(cur: psycopg.Cursor, 
                       conn: psycopg.Connection,
                       schema: str,
                       table: str,
                       name: str, 
                       description: str = None, 
                       config: str = None) -> int:
    try:
        if name is None or name.strip() == "":
            raise ValueError("Empty Name")
        
        if isinstance(config, dict):
            config = json.dumps(config)
        
        query = sql.SQL("""
            INSERT INTO {}.{} (name, description, config)
            VALUES (%s, %s, %s)
            RETURNING advisor_id;
        """).format(
            sql.Identifier(schema),
            sql.Identifier(table)
        )

        cur.execute(query, (name, description, config))
        advisor_id = cur.fetchone()[0]
        conn.commit()
        logging.info(f"Created new advisor: {name}")
        return advisor_id
        
    except Exception as e:
        logging.error(f"Failed to create new advisor: {e}")
        conn.rollback()
        raise