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


def get_missing_columns(df: pd.DataFrame, table_coluns: list[str]) -> list[str]:
    df_columns = set(df.columns)
    table_columns_set = set(table_coluns)

    missing_columns = table_columns_set - df_columns

    return list(missing_columns)


def create_table(cur: psycopg.Cursor, schema: str, table: str, cols: list[str]) -> None:
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