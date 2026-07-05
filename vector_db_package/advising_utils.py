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

def cosine_similarity_matrix(query_vec, embedding_matrix):
    query_vec = np.array(query_vec, dtype=np.float32)
    embedding_matrix = np.array(embedding_matrix, dtype=np.float32)

    query_norm = np.linalg.norm(query_vec)
    embedding_norms = np.linalg.norm(embedding_matrix, axis=1)

    return (embedding_matrix @ query_vec) / (embedding_norms * query_norm)

def run_advisor_query(cur: psycopg.Cursor,
                      postgres_info: dict,
                      table_info: dict,
                      model: SentenceTransformer,
                      advisor_id: int,
                      query_text: str,
                      top_k: int = 30) -> pd.DataFrame:

    model_name = model.model_card_data.base_model
        
    query_embedding = model.encode(query_text).tolist()

    query = """
        SELECT
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.chunk_text,
            c.embedding,
            c.embedding_model,
            d.url
        FROM project.chunks c
        JOIN project.documents d
            ON c.document_id = d.document_id
        JOIN project.advisor_documents ad
            ON d.document_id = ad.document_id
        WHERE ad.advisor_id = %s
        AND c.embedding_model = %s
        AND c.embedding IS NOT NULL;
    """

    cur.execute(query, (advisor_id, model_name))
    rows = cur.fetchall()

    embeddings = [row["embedding"] for row in rows]
    scores = cosine_similarity_matrix(query_embedding, embeddings)
