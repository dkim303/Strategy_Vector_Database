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

def check_ollama_connection() -> None:
    try:
        ollama.list()
    except Exception:
        raise ConnectionError("Ollama Connection Failed")