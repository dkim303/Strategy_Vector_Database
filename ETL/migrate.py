import sqlalchemy as sa
import json
import pandas as pd
import numpy as np
import pyodbc
import yaml
import requests

"""
Script will migrate data from specified embeddings file and vectors file into database
"""

def main(config_file: str) -> int:
    run_status = "Failure"
    num_migrated = 0

    try:
        # Read YAML file for information:
        with open(config_file, "r") as config:
            config_info = yaml.safe_load(config)

            sql_database = config_info.get("database_name")
            sql_embeddings_table = config_info.get("e_table")
            sql_paragraphs_table = config_info.get()
            sql_uid = config_info.get("username")
            sql_pwd = config_info.get("password")

    except Exception as e:
        print(e)

    finally:
        print(f"Migration End Status: {run_status}")
        print(f"Total data points migrated: {num_migrated}")

if __name__ == "__main__":
    main()