from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from vector_db_package import utils
import requests
import json
import sys
from pathlib import Path
import os
import logging
import time

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    # initialize paths and files for logging
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    #makes logs directory if it DNE, exists_ok=True prevents error if it already exis
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    LOG_FILE = os.path.join(LOG_DIR, f"Database_{timestamp}.log")
    logging.basicConfig(
        filename=LOG_FILE,
        level = logging.INFO,
        format = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
    )
    logging.info("Program Started")

    print("Load in embeddings matrix and metadata")
    e_name = input("Enter embeddings file name (ending with .npy): ")
    d_name = input("Enter metadata file name (ending with .json): ")

    try:
        embeddings_path = Path(e_name)
        data_path = Path(d_name)

        # check for mismatch in file existance
        if (embeddings_path.exists() == False and data_path.exists() == True):
            logging.critical("Entered embeddings file exist without valid data file")
            raise Exception("Error: Embeddings file does not exist but Data file exists.")
        elif (embeddings_path.exists() == True and data_path.exists() == False):
            logging.critical("Entered data file exist without valid embeddings file")
            raise Exception("Error: Data file does not exist but Embeddings file exists.")   
        
        # later on add function to check if all items in the 2 files match correctly

        if embeddings_path.exists():
            embeddings_matrix = np.load(e_name)
        # case of creating empty matrix, dim-384, num vectors is 0
        else:
            embeddings_matrix = np.empty((0,384), dtype=np.float32)

        if data_path.exists():
            with open(d_name) as data_fp:
                data = json.load(data_fp)
                data = {int(k): v for k, v in data.items()}
        else:
        # case of creating empty data list, no data recorded yet in JSON format
            data = {}

    except Exception as e:
        print(e)
        sys.exit(1)

    should_continue = True
    while should_continue:
        # handles all user inputs for command sequences and program quitting
        should_continue, embeddings = utils.interface(embeddings_matrix, data, e_name, d_name)
            
    logging.info("Program Terminated")
    