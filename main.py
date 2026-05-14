from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import funcs
import requests
import json

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":

    print("Load in embeddings matrix and metadata")
    e_name = input("Enter embeddings file name: ")
    if 

    d_name = input("Enter metadata file name: ")

    funcs.menu()

    selected_num = input()

    try:
        match selected_num:
            case 1:
                query = input("Enter query: ")
                query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            case 2:
                url = input("Enter new URL: ")
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if response.raise_for_status == False:
                    raise ValueError("Error: invalid URL given.")
            case 3:
                embeddings_file_name = input("Enter the name of the embeddings file for export")
                metadata_file_name = input("Enter the name of the metadata file for export")

                np.save(embeddings_file_name, embeddings_matrix)
                        
                with open(metadata_file_name) as metadata_fp:
                    json.dump(metadata_file_name, metadata)

            case _:
                raise ValueError("Error: invalid command given.") 
    except ValueError as e:
        print(e)
            
    