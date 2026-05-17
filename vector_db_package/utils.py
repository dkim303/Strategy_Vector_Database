from bs4 import BeautifulSoup
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from io import BytesIO
import requests
import json
import sys
from pathlib import Path


def menu():
	print(f"Enter 1 to make query.")
	print("Enter 2 to add new URL to database.")
	print("Enter q to quit.")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def add_vectorized_website(url, embeddings, dictionary):
	response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
	response.raise_for_status()

	soup = BeautifulSoup(response.text, "html.parser")

	paragraphs = soup.find_all("p")

	vectors = list(embeddings)

	# for each paragraph on the page, turn the paragraph into a vector, then log it
	for p in paragraphs:
		text = p.get_text(strip=True)

		if not text:
			continue

		vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
		vectors.append(vector)

		dictionary[len(dictionary)] = { "url": url, "type":"website", "text": text}

	embeddings = np.array(vectors, dtype=np.float32)
	return embeddings

def chunk_text(text, max_words=500):
	words = text.split()
	chunks = []

	for i in range(0, len(words), max_words):
		chunk = " ".join(words[i:i + max_words])
		chunks.append(chunk)

	return chunks

def add_vectorized_PDF(url, embeddings, dictionary):
	response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
	response.raise_for_status()

	pdf = BytesIO(response.content)
	reader = PdfReader(pdf)

	vectors = list(embeddings)

	for page_num, page in enumerate(reader.pages):
		text = page.extract_text()
		if not text:
			continue

		chunks = chunk_text(text)

		for chunk_num, chunk in enumerate(chunks):
			vector = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
			vectors.append(vector)
			dictionary[len(dictionary)] = {"url": url, "type":"pdf", "page number":page_num+1,"text": chunk}

	embeddings = np.array(vectors, dtype=np.float32)
	return embeddings

def update_files(e_name, d_name, embedding_matrix, data):
	np.save(e_name, embedding_matrix)

	with open(d_name, "w") as d_fp:
		json.dump(data, d_fp)
	
def check_duplicate(url, dictionary):
	for index in range(len(dictionary)):
		if dictionary[index]["url"] == url:
			return True
		
	return False

def cosine_compare(query_vector, target_vector):
	similarity = (query_vector @ target_vector) / (np.linalg.norm(query_vector) * (np.linalg.norm(target_vector)))
	return similarity

def search_database(query_vector, embeddings, data, K=10):
	# case of trying to search empty database
	if embeddings.shape[0] == 0:
		print("Error: database is empty.")
		return []

	index_cosine_pairs = []
	index = 0

	for vector in embeddings:
		similarity_coefficient = cosine_compare(query_vector, vector)
		index_cosine_pairs.append((index,similarity_coefficient))
		index += 1

	index_cosine_pairs.sort(key=lambda x: x[1], reverse=True)

	results = index_cosine_pairs[0:K]
	return results

def interface(embeddings, metadata, e_name, d_name):
	# give user options on terminal
	menu()
	selected_num = input()

	try:
		match str(selected_num):

			# Making A Query
			case "1":
				query = input("Enter query: ")
				query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

				# determine K value for how many results to return
				K = int(input("Enter K-value for top-K search: "))
				if K <= 0:
					print("Error: K-value must be > 0.")
					return True, embeddings

				# make query here
				top_K_indicies = search_database(query_vector, embeddings, metadata, K)

				return True, embeddings
			
			case "2":
				should_continue = True
				while should_continue:
					mode = input("Enter 0 to log a Website, or Enter 1 to log a PDF: ")

					# Website URL Entering
					if mode == "0":
						url = input("Enter new Website URL: ")
						response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
						response.raise_for_status()
						
						is_duplicate = check_duplicate(url, metadata)
						if is_duplicate == True:
							raise ValueError("Error: URL already logged.")
						
						embeddings = add_vectorized_website(url, embeddings, metadata)
						update_files(e_name, d_name, embeddings, metadata)

						return True, embeddings
						
					# PDF URL Entering
					elif mode == "1":
						url = input("Enter new PDF URL: ")
						response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
						response.raise_for_status()
						
						is_duplicate = check_duplicate(url, metadata)
						if is_duplicate == True:
							raise ValueError("Error: URL already logged.")
						
						embeddings = add_vectorized_PDF(url, embeddings, metadata)
						update_files(e_name, d_name, embeddings, metadata)

						return True, embeddings

					# Invalid URL Type
					else:
						print("Invalid URL type entered, entry skipped.")
						return True, embeddings

			# Exit program
			case "q":
				print("Quitting Program.")
				return False, embeddings

			# Invalid Option Entered
			case _:
				raise ValueError("Error: invalid command given.")

	except ValueError as e:
		print(e)
		return True, embeddings
	except requests.exceptions.RequestException as e:
		print(e)
		return True, embeddings