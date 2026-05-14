from bs4 import BeautifulSoup
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from io import BytesIO

def menu():
	print(f"Enter 1 to make query (default database)")
	print("Enter 2 to add new URL to database")
	print("Enter 3 to get files of the embeddings and the mappings")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def add_vectorized_website(url, vectors, dictionary):
	response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
	response.raise_for_status()

	soup = BeautifulSoup(response.text, "html.parser")

	paragarphs = soup.find_all("p")

	# for each paragraph on the page, turn the paragraph into a vector, then log it
	for p in paragarphs:
		text = p.get_text(strip=True)

		vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
		vectors.append(vector)

		dictionary[len(dictionary)] = { "url": url, "text": text}

def chunk_text(text, max_words=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks

def add_vectorized_PDF(url, vectors, dictionary):
	response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
	response.raise_for_status()

	pdf = BytesIO(response.content)
	reader = PdfReader(pdf)

	for page_num, page in enumerate(reader.pages):
		text = page.extract_text()
		if not text:
			continue

		chunks = chunk_text(text)

		for chunk_num, chunk in enumerate(chunks):
			vector = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
			vectors.append(vector)
			dictionary[len(dictionary)] = {"url": url, "text": text}
	
def cosine_compare(query_vector, target_vector):
	similarity = (query_vector @ target_vector) / (np.linalg.norm(query_vector) * (np.linalg.norm(target_vector)))
	return similarity

def search_database(query_vector, vectors, dictionary, K):

	top_K_vectors = np.array()

	for vector in vectors:
		similarity_coefficient = cosine_compare(query_vector, vector)
		