from bs4 import BeautifulSoup
import requests

url = "https://en.wikipedia.org/wiki/July_Crisis"

response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

paragarphs = soup.find_all("p")

for p in paragarphs:
    text = p.get_text(strip=True)
    if text:
        print(text)
