"""

This script reads all downloaded HTML files from `confluence_pages/`,
converts the HTML to clean text, chunks it intelligently, embeds each chunk,
and stores the vectors into a local vector database (FAISS).


* Clean HTML to readable Markdown text (via BeautifulSoup).
* Sentence-based chunking (optional overlap for context).
* Embedding using OpenAI or HuggingFace models.
* Vector index built with FAISS for fast semantic search.

"""

import os
import glob
import hashlib
import logging
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


INPUT_DIR = "downloaded_wikis/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index/index.faiss"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def html_to_text(html: str) -> str:
    """Convert Confluence HTML to readable plain text."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    """Split long text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks: List[str], model) -> np.ndarray:
    return np.array(model.encode(chunks, show_progress_bar=True, convert_to_numpy=True))


def save_faiss_index(vectors: np.ndarray, metadata: List[str], index_path=INDEX_PATH):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)
    with open(index_path + ".meta", "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(meta + "\n")
    print(f"Saved {len(metadata)} vectors to {index_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    model = SentenceTransformer(EMBEDDING_MODEL)

    html_files = sorted(glob.glob(f"{INPUT_DIR}/*.html"))
    all_chunks = []
    all_sources = []

    for file_path in html_files:
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        text = html_to_text(html)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([file_path] * len(chunks))

    logging.info("Total chunks: %d", len(all_chunks))

    embeddings = embed_chunks(all_chunks, model)

    metadata_lines = [f"{path} :: {hashlib.sha1(chunk.encode()).hexdigest()}" for path, chunk in zip(all_sources, all_chunks)]
    save_faiss_index(embeddings, metadata_lines)


if __name__ == "__main__":
    main()
