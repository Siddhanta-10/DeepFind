"""
This script loads the FAISS index (built in **Step 2**) and answers natural‑
language questions about your Confluence wiki entirely **offline**:

1. Embeds the user query with the *same* MiniLM model used for documents.
2. Retrieves the top‑k most relevant wiki chunks from the FAISS index.
3. Generates an answer using the lightweight TinyLlama‑1.1B‑Chat model

If you prefer an even smaller or quantized model, swap `MODEL_NAME` below to any
CPU‑friendly GGUF/ggml or `phi-2` checkpoint and set `device="cpu"`.
"""


import os
import logging
from pathlib import Path
from typing import List, Tuple

import faiss                   # Vector DB
import numpy as np
from bs4 import BeautifulSoup  # HTML → text
from sentence_transformers import SentenceTransformer  # Embeddings
from transformers import pipeline


#     import openai
#     def answer_with_gpt35(query, contexts):
#         context_block = "\n\n".join(contexts)
#         prompt = f"""
#         Use only the following context to answer the question:
#         {context_block}
#
#         Question: {query}
#         Answer:
#         """
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=512,
#         )
#         return response.choices[0].message.content.strip()
#
# Then call this instead of `answer_with_llm()` in the main loop.


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = Path("faiss_index/index.faiss")
META_PATH = INDEX_PATH.with_suffix(".faiss.meta")
TOP_K = 5
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Helpers – load FAISS + metadata
# ---------------------------------------------------------------------------

def load_index_and_meta() -> Tuple[faiss.IndexFlatL2, List[str]]:
    index = faiss.read_index(str(INDEX_PATH))
    with META_PATH.open("r", encoding="utf-8") as f:
        metadata = [line.rstrip("\n") for line in f]
    return index, metadata

# ---------------------------------------------------------------------------
# Semantic search over FAISS
# ---------------------------------------------------------------------------

def search(query: str, embed_model, index, metadata, k: int = TOP_K):
    query_vec = np.array(embed_model.encode([query]), dtype="float32")
    D, I = index.search(query_vec, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# ---------------------------------------------------------------------------
# Get clean text back from saved HTML
# ---------------------------------------------------------------------------

def meta_to_text(meta_line: str) -> str:
    """Extract plain text from the HTML file referenced in *meta_line*."""
    html_path = meta_line.split(" :: ")[0]
    if not Path(html_path).exists():
        return ""
    with open(html_path, "r", encoding="utf-8") as fp:
        html = fp.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n", strip=True)

# ---------------------------------------------------------------------------
# Answer generation via TinyLlama (HF pipeline)
# ---------------------------------------------------------------------------

def answer_with_llm(query: str, contexts: List[str], llm_pipeline) -> str:
    """Compose prompt + call TinyLlama pipeline."""
    context_block = "\n\n".join(contexts)
    prompt = (
        "An internal documentation assistant. Answer only from the "
        "context below. If the answer is not present, says don't know.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )
    generated = llm_pipeline(prompt, max_new_tokens=256, do_sample=False)[0][
        "generated_text"
    ]
    # Return only text after "Answer:" to keep response clean
    return generated.split("Answer:")[-1].strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Loading embedding model → %s", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    logging.info("Loading FAISS index (%s) + metadata", INDEX_PATH)
    index, metadata = load_index_and_meta()

    logging.info("Booting local LLM (%s)", MODEL_NAME)
    llm = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=DEVICE,
        trust_remote_code=True,
    )

    print("\nAsk questions about your Confluence wiki. Type 'exit' to quit.")
    while True:
        query = input("\n?  » ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        hits = search(query, embed_model, index, metadata)
        contexts = [meta_to_text(m) for m in hits]
        if not any(contexts):
            print("No relevant wiki chunks found.")
            continue

        answer = answer_with_llm(query, contexts, llm)
        print("\n  Answer:\n", answer)


if __name__ == "__main__":
    main()
