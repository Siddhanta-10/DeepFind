"""

Runs a small Flask server that exposes:
1. **/api/ask**   – POST JSON {"question": "..."} → JSON answer.
2. **/**          – Simple HTML chat page for users.


Environment variables (optional):
* `MODEL_NAME` – override default TinyLlama model.
* `DEVICE`     – "cpu" (default) or e.g. "cuda:0" if you have GPU.
"""

import os
import logging
from pathlib import Path
from typing import List

from flask import Flask, request, jsonify, render_template_string
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import pipeline


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_PATH = Path("faiss_index/index.faiss")
META_PATH = FAISS_PATH.with_suffix(".faiss.meta")
MODEL_NAME = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = os.getenv("DEVICE", "cpu")
TOP_K = 5

# ---------------------------------------------------------------------------
# Load models + index once at startup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

log.info("Loading embedding model…")
embedder = SentenceTransformer(EMBEDDING_MODEL)

log.info("Loading FAISS index (%s)…", FAISS_PATH)
index = faiss.read_index(str(FAISS_PATH))
metadata = META_PATH.read_text(encoding="utf-8").splitlines()

log.info("Booting TinyLlama (%s) on %s…", MODEL_NAME, DEVICE)
lm = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE,
    trust_remote_code=True,
)


def search(question: str) -> List[str]:
    """Return top‑k metadata lines for *question*."""
    vec = np.array(embedder.encode([question]), dtype="float32")
    _, idxs = index.search(vec, TOP_K)
    return [metadata[i] for i in idxs[0] if i < len(metadata)]


def meta_to_text(meta_line: str) -> str:
    html_path = meta_line.split(" :: ")[0]
    if not Path(html_path).exists():
        return ""
    soup = BeautifulSoup(Path(html_path).read_text("utf-8"), "html.parser")
    return soup.get_text("\n", strip=True)


def answer(question: str) -> str:
    context_lines = [meta_to_text(m) for m in search(question)]
    context = "\n\n".join(context_lines)
    prompt = (
        "You are an internal documentation assistant. Answer only from the "
        "context. If unsure, say you don't know.\n\nContext:\n" + context +
        f"\n\nQuestion: {question}\nAnswer:"
    )
    out = lm(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    return out.split("Answer:")[-1].strip()


app = Flask(__name__)

CHAT_HTML = """
<!doctype html><title>Wiki QA Chat</title>
<style>body{font-family:sans-serif;max-width:720px;margin:40px auto}textarea{width:100%;height:80px}</style>
<h2>Internal Wiki Q&A</h2>
<form method=post>
  <textarea name=q placeholder="Ask a question…">{{q}}</textarea><br>
  <button>Ask</button>
</form>
{% if answer %}<h3>Answer</h3><pre>{{answer}}</pre>{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def chat_ui():
    q = answer_text = ""
    if request.method == "POST":
        q = request.form.get("q", "").strip()
        if q:
            answer_text = answer(q)
    return render_template_string(CHAT_HTML, q=q, answer=answer_text)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    if not request.is_json:
        return jsonify({"error": "JSON required"}), 400
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question missing"}), 400
    return jsonify({"answer": answer(question)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)