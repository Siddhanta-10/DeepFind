# 🧠 DeepFind – Internal Wiki Q&A Assistant

DeepFind is a fully private, offline question-answering (QA) assistant powered by a lightweight open-source LLM (TinyLlama) to answer natural language questions based on your internal **Confluence** or **Wiki** documentation.

It is designed to:

* 🛡️ Run **entirely offline** (no cloud dependency)
* 📚 Automatically crawl and index large internal wiki hierarchies
* 💬 Provide a simple **web chat UI** and a **REST API** for integrations
* 🧠 Use fast semantic search and local LLM to generate answers

---

## 📆 Project Structure

```
.
├── ConfluenceCrawler.py               #1: Wiki crawler using REST API
├── ConfluenceEmbeddingPipeline.py     #2 output: vector index & metadata
├── ConfluenceQAPipeline.py            #3: CLI-based semantic search + TinyLlama answer
├── DeepFind.py                        #4: Web UI + REST API for local answering
├── requirements.txt                   # All required Python packages
├── README.md
```

---

## ⚙️ Installation

```bash
# Clone this repo
$ git clone https://github.com/Siddhanta-10/DeepFind
$ cd deepfind

# Install dependencies
$ pip install -r requirements.txt

# Optional (to suppress symlink warnings on Windows)
$ set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### 🧹 Requirements

* Python 3.8+
* RAM: 4–8 GB minimum (TinyLlama runs comfortably on CPU)
* Internet (only for first-time TinyLlama model download)

---

## 🚀 Usage

### Step 1: Crawl Confluence or Wiki Pages (not in this repo)

Use your REST API–based crawler to download pages into a folder. Each page should be saved as an HTML file and referenced in a metadata file.
```bash
$ python ConfluenceCrawler.py
```

### Step 2: Build FAISS index (embedding)

```bash
$ python ConfluenceEmbeddingPipeline.py
```

* Splits documents into chunks
* Embeds them using MiniLM
* Saves FAISS index to `faiss_index/index.faiss`

> **NOTE**: These steps are already done if you have the `faiss_index/` folder.

### Step 3: Run QA in CLI (offline)

```bash
$ python ConfluenceQAPipeline.py
```

This will let you ask questions and get LLM-based answers in the terminal.

### Optional: Start Web UI or REST API

```bash
$ python DeepFind.py        # dev mode
# OR
$ waitress-serve --port=8000 DeepFind:app  # production
```

Visit [http://localhost:5000](http://localhost:5000) to chat.

### Optional: Use GPT-3.5 instead of TinyLlama

```bash
$ pip install openai
$ export OPENAI_API_KEY=your-key-here
```

Then modify `answer()` function in the code to use OpenAI API.

---

## 🔐 Privacy First

* This system **does not send any data to external services**.
* You can run this fully **inside your company firewall**.
