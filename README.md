# Offline RAG System (Ollama + FAISS)

A **fully offline Retrieval-Augmented Generation (RAG) system** built in Python.
This project allows you to ingest documents, generate and persist embeddings, and query them locally using an LLM — **without any internet connection once models are installed**.

The system is optimized for performance by **persisting document embeddings** and **caching queries and responses in memory**.

---

## ✨ Features

- 🔒 **100% Offline RAG** (after model setup)
- 📄 Document ingestion and chunking
- 🧠 Embedding generation using **nomic-embed-text**
- 🗄️ Vector search powered by **FAISS**
- 💾 Persistent embeddings stored as `.npy` files (no re-embedding on restart)
- ⚡ In-memory caching for:

  - Query embeddings
  - Model responses

- 🖥️ Simple **command-line interface (CLI)**
- 🧩 Modular and easy to extend

---

## 🧱 Tech Stack

- **LLM Runtime:** Ollama
- **LLM:** llama3.2
- **Embedding Model:** nomic-embed-text
- **Vector Database:** FAISS
- **Language:** Python (3.8+)

---

## 🏗️ Architecture Overview

1. Documents are ingested from disk
2. Documents are chunked into smaller text segments
3. Chunks are embedded using `nomic-embed-text`
4. Embeddings are:

   - Indexed in FAISS
   - Persisted to disk as `.npy` files

5. User queries (via CLI) are:

   - Embedded
   - Cached in memory

6. Relevant chunks are retrieved via FAISS similarity search
7. Retrieved context is passed to `llama3.2`
8. Final responses are:

   - Returned to the user
   - Cached in memory for faster repeat queries

> ⚠️ Note: Only **document embeddings** are persisted. Query and response caches are **in-memory only** for now.

---

## 🚀 Getting Started

### 1️⃣ Install Ollama

Download and install Ollama from:

👉 [https://ollama.com](https://ollama.com)

---

### 2️⃣ Install Required Models

Once Ollama is installed, pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

These models are stored locally and used fully offline.

---

### 3️⃣ Clone the Repository

```bash
git clone https://github.com/miracletim/faiss-rag-offline.git
cd faiss-rag-offline
```

---

### 4️⃣ Install Python Dependencies

Ensure you have **Python 3.8 or higher**, then run:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the System

Simply run the app entry point:

```bash
python app.py
```

The system is **self-guided** and will:

- Inform you if required folders, files, or models are missing
- Guide you through the setup if something is not configured correctly

---

## 📂 Embedding Persistence

- Document embeddings are saved as `.npy` files
- On subsequent runs, embeddings are **loaded from disk** instead of recomputed
- This significantly improves startup and query performance

---

## 🧠 Caching Strategy

| Cached Item         | Storage       | Persisted |
| ------------------- | ------------- | --------- |
| Document embeddings | Disk (`.npy`) | ✅ Yes    |
| Query embeddings    | Memory        | ❌ No     |
| LLM responses       | Memory        | ❌ No     |

---

## 📌 Requirements

- Python **3.8+**
- Ollama (installed locally)
- llama3.2 model
- nomic-embed-text model

All Python dependencies are listed in `requirements.txt`.

---

## 🔮 Future Improvements

- Persist query & response cache
- Support for multiple embedding files
- Configurable chunk sizes
- Streaming responses
- Optional UI (web or desktop)

---

## 🤝 Contributing

Contributions, ideas, and improvements are welcome. Feel free to fork the repo and submit a pull request.

---

## 📜 License

MIT

---

## 🧠 Author

**Miracle Timothy**
Full Stack Developer | AI Systems Builder

---

> _"Offline-first AI systems are not a limitation — they are a design choice."_ 🚀
