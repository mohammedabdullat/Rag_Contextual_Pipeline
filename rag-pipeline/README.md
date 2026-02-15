# Contextual RAG Pipeline

A production-grade **Contextual Retrieval-Augmented Generation** pipeline implementing [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) technique with BM25, TF-IDF, and dense embedding baselines, served via a FastAPI REST API.

![System Architecture](docs/design_diagram.png)

---

## Features

- **Anthropic Contextual Retrieval** — LLM generates a situating context for each chunk before embedding, dramatically improving retrieval quality
- **Two chunking strategies** — Fixed-size token windows + semantic paragraph-boundary chunking
- **Three retrieval methods** — Contextual embeddings (ChromaDB), BM25, TF-IDF
- **Hybrid retrieval** — Reciprocal Rank Fusion across all three methods
- **Swap-friendly LLM config** — Change `OPENAI_BASE_URL` + model name to use any provider
- **FastAPI REST API** with auto-generated OpenAPI docs at `/docs`
- **Benchmarking** — Latency, cosine similarity, Recall@K

---

## Project Structure

```
rag-pipeline/
├── data/
│   ├── paper.pdf              # Place your PDF here
│   └── ground_truth.json      # 12 QA pairs with page + span annotations
├── src/
│   ├── config.py              # Central settings (swap LLMs here)
│   ├── ingestion/
│   │   ├── pdf_parser.py      # PyMuPDF + pdfplumber (text + tables)
│   │   └── chunker.py         # FixedSizeChunker + SemanticChunker
│   ├── contextual/
│   │   └── context_generator.py  # Anthropic-style context prepending
│   ├── indexing/
│   │   ├── embedding_index.py    # ChromaDB vector store
│   │   ├── bm25_index.py         # BM25 sparse index
│   │   └── tfidf_index.py        # TF-IDF index
│   ├── retrieval/
│   │   └── retriever.py          # Unified retriever + RRF fusion
│   ├── generation/
│   │   └── answer_generator.py   # OpenAI GPT answer generation
│   ├── api/
│   │   ├── main.py               # FastAPI application
│   │   └── models.py             # Pydantic request/response models
│   └── evaluation/
│       └── benchmarker.py        # Latency + cosine sim + Recall@K
├── scripts/
│   ├── build_index.py         # CLI: parse PDF + build all indices
│   └── run_benchmark.py       # CLI: run evaluation benchmark
├── tests/
│   └── test_pipeline.py       # Unit tests (pytest)
├── docs/
│   └── design_diagram.png     # System architecture diagram
├── .env                       # Configuration template
└── requirements.txt
```

---

## Installation

### Prerequisites
- Python 3.10+
- OpenAI API key (or any OpenAI-compatible API key)

### Steps

```bash
# 1. Clone or unzip the project
cd rag-pipeline

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 5. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your-key-here

# 6. Place your PDF
cp /path/to/your/paper.pdf data/paper.pdf
```

---

## Usage

### Step 1 — Build the index

```bash
python scripts/build_index.py
# Or force rebuild:
python scripts/build_index.py --overwrite
```

This will:
1. Parse the PDF (text + tables)
2. Chunk with both strategies (fixed + semantic)
3. Enrich each chunk with Anthropic-style context via LLM
4. Build ChromaDB, BM25, and TF-IDF indices

### Step 2 — Start the API server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3 — Query the API

```bash
# Health check
curl http://localhost:8000/health

# Query (hybrid method, top-5 results)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"q": "What is the main contribution of the paper?", "k": 5, "method": "hybrid"}'

# Use a specific method
curl -X POST http://localhost:8000/query \
  -d '{"q": "What datasets were used?", "k": 3, "method": "contextual"}'
```

**OpenAPI interactive docs:** http://localhost:8000/docs

---

## API Reference

### `POST /query`

**Request:**
```json
{
  "q": "What is the main contribution of this paper?",
  "k": 5,
  "method": "hybrid"
}
```

`method` options: `contextual` | `bm25` | `tfidf` | `hybrid`

**Response:**
```json
{
  "query": "What is the main contribution...",
  "answer": "The paper proposes...",
  "sources": {
    "contextual": [{"chunk_id": "abc123", "text": "...", "score": 0.91, "page_number": 2}],
    "bm25":       [{"chunk_id": "def456", "text": "...", "score": 12.3, "page_number": 3}],
    "tfidf":      [...],
    "hybrid":     [...]
  },
  "latency": {
    "contextual_ms": 120.5,
    "bm25_ms": 4.2,
    "tfidf_ms": 3.1,
    "hybrid_ms": 0.5,
    "total_ms": 890.3
  },
  "retrieval_method_used": "hybrid"
}
```

### `GET /health`

Returns pipeline status and number of indexed chunks.

### `POST /index/build?overwrite=false`

Rebuilds all indices from the configured PDF.

---

## Swapping LLM Providers

Edit `.env` — no code changes needed:

```bash
# Anthropic Claude (via OpenAI-compatible endpoint)
OPENAI_BASE_URL=https://api.anthropic.com/v1
OPENAI_CHAT_MODEL=claude-3-5-haiku-20241022

# Google Gemini
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_CHAT_MODEL=gemini-1.5-flash

# Local Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_CHAT_MODEL=llama3.2
OPENAI_API_KEY=ollama
```

---

## Benchmarking

```bash
python scripts/run_benchmark.py
# Output: benchmark_report.md
```

Evaluates all 12 ground truth QA pairs and reports:
- Average query latency (ms) per method
- Cosine similarity between generated and ground truth answer embeddings
- Recall@K (K=1,3,5) per method

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector store | ChromaDB | Fully OSS, persistent, no server needed |
| Embedding model | text-embedding-3-small | Best cost/quality ratio; swap-friendly |
| Chunking strategies | Fixed-size + Semantic | Covers both keyword and semantic queries |
| Context enrichment | OpenAI LLM | Implements Anthropic's technique; LLM-agnostic via config |
| Hybrid fusion | Reciprocal Rank Fusion | Proven, parameter-free fusion across heterogeneous scores |
| API framework | FastAPI | Auto OpenAPI docs, async, Pydantic validation |
| Persistence | Pickle (BM25/TF-IDF) + ChromaDB | Simple and fast for demo; swap to Redis/Postgres for prod |

---

## Contextual Retrieval — How It Works

Standard RAG embeds raw chunks. This often loses context — a chunk saying *"the accuracy improved by 4.2%"* is ambiguous without knowing which experiment it refers to.

**Anthropic's solution:** Before embedding, ask the LLM:
> *"Given the full document, write a 2-3 sentence context situating this chunk."*

The chunk is then stored as:
```
"This chunk discusses ablation results from Section 4, evaluating the 
proposed transformer architecture on the GLUE benchmark. 
The accuracy improved by 4.2%..."
```

This contextualizes every embedding, dramatically improving retrieval relevance.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | Your API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM provider endpoint |
| `LLM_CHAT_MODEL` | `gpt-4o-mini` | Chat model for generation + context |
| `LLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `PDF_PATH` | `data/paper.pdf` | Input PDF path |
| `CHUNK_SIZE` | `512` | Token window size |
| `CHUNK_OVERLAP` | `50` | Token overlap between fixed-size chunks |
| `TOP_K` | `5` | Default retrieval top-K |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | ChromaDB storage path |
