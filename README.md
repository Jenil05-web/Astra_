# ◈ ASTRA — Self-Optimizing RAG Engine

> Upload a PDF. Ask questions. Watch the pipeline tune itself.

**Live Demo → [ekx44y3eegkk8mxshkztcn.streamlit.app](https://ekx44y3eegkk8mxshkztcn.streamlit.app/)**

---

## What is ASTRA?

ASTRA (Autonomous Self-Tuning RAG Architecture) is an AI-powered document intelligence system that doesn't just answer questions — it **continuously optimizes itself** to answer them better.

Most RAG pipelines are static. You pick a chunk size, a retrieval strategy, and a prompt — and hope for the best. ASTRA takes a different approach: it uses a **LangGraph-powered feedback loop** to evaluate its own performance with Ragas metrics and automatically tune its parameters until it hits a quality threshold you define.

---

## Features

### ◈ Mode A — Autonomous Optimizer
- Builds a FAISS vector index from your PDF
- Evaluates retrieval quality using **Faithfulness**, **Context Recall**, and **Context Precision** (Ragas)
- Automatically tunes chunk size, top-k retrieval, and prompt variant across iterations
- Visualizes score progression with an interactive Altair chart
- Logs every decision in a live terminal with timestamps
- Displays optimal config once the target score is reached

### ▶ Mode B — Production Chat
- Chat with your document using the optimized pipeline
- Query rewriting — every question is expanded into 2 sub-queries for broader retrieval
- Live telemetry: latency, token usage, estimated cost, retrieved chunks
- Full source context viewer with FAISS chunk inspection
- Raw JSON payload export

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph |
| LLM | OpenAI GPT-3.5-Turbo |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | FAISS |
| Evaluation | Ragas (Faithfulness, Context Recall, Context Precision) |
| Tracing | LangSmith |
| Document Loading | PyPDF |

---

## Architecture

```
PDF Upload
    │
    ▼
FAISS Index Builder
(chunk_size, chunk_overlap, top_k)
    │
    ▼
┌─────────────────────────────────┐
│        LangGraph Loop           │
│                                 │
│  execute_node → evaluate_node   │
│       ↑              │          │
│       │         ┌────┘          │
│       │         ▼               │
│    tune_node ←─ score < target  │
│                                 │
│         END ←── score ≥ target  │
└─────────────────────────────────┘
    │
    ▼
Optimal Config → Production Chat
```

**Tuning Logic:**
- `context_recall < 0.7` → increase chunk size (capture more context)
- `context_precision < 0.7` → decrease top-k (retrieve fewer, better chunks)
- `faithfulness < 0.7` → rotate prompt variant (v1 → v2 → v3)

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/jenil05/astra_
cd astra_
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key   # optional, for tracing
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## Streamlit Cloud Deployment

This app is deployed on Streamlit Cloud. To deploy your own:

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Add `OPENAI_API_KEY` and `LANGCHAIN_API_KEY` in **Secrets** (Settings → Secrets)
4. Make sure `.python-version` file exists in repo root containing `3.11`

**Secrets format:**
```toml
OPENAI_API_KEY = "sk-..."
LANGCHAIN_API_KEY = "ls__..."
```

---

## Project Structure

```
astra_/
├── app.py              # Streamlit UI — both modes, all visualizations
├── core.py             # RAG engine, LangGraph pipeline, Ragas evaluation
├── requirements.txt    # Pinned dependencies
├── .python-version     # Pins Python 3.11 for Streamlit Cloud
└── .env                # API keys (local only, not committed)
```

---

## Ragas Metrics Explained

| Metric | What it measures | Low score means |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | LLM is hallucinating |
| **Context Recall** | Did retrieval capture the relevant information? | Chunks are too small |
| **Context Precision** | Are retrieved chunks actually relevant? | Too many irrelevant chunks retrieved |

---

## Built by

**Jenil** — [LinkedIn](https://www.linkedin.com/in/jenil05)

---

*ASTRA is a research/portfolio project demonstrating autonomous RAG optimization with LangGraph and Ragas.*
