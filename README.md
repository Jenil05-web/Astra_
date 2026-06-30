# ◈ ASTRA: Autonomous Self-Tuning RAG Architecture
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
**ASTRA** is a document intelligence system that doesn't just answer questions—it **optimizes its own brain**. While most RAG (Retrieval-Augmented Generation) pipelines are static and prone to "hallucination debt," ASTRA uses a LangGraph-powered feedback loop to autonomously tune its retrieval parameters until it hits your quality standards...
---
## 🚀 The Core Innovation: "The Self-Correction Loop"
Most developers "guess" their chunk size and Top-K. ASTRA treats these as **hyperparameters** to be optimized. 
> **The Logic:** ASTRA runs an initial retrieval, calculates Ragas metrics, and if the scores are below the target threshold, the `tune_node` modifies the FAISS indexing strategy in real-time.

| Metric | Trigger | Response |
| :--- | :--- | :--- |
| **Context Recall** | Score < 0.7 | **Increase Chunk Size** (Capture more surrounding context) |
| **Context Precision** | Score < 0.7 | **Decrease Top-K** (Filter out noise/irrelevant chunks) |
| **Faithfulness** | Score < 0.7 | **Prompt Rotation** (Cycle through prompt templates to reduce hallucination) |

---
## ✨ Features
### 🤖 Mode A: Autonomous Optimizer
*   **Dynamic FAISS Indexing:** Builds and rebuilds indices on the fly based on evaluation data.
*   **Live Score Progression:** Watch real-time Altair charts as the system "learns" your document.
*   **Decision Logging:** A transparent terminal view of every parameter shift and why it happened.
### ⚡ Mode B: Production Chat
*   **Multi-Query Expansion:** Rewrites every user question into 2 sub-queries to maximize retrieval breadth.
*   **Live Telemetry:** Tracks latency, token counts, and estimated API costs per message.
*   **Context Inspector:** Peek under the hood at the raw FAISS chunks used for the answer.
---
## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Orchestration** | [LangGraph](https://www.langchain.com/langgraph) |
| **Intelligence** | OpenAI GPT-3.5-Turbo |
| **Embeddings** | `text-embedding-ada-002` |
| **Vector Store** | FAISS |
| **Evaluation** | [Ragas](https://ragas.io/) |
| **Interface** | Streamlit |
| **Observability** | LangSmith |

---
## 🏗 Architecture
```mermaid
graph TD
    A[PDF Upload] --> B[FAISS Indexer]
    B --> C{LangGraph Loop}
    C --> D[Execute RAG]
    D --> E[Evaluate with Ragas]
    E --> F{Score >= Target?}
    F -- No --> G[Tune Parameters]
    G --> B
    F -- Yes --> H[Finalize Optimal Config]
    H --> I[Production Chat Mode]
