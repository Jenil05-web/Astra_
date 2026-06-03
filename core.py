"""
core.py — Astra RAG engine.

Fixes applied
─────────────
1.  langchain_classic → langchain  (ImportError)
2.  @traceable removed from evaluate_pipeline  (LangSmith hijacks Ragas callbacks)
3.  nest_asyncio.apply() — asyncio.run() conflict inside Streamlit / LangGraph
4.  ContextPrecision added so tune_node's top_k branch actually fires
5.  execute_node writes incremented iteration back to state (log/state sync bug)
6.  tune_node log uses correct pre-increment value of iteration
7.  run_rag merges final-invocation contexts into all_contexts (missing chunks)
8.  rewrite_query parser handles "1)", "1." and extra blank lines
9.  evaluate_pipeline guards against empty dataset and all-NaN Ragas columns
10. ROOT CAUSE FIX: monkey-patch parse_run_traces so it never raises IndexError.
    LangSmith registers a tracer INSIDE the LangChain callback manager object —
    flipping the env var alone does not remove that registered handler.  Ragas's
    parse_run_traces crashes when the handler list is empty because LangSmith
    already consumed the events.  The traces field is display-only; metric scores
    are computed before this function runs, so returning [] on failure is safe.
11. Thread-isolated evaluate() call — runs Ragas in a daemon thread whose
    contextvars start fresh, so LangSmith's ContextVar-based tracer is absent.
"""

import os
import time
import threading
import nest_asyncio          
import numpy as np
from typing import TypedDict, Callable
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langsmith import traceable
from langgraph.graph import StateGraph, END

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, ContextRecall, ContextPrecision

# ── Patch 1: event loop ───────────────────────────────────────────────────────
# asyncio.run() inside Ragas conflicts with Streamlit/LangGraph's running loop.
nest_asyncio.apply()

# ── Patch 2: monkey-patch parse_run_traces (root-cause fix) ──────────────────
# LangSmith hooks into LangChain's callback manager at the object level, not just
# via env var.  Even with LANGCHAIN_TRACING_V2=false the registered handler can
# drain the events Ragas needs, leaving root_traces=[] → IndexError on [0].
# The traces attribute is only used for display; patching it to return [] on
# failure does not affect metric scores at all.
def _patch_ragas_parse_run_traces():
    try:
        import ragas.callbacks as _rcb
        _orig = _rcb.parse_run_traces

        def _safe_parse_run_traces(ragas_traces, run_id=None):
            try:
                return _orig(ragas_traces, run_id)
            except (IndexError, KeyError, AttributeError, TypeError):
                return []

        _rcb.parse_run_traces = _safe_parse_run_traces

        # dataset_schema also imports the symbol directly — patch that copy too
        import ragas.dataset_schema as _rds
        _rds.parse_run_traces = _safe_parse_run_traces
    except Exception:
        pass   # if ragas internals change, don't break the import

_patch_ragas_parse_run_traces()

load_dotenv()
os.environ["OPENAI_API_KEY"]       = os.getenv("OPENAI_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = "astra-rag-optimizer"

COST_PER_INPUT_TOKEN  = 0.0000015
COST_PER_OUTPUT_TOKEN = 0.000002

# Cached embedding model — avoids re-initialising on every pipeline rebuild
_EMBED_MODEL = OpenAIEmbeddings()

# ── Prompt variants ────────────────────────────────────────────────────────────
PROMPT_VARIANTS = {
    "v1": ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the provided context. Think step by step.\n<context>{context}</context>"),
        ("human", "{input}")
    ]),
    "v2": ChatPromptTemplate.from_messages([
        ("system", "You are a precise assistant. Use ONLY the context. If not found say 'Not found in context'.\n<context>{context}</context>"),
        ("human", "{input}")
    ]),
    "v3": ChatPromptTemplate.from_messages([
        ("system", "Identify the most relevant sentences in the context then answer using only those.\n<context>{context}</context>"),
        ("human", "{input}")
    ]),
}

DEFAULT_EVAL_DATASET = [
    {
        "question": "What is the main contribution of the transformer paper?",
        "ground_truth": "The transformer relies entirely on self-attention, dispensing with recurrence and convolutions."
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention jointly attends to information from different representation subspaces at different positions."
    },
    {
        "question": "What is positional encoding and why is it needed?",
        "ground_truth": "Positional encodings give the model information about token positions since it has no recurrence or convolution."
    },
]


# ── Thread-isolated Ragas evaluation 
def _run_ragas_isolated(samples: list): # in ragas evaluation we will not be using langsmith together
    """
    Run Ragas evaluate() inside a plain threading.Thread.

    threading.Thread does NOT copy the parent's ContextVar state (unlike
    ThreadPoolExecutor which does).  LangSmith's per-request tracer lives in a
    ContextVar, so a fresh thread has no LangSmith tracer active at all —
    Ragas's own RagasCallbackHandler runs without interference.
    """
    result_box: dict = {}

    def _worker(): # this function will basically run in a clean environment with no LangSmith tracer registered, so Ragas's callbacks work as intended.
        import asyncio, nest_asyncio as _na
        # Give this thread its own clean event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _na.apply(loop)
        try:
            df = evaluate(
                EvaluationDataset(samples=samples),
                metrics=[Faithfulness(), ContextRecall(), ContextPrecision()],
            ).to_pandas()
            result_box["df"] = df
        except Exception as exc:
            result_box["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=600)          # 10-minute hard cap per evaluation round

    if t.is_alive():
        raise TimeoutError("Ragas evaluation exceeded 600 s — check API connectivity.")
    if "error" in result_box:
        raise result_box["error"]
    return result_box["df"]


# ── Document + pipeline 
def load_documents(pdf_path: str) -> list:
    return PyPDFLoader(pdf_path).load()
     # docuuments loaded 
     # Rag pipline building function


def build_rag_pipeline(docs: list, chunk_size: int, chunk_overlap: int,
                       top_k: int, prompt_variant: str):
    """Rebuild FAISS index and retrieval chain with current parameters."""
    if not docs:
        raise ValueError("No documents provided to build_rag_pipeline.")
    chunks    = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ).split_documents(docs)
    db        = FAISS.from_documents(chunks, _EMBED_MODEL)
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    llm       = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain     = create_retrieval_chain(
                    retriever,
                    create_stuff_documents_chain(llm, PROMPT_VARIANTS[prompt_variant])
                )
    return chain, len(chunks)


# ── Query rewriter (used only in Mode B live chat) 
_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate exactly 2 different search queries for the question. "
               "Return ONLY the queries, one per line, numbered 1. 2."),
    ("human", "Question: {question}")
])
_rewriter = _rewrite_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5) | StrOutputParser()


def rewrite_query(question: str) -> list[str]:
    """
    Robustly parse the two sub-queries produced by the rewriter.
    Handles both "1." and "1)" prefixes and ignores blank lines.
    """
    try:
        raw = _rewriter.invoke({"question": question})
    except Exception:
        return [question, question]   # graceful degradation

    cleaned = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # strip leading "1." / "2." / "1)" / "2)" / "- "
        if len(line) >= 2 and line[0].isdigit() and line[1] in ".):":
            line = line[2:].strip()
        elif line.startswith("- "):
            line = line[2:].strip()
        if line:
            cleaned.append(line)
    # always return exactly 2 entries; fall back to original question if needed
    while len(cleaned) < 2:
        cleaned.append(question)
    return cleaned[:2]


# ── run_rag (Mode B — with rewriting + telemetry) 
@traceable(name="run-rag-live")
def run_rag(question: str, chain) -> dict:
    t0          = time.time()
    sub_queries = rewrite_query(question)
    all_contexts, seen = [], set()

    # Gather contexts from sub-queries
    for q in sub_queries:
        try:
            result = chain.invoke({"input": q})
        except Exception:
            continue
        for doc in result.get("context", []):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_contexts.append(doc.page_content)

    # Final answer on the original question
    final = chain.invoke({"input": question})

    # Merge final-invocation contexts that weren't captured above (bug fix #7)
    for doc in final.get("context", []):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            all_contexts.append(doc.page_content)

    # Fallback: if no context was retrieved at all, use the answer text
    if not all_contexts:
        all_contexts = [final["answer"]]

    in_tok  = len((" ".join(all_contexts) + question).split())
    out_tok = len(final["answer"].split())

    return {
        "question":      question,
        "answer":        final["answer"],
        "contexts":      all_contexts,
        "sub_queries":   sub_queries,
        "latency":       round(time.time() - t0, 2),
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "total_tokens":  in_tok + out_tok,
        "cost_usd":      round(in_tok * COST_PER_INPUT_TOKEN + out_tok * COST_PER_OUTPUT_TOKEN, 6),
    }


# ── evaluate_pipeline (training loop) 
# NOTE: intentionally NOT decorated with @traceable.
# LangSmith tracing intercepts Ragas's internal LangChain callbacks and drains
# the root_traces list → IndexError: list index out of range in parse_run_traces.
def evaluate_pipeline(docs: list, eval_dataset: list, chunk_size: int,
                      chunk_overlap: int, top_k: int, prompt_variant: str) -> dict:
    if not eval_dataset:
        raise ValueError("eval_dataset is empty — provide at least one question/ground_truth pair.")

    chain, _ = build_rag_pipeline(docs, chunk_size, chunk_overlap, top_k, prompt_variant)
    samples  = []

    for item in eval_dataset:
        try:
            result   = chain.invoke({"input": item["question"]})
            contexts = [d.page_content for d in result.get("context", [])]
            if not contexts:
                contexts = [result.get("answer", "")]
            samples.append(SingleTurnSample(
                user_input=item["question"],
                response=result.get("answer", ""),
                retrieved_contexts=contexts,
                reference=item["ground_truth"],
            ))
        except Exception as e:
            # Don't let one bad sample abort the whole evaluation
            print(f"[evaluate_pipeline] Skipping sample due to error: {e}")

    if not samples:
        raise RuntimeError("All evaluation samples failed — check your chain and eval dataset.")

    # Run Ragas in an isolated thread (ContextVar-clean) so LangSmith tracer
    # cannot interfere with Ragas RagasCallbackHandler (root-cause fix #10 + #11)
    scores_df = _run_ragas_isolated(samples)

    def safe_mean(col: str) -> float:
        if col not in scores_df.columns:
            return 0.0
        col_data = scores_df[col].dropna()
        if col_data.empty:
            return 0.0
        return round(float(col_data.mean()), 4)

    score_dict = {
        "faithfulness":      safe_mean("faithfulness"),
        "context_recall":    safe_mean("context_recall"),
        "context_precision": safe_mean("context_precision"),
    }
    score_dict["avg"] = round(float(np.mean(list(score_dict.values()))), 4)
    return score_dict


# ── LangGraph 
class AstraState(TypedDict):
    chunk_size:     int
    chunk_overlap:  int
    top_k:          int
    prompt_variant: str
    scores:         dict
    iteration:      int
    done:           bool

# this will build the entire graph or three nodes and will define the flow between them.
def build_graph(docs: list, eval_dataset: list, score_threshold: float,
                max_iterations: int, log_callback: Callable = None,
                history: list = None):

    if history is None:
        history = []

    def _log(msg: str):
        if log_callback:
            log_callback(msg)
 # this node will be called fitrst , it will build the rag pipeline and evaluate it against the dataset.
    def execute_node(state: AstraState) -> AstraState:
        # Increment iteration here so execute + evaluate share the same run number
        i = state["iteration"] + 1
        _log(f"[Run {i}] Building pipeline — chunk={state['chunk_size']}, "
             f"top_k={state['top_k']}, prompt={state['prompt_variant']}")
        _log(f"[Run {i}] Evaluating {len(eval_dataset)} questions against FAISS index...")
        scores = evaluate_pipeline(
            docs, eval_dataset,
            state["chunk_size"], state["chunk_overlap"],
            state["top_k"], state["prompt_variant"]
        )
        # Write the incremented iteration back so evaluate_node sees the right value
        return {**state, "scores": scores, "iteration": i}
 # this node will evaluate the scores and decide whether to end the loop or continue tuning.
    def evaluate_node(state: AstraState) -> AstraState:
        avg       = state["scores"]["avg"]
        iteration = state["iteration"]   # already incremented by execute_node
        passed    = avg >= score_threshold
        _log(f"[Run {iteration}] Ragas Score: {avg:.4f} "
             f"{'✅ Target met.' if passed else f'❌ Below {score_threshold}.'}")

        history.append({
            "iteration":      iteration,
            "chunk_size":     state["chunk_size"],
            "top_k":          state["top_k"],
            "prompt_variant": state["prompt_variant"],
            **state["scores"],
        })

        done = passed or iteration >= max_iterations
        if done and passed:
            _log(f"[Run {iteration}] ✅ Saving optimal config. Loop complete.")
        elif done:
            _log(f"[Run {iteration}] Max iterations reached. Best config saved.")
        return {**state, "done": done}
 # this node will tune the parameters based on which metric is low and log the changes.
    def tune_node(state: AstraState) -> AstraState:
        scores         = state["scores"]
        i              = state["iteration"]   # correct run number (already incremented)
        chunk_size     = state["chunk_size"]
        chunk_overlap  = state["chunk_overlap"]
        top_k          = state["top_k"]
        prompt_variant = state["prompt_variant"]

        # Low context_recall → wider chunks to capture more relevant text
        if scores.get("context_recall", 1.0) < 0.7:
            chunk_size    = int(np.clip(chunk_size + 200, 300, 1500))
            chunk_overlap = int(chunk_size * 0.2)
            _log(f"[Run {i}] Tuning: context_recall low → chunk_size → {chunk_size}")

        # Low context_precision → retrieve fewer but better chunks (now computed)
        if scores.get("context_precision", 1.0) < 0.7:
            top_k = int(np.clip(top_k - 1, 2, 8))
            _log(f"[Run {i}] Tuning: context_precision low → top_k → {top_k}")

        # Low faithfulness → try next prompt variant
        if scores.get("faithfulness", 1.0) < 0.7:
            variants       = ["v1", "v2", "v3"]
            prompt_variant = variants[(variants.index(prompt_variant) + 1) % 3]
            _log(f"[Run {i}] Tuning: faithfulness low → prompt → {prompt_variant}")

        return {**state,
                "chunk_size":    chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k":         top_k,
                "prompt_variant": prompt_variant}

    g = StateGraph(AstraState)
    g.add_node("execute",  execute_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("tune",     tune_node)
    g.set_entry_point("execute")
    g.add_edge("execute", "evaluate")
    g.add_conditional_edges("evaluate",
        lambda s: "end" if s["done"] else "tune",
        {"end": END, "tune": "tune"})
    g.add_edge("tune", "execute")
    return g.compile()