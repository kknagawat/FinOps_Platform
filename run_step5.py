"""
run_step5.py
============
Step 5 — RAG Pipeline Runner
-------------------------------
Builds the FAISS index from policy documents and runs a retrieval
quality evaluation across representative policy questions.

What this script does
---------------------
  1. Loads all 4 markdown documents from docs/
  2. Applies heading-aware chunking (400-char target, 50-char overlap)
  3. Generates sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim)
  4. Builds a FAISS IndexFlatIP and saves to rag_index/
  5. Runs 10 retrieval quality tests, reporting:
       - Top-1 source attribution
       - Hybrid score (semantic + BM25)
       - Whether the expected section was retrieved

Prerequisites
-------------
  pip install sentence-transformers faiss-cpu rank-bm25

Usage
-----
  python run_step5.py           # build index + run quality tests
  python run_step5.py --query "What is the enterprise refund window?"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")
for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

log = logging.getLogger(__name__)

# =============================================================================
# Retrieval quality test cases
# =============================================================================

EVAL_QUERIES = [
    {
        "query":            "What is the refund policy for enterprise customers?",
        "expected_source":  "refund_policy.md",
        "expected_section": "Enterprise Plans",
    },
    {
        "query":            "How many days do I have to request a refund on a monthly plan?",
        "expected_source":  "refund_policy.md",
        "expected_section": "Monthly Subscriptions",
    },
    {
        "query":            "What is the first response SLA for critical priority tickets?",
        "expected_source":  "sla_policy.md",
        "expected_section": "First Response SLA",
    },
    {
        "query":            "What uptime does the Pro plan guarantee?",
        "expected_source":  "sla_policy.md",
        "expected_section": "Uptime Commitments",
    },
    {
        "query":            "How are billing disputes handled and who resolves them?",
        "expected_source":  "escalation_procedures.md",
        "expected_section": "Billing Dispute Escalation",
    },
    {
        "query":            "What triggers automatic ticket escalation?",
        "expected_source":  "escalation_procedures.md",
        "expected_section": "Automatic Escalation Triggers",
    },
    {
        "query":            "What happens when a customer downgrades their subscription?",
        "expected_source":  "pricing_tiers.md",
        "expected_section": "Downgrades",
    },
    {
        "query":            "What add-ons are available for Pro plan and above?",
        "expected_source":  "pricing_tiers.md",
        "expected_section": "Add-Ons (Available on Pro and above)",
    },
    {
        "query":            "What retention offers are available for at-risk customers?",
        "expected_source":  "sla_policy.md",
        "expected_section": "Retention Offers for At-Risk Customers",
    },
    {
        "query":            "How are SLA credits calculated for service outages?",
        "expected_source":  "sla_policy.md",
        "expected_section": "SLA Credit Calculation",
    },
]


def run_step5(custom_query: str | None = None) -> None:
    BANNER = "█" * 72
    t_total = time.perf_counter()

    print()
    print(BANNER)
    print("  FINOPS ANALYTICS PLATFORM — STEP 5")
    print("  RAG-Powered Knowledge Agent")
    print(BANNER)
    print()

    # ── Section 1: Document ingestion + chunking ──────────────────────────────
    _section("PHASE 1 — Document Ingestion and Chunking")

    from src.rag.chunker import load_all_documents
    chunks = load_all_documents()

    print(f"  {'File':<35} {'Chunks':>6}")
    print("  " + "─" * 43)
    from collections import Counter
    per_file = Counter(c.source for c in chunks)
    for fname, count in sorted(per_file.items()):
        print(f"  {fname:<35} {count:>6}")
    print(f"  {'TOTAL':<35} {len(chunks):>6}\n")

    print(f"  Chunking strategy:")
    print(f"    Method      : Heading-aware split (##/### boundaries)")
    print(f"    Target size : 400 characters per chunk")
    print(f"    Overlap     : 50 characters between adjacent chunks")
    print(f"    Rationale   : Each heading = one policy rule = one chunk.")
    print(f"                  Prevents cutting mid-sentence in SLA tables.")
    print()

    # Show chunk size distribution
    sizes = [c.char_count for c in chunks]
    print(f"  Chunk size stats: min={min(sizes)}  "
          f"mean={sum(sizes)//len(sizes)}  max={max(sizes)} chars")

    # ── Section 2: Embedding + FAISS index ───────────────────────────────────
    _section("PHASE 2 — Embedding and Indexing (FAISS)")

    t0 = time.perf_counter()
    from src.rag.embedder import build_and_save, load_index

    print("  Generating embeddings (all-MiniLM-L6-v2, 384-dim)…", end="", flush=True)
    index, metadata = build_and_save()
    embed_ms = (time.perf_counter() - t0) * 1000
    print(f" done ({embed_ms:.0f} ms)")

    print(f"\n  Index type  : FAISS IndexFlatIP (exact cosine search)")
    print(f"  Vectors     : {index.ntotal}")
    print(f"  Dimensions  : 384")
    print(f"  Saved to    : rag_index/faiss.index + rag_index/metadata.json")
    print(f"\n  Metadata per chunk: source, doc_title, section, text, chunk_id")

    # ── Section 3: Retrieval quality evaluation ───────────────────────────────
    _section("PHASE 3 — Retrieval Quality Evaluation (Hybrid: FAISS + BM25)")

    from src.rag.retriever import get_retriever
    retriever = get_retriever()

    queries = [{"query": custom_query, "expected_source": None, "expected_section": None}] \
              if custom_query else EVAL_QUERIES

    print(f"  Hybrid weights: α=0.70 (semantic)  +  0.30 (BM25)\n")
    print(f"  {'Query':<52} {'Top-1 Source':<40} {'Score':>7}  Status")
    print("  " + "─" * 110)

    hits = 0
    for ev in queries:
        results = retriever.retrieve(ev["query"], top_k=3)
        if not results:
            print(f"  {'[no results]':<52}")
            continue

        top        = results[0]
        source_ok  = ev["expected_source"] is None or top["source"]   == ev["expected_source"]
        section_ok = ev["expected_section"] is None or \
                     ev["expected_section"].lower() in top["section"].lower()
        correct    = source_ok and section_ok
        if correct:
            hits += 1

        status = "✓" if correct else "~"
        q_disp = ev["query"][:50] + ("…" if len(ev["query"]) > 50 else "")
        src    = f"{top['source']} — {top['section']}"[:40]
        print(f"  {q_disp:<52} {src:<40} {top['hybrid_score']:>6.3f}  {status}")

        # Show top-2 and top-3 in indented detail
        for i, r in enumerate(results[1:3], 2):
            src2 = f"{r['source']} — {r['section']}"[:40]
            print(f"  {'':52} [{i}] {src2:<36} {r['hybrid_score']:>6.3f}")

    if not custom_query:
        print(f"\n  Retrieval accuracy: {hits}/{len(queries)} top-1 hits correct")

    # ── Section 4: Source attribution example ────────────────────────────────
    _section("PHASE 4 — Source Attribution Demo")

    demo_q = "What are the enterprise customer refund terms?"
    results = retriever.retrieve(demo_q, top_k=2)
    print(f"  Question: \"{demo_q}\"\n")
    for i, r in enumerate(results, 1):
        print(f"  Result {i}: {r['citation']}")
        print(f"  Score   : hybrid={r['hybrid_score']}  "
              f"semantic={r['semantic_score']}  bm25={r['bm25_score']}")
        print(f"  Text    : {r['text'][:200]}…")
        print()

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    SEP2 = "═" * 72
    print(SEP2)
    print("  STEP 5 COMPLETE")
    print(SEP2)
    print(f"  Chunks indexed   : {len(chunks)}")
    print(f"  FAISS vectors    : {index.ntotal}")
    print(f"  Hybrid search    : FAISS (semantic 70%) + BM25 (keyword 30%)")
    print(f"  Source attrib.   : source + section on every retrieved chunk")
    print(f"  Agent integration: knowledge_retrieval_tool updated to use this pipeline")
    if not custom_query:
        print(f"  Retrieval quality: {hits}/{len(queries)} top-1 correct")
    print(f"  Total time       : {elapsed:.2f}s")
    print()
    print(f"  Output files:")
    print(f"    ✓  rag_index/faiss.index")
    print(f"    ✓  rag_index/metadata.json")
    print(SEP2)


def _section(title: str) -> None:
    print()
    print("┌" + "─" * 70 + "┐")
    print(f"│  {title:<68}│")
    print("└" + "─" * 70 + "┘")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5 — RAG Pipeline")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a custom retrieval query instead of eval suite")
    args = parser.parse_args()
    run_step5(custom_query=args.query)