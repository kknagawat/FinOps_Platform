"""
src/rag/retriever.py
====================
Step 5 — Retrieval with Hybrid Search and Source Attribution

Retrieval strategy: Hybrid = Semantic (FAISS) + Keyword (BM25)
---------------------------------------------------------------
Pure semantic search (FAISS) excels at paraphrased questions:
  "What happens if I want money back?" → finds "refund policy" chunks

Pure keyword search (BM25) excels at exact-term queries:
  "SLA 99.95% uptime" → finds exact number in SLA table

Hybrid combines both:
  final_score = α × cosine_score + (1 - α) × bm25_score
  where α = 0.7 (semantic-dominant, since our questions are natural language)

This is the "Bonus: Hybrid Search" deliverable from Step 5.

Source attribution
------------------
Every retrieved chunk carries:
  source  : filename (e.g. "refund_policy.md")
  section : nearest heading (e.g. "Enterprise Plans")
  citation: formatted string "[Source: refund_policy.md — Enterprise Plans]"

These are passed back through knowledge_retrieval_tool to the agent,
which is instructed to include them verbatim in every policy answer.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Hybrid weight: fraction from semantic search (vs. BM25)
ALPHA = 0.70


# =============================================================================
# BM25 scorer
# =============================================================================

def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def _bm25_scores(
    query_tokens: list[str],
    corpus:       list[str],
    k1: float = 1.5,
    b:  float = 0.75,
) -> np.ndarray:
    """
    Compute BM25 scores for every document in corpus against the query.

    BM25 (Best Match 25) is a bag-of-words ranking function that improves
    on raw TF-IDF by:
      - Normalising for document length (b parameter)
      - Saturating term frequency (k1 parameter)

    Parameters
    ----------
    k1 : controls TF saturation. 1.5 is standard (range 1.2–2.0)
    b  : controls length normalisation. 0.75 is standard (range 0–1)

    Returns ndarray of shape (len(corpus),) with scores ∈ [0, ∞).
    Scores are normalised to [0, 1] at the end for hybrid combination.
    """
    N   = len(corpus)
    tokenised_corpus = [_tokenise(doc) for doc in corpus]
    avg_dl = np.mean([len(d) for d in tokenised_corpus]) or 1.0

    scores = np.zeros(N, dtype=np.float32)

    for term in set(query_tokens):  # deduplicate query tokens
        # Document frequency: how many docs contain this term
        df = sum(1 for doc in tokenised_corpus if term in doc)
        if df == 0:
            continue
        # IDF component (Robertson-Spärck Jones)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for i, doc_tokens in enumerate(tokenised_corpus):
            tf  = doc_tokens.count(term)
            dl  = len(doc_tokens)
            # BM25 TF component with length normalisation
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores[i] += idf * tf_norm

    # Normalise to [0, 1]
    max_score = scores.max()
    if max_score > 0:
        scores /= max_score

    return scores


# =============================================================================
# Main retriever
# =============================================================================

class RAGRetriever:
    """
    Hybrid retriever: FAISS semantic search + BM25 keyword search.

    Usage
    -----
    retriever = RAGRetriever()          # loads index from disk
    results   = retriever.retrieve("What is the enterprise refund window?", top_k=3)

    Each result dict has:
        text            : the chunk text
        source          : filename
        section         : heading
        semantic_score  : cosine similarity from FAISS (0-1)
        bm25_score      : normalised BM25 score (0-1)
        hybrid_score    : α*semantic + (1-α)*bm25
        citation        : formatted source string
    """

    def __init__(self):
        from src.rag.embedder import load_index, build_and_save
        self.index, self.metadata = load_index()

        if self.index is None:
            log.info("No existing index found — building from docs/ …")
            self.index, self.metadata = build_and_save()

        # Pre-tokenise corpus for BM25 (done once, reused per query)
        self._corpus_texts = [m["text"] for m in self.metadata]
        log.info("RAGRetriever ready — %d chunks indexed", len(self.metadata))

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Steps
        -----
        1. Embed the query using the same model as the index.
        2. FAISS inner-product search → top-(top_k × 3) candidates.
        3. BM25 scores for all corpus chunks.
        4. Combine: final = α × cosine + (1-α) × bm25.
        5. Sort by final score, return top_k with metadata.
        6. Add source attribution (citation string) to each result.

        Parameters
        ----------
        query  : natural language question
        top_k  : number of chunks to return (default 3)

        Returns
        -------
        list[dict] sorted by hybrid_score descending
        """
        if not self.metadata:
            log.warning("No chunks in index — returning empty results")
            return []

        from src.rag.embedder import embed_texts

        # ── Semantic scores via FAISS ─────────────────────────────────────────
        q_vec    = embed_texts([query])              # (1, 384)
        n_search = min(len(self.metadata), top_k * 3)
        cosine_scores, indices = self.index.search(q_vec, n_search)

        # Map FAISS results back to full-corpus score array
        semantic_full = np.zeros(len(self.metadata), dtype=np.float32)
        for score, idx in zip(cosine_scores[0], indices[0]):
            if idx >= 0:
                # Cosine scores from IndexFlatIP ∈ [-1, 1]; clip to [0, 1]
                semantic_full[idx] = float(np.clip(score, 0, 1))

        # ── BM25 scores ───────────────────────────────────────────────────────
        query_tokens = _tokenise(query)
        bm25_full    = _bm25_scores(query_tokens, self._corpus_texts)

        # ── Hybrid combination ────────────────────────────────────────────────
        hybrid_scores = ALPHA * semantic_full + (1 - ALPHA) * bm25_full

        # ── Build result list ─────────────────────────────────────────────────
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results: list[dict] = []
        for idx in top_indices:
            if hybrid_scores[idx] <= 0:
                continue
            chunk = dict(self.metadata[idx])
            chunk["semantic_score"] = round(float(semantic_full[idx]), 4)
            chunk["bm25_score"]     = round(float(bm25_full[idx]),     4)
            chunk["hybrid_score"]   = round(float(hybrid_scores[idx]), 4)
            chunk["citation"]       = (
                f"[Source: {chunk['source']} — {chunk['section']}]"
            )
            results.append(chunk)

        return results

    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadata) > 0


# =============================================================================
# Singleton: one retriever instance shared across the process
# =============================================================================

_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """Return the process-level singleton retriever (builds index on first call)."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    retriever = get_retriever()

    test_queries = [
        "What is the refund policy for enterprise customers?",
        "What are the SLA response times for critical tickets?",
        "How does billing dispute escalation work?",
        "What is the uptime SLA for the Pro plan?",
        "What happens when a customer downgrades their plan?",
        "What are the retention offers for at-risk customers?",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("─" * 60)
        results = retriever.retrieve(q, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"  [{i}] hybrid={r['hybrid_score']:.3f}  "
                  f"sem={r['semantic_score']:.3f}  "
                  f"bm25={r['bm25_score']:.3f}")
            print(f"       {r['citation']}")
            print(f"       {r['text'][:120]}…")