"""
src/rag/embedder.py
===================
Step 5 — Embedding and FAISS Indexing

Embedding model choice: all-MiniLM-L6-v2
-----------------------------------------
  • Open-source (no API key required)
  • 384-dimensional dense vectors
  • 22M parameters — fast inference on CPU (~5ms per chunk on MacBook)
  • Strong semantic similarity performance on sentence-level retrieval
  • Normalised embeddings → cosine similarity = dot product (enables IndexFlatIP)

FAISS index type: IndexFlatIP  (Inner Product on L2-normalised vectors)
-----------------------------------------------------------------------
  With normalised vectors, dot product == cosine similarity.
  IndexFlatIP is an exact search index — no approximation, no information loss.
  For 28 chunks this is optimal; approximate indexes (HNSW, IVF) are only
  beneficial above ~10,000 vectors where exact search becomes slow.

Persistence
-----------
  The index and metadata are saved to rag_index/ so we build once and
  reuse on every agent call. Rebuilding is triggered only when documents change.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import faiss
import numpy as np

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parents[2] / "rag_index"
INDEX_DIR.mkdir(exist_ok=True)

INDEX_PATH    = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
MODEL_NAME    = "BAAI/bge-small-en-v1.5"

# Module-level cache so the model loads once per Python process
_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
        log.info("Embedding model loaded: fastembed BAAI/bge-small-en-v1.5  (dim=384)")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised float32 vectors (384-dim).
    fastembed uses ONNX Runtime — no PyTorch required, works on Python 3.13.
    """
    model = _get_model()
    vecs  = np.array(list(model.embed(texts)), dtype=np.float32)
    # L2-normalise so dot product == cosine similarity in FAISS IndexFlatIP
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def build_index(chunks: list) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Build a FAISS IndexFlatIP from a list of Chunk objects.

    Parameters
    ----------
    chunks : list[Chunk]  — output of chunker.load_all_documents()

    Returns
    -------
    (index, metadata_list)
        index         : FAISS index, one vector per chunk
        metadata_list : list of dicts matching vector positions in the index
    """
    log.info("Building FAISS index from %d chunks …", len(chunks))

    texts    = [c.text for c in chunks]
    vectors  = embed_texts(texts)           # (28, 384)
    dim      = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)          # Inner Product = cosine on normalised vecs
    index.add(vectors)

    metadata = [c.to_dict() for c in chunks]

    log.info("Index built: %d vectors, dim=%d", index.ntotal, dim)
    return index, metadata


def save_index(index: faiss.IndexFlatIP, metadata: list[dict]) -> None:
    """Persist FAISS index and metadata to rag_index/."""
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    log.info("Index saved → %s  (%d vectors)", INDEX_PATH, index.ntotal)


def load_index() -> tuple[faiss.IndexFlatIP | None, list[dict]]:
    """
    Load FAISS index and metadata from disk.

    Returns (None, []) if the index hasn't been built yet.
    """
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, []
    index    = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text())
    log.info("Index loaded: %d vectors from %s", index.ntotal, INDEX_PATH)
    return index, metadata


def build_and_save() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Full pipeline: load docs → chunk → embed → build FAISS index → save.
    Called once during setup; subsequent calls just use load_index().
    """
    from src.rag.chunker import load_all_documents
    chunks   = load_all_documents()
    index, metadata = build_index(chunks)
    save_index(index, metadata)
    return index, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s │ %(message)s")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    index, metadata = build_and_save()
    print(f"\nIndex built: {index.ntotal} vectors")
    print(f"Metadata entries: {len(metadata)}")
    print(f"\nFirst 5 chunks indexed:")
    for m in metadata[:5]:
        print(f"  [{m['source']} — {m['section']}]  {len(m['text'])} chars")