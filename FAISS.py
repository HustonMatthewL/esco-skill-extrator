# vector_db.py
from __future__ import annotations
import os
import json
import uuid
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, Union, Callable, Literal

import numpy as np

try:
    import faiss  # optional; used if available for fast search
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


# ---------- Embedding interface & helpers ----------

class EmbeddingFn(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an array of shape (N, D) for N input strings."""


class OllamaEmbedder:
    """
    Minimal client for a local Ollama instance (http://localhost:11434 by default).
    Requires a model that supports embeddings via the /api/embed endpoint (Ollama ≥ 0.1.26).
    Falls back to the legacy /api/embeddings route if present.

    Example embedding models: 'mxbai-embed-large', 'nomic-embed-text', or 'bge-m3'.
    Note: most chat-only models (e.g., llama3/llama4 variants) do not provide embeddings.
    """
    def __init__(self, model: str = "llama4:7b", base_url: str = None, timeout: float = 60.0):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def embed(self, texts: List[str]) -> np.ndarray:
        import requests  # lightweight; used only when you call embed()
        vecs: List[List[float]] = []
        for t in texts:
            # Prefer modern /api/embed with {input: ...}; fall back to legacy /api/embeddings with {prompt: ...}
            primary_url = f"{self.base_url}/api/embed"
            legacy_url = f"{self.base_url}/api/embeddings"

            # First try the modern endpoint
            resp = requests.post(primary_url, json={"model": self.model, "input": t}, timeout=self.timeout)
            if resp.status_code == 404:
                # Try legacy route
                resp = requests.post(legacy_url, json={"model": self.model, "prompt": t}, timeout=self.timeout)

            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                raise RuntimeError(
                    "Ollama embeddings call failed. Tried /api/embed and /api/embeddings. "
                    f"Base URL: {self.base_url} | Model: {self.model}. "
                    "Make sure Ollama is running and that you've pulled an embedding model, e.g.\n"
                    "    ollama pull nomic-embed-text\n"
                    "(Most chat models do not expose embeddings.)"
                ) from e

            data = resp.json()
            # Some builds return 'embeddings' (list of floats) and others 'embedding'.
            emb = data.get("embeddings") or data.get("embedding")
            if emb is None:
                raise ValueError(f"Unexpected embeddings response schema: keys={list(data.keys())}")

            # If API returns a list-of-lists for batch, take the first element
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                emb = emb[0]

            vecs.append(emb)

        return np.asarray(vecs, dtype=np.float32)


# ---------------- Simple demo embedder ----------------
class SimpleEmbedder:
    """
    Very small, dependency-free embedder for demos/tests.
    Creates a fixed-size bag-of-words hash vector.
    Satisfies the EmbeddingFn Protocol by exposing `.embed(texts) -> np.ndarray`.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                vecs[i, hash(tok) % self.dim] += 1.0
        return vecs


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _as_list(x: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(x, str):
        return [x]
    return list(x)


# ---------- Core Vector DB ----------

@dataclass
class _Category:
    ids: List[str]
    vectors: np.ndarray  # shape (N, D), normalized
    index: Any  # faiss index or None


class VectorDB:
    """
    In-memory, category-aware vector store with:
      - add/remove items
      - nearest-neighbor search scoped by category
      - save/load to disk
      - pluggable embedding backend (e.g., local Ollama)

    Uses cosine similarity (via inner product on normalized vectors).
    If FAISS is installed, searches are accelerated; otherwise uses NumPy.
    """

    def __init__(self, embedder: EmbeddingFn):
        self._embedder = embedder
        self._dim: Optional[int] = None
        self._cats: Dict[str, _Category] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}  # id -> {text, category, metadata}

    # ----- Public API -----

    def add_texts(
        self,
        category: str,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Embed and add texts to a category. Returns the assigned IDs.
        """
        if not texts:
            return []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        if len(ids) != len(texts):
            raise ValueError("ids (if provided) must have same length as texts.")
        if metadata is None:
            metadata = [{} for _ in texts]
        if len(metadata) != len(texts):
            raise ValueError("metadata (if provided) must have same length as texts.")

        vectors = self._embedder.embed(texts).astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError("embedder must return a 2D array (N, D).")
        self._maybe_set_dim(vectors.shape[1])
        if vectors.shape[1] != self._dim:
            raise ValueError(f"Embedding dim mismatch: got {vectors.shape[1]}, expected {self._dim}.")

        vectors = _normalize_rows(vectors)
        self._ensure_cat(category)
        cat = self._cats[category]

        if cat.vectors.size == 0:
            cat.vectors = vectors
        else:
            cat.vectors = np.vstack([cat.vectors, vectors])
        cat.ids.extend(ids)

        # store metadata
        for _id, t, m in zip(ids, texts, metadata):
            self._meta[_id] = {"text": t, "category": category, "metadata": m}

        self._rebuild_index(category)
        return ids

    def add_embeddings(
        self,
        category: str,
        vectors: np.ndarray,
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add precomputed embeddings to a category. Vectors will be normalized.
        """
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array (N, D).")
        self._maybe_set_dim(vectors.shape[1])
        if vectors.shape[1] != self._dim:
            raise ValueError(f"Embedding dim mismatch: got {vectors.shape[1]}, expected {self._dim}.")
        n = vectors.shape[0]
        ids = ids or [str(uuid.uuid4()) for _ in range(n)]
        if len(ids) != n:
            raise ValueError("ids (if provided) must have length equal to number of vectors.")
        metadata = metadata or [{} for _ in range(n)]
        if len(metadata) != n:
            raise ValueError("metadata (if provided) must have length equal to number of vectors.")
        texts = texts or [None] * n

        self._ensure_cat(category)
        cat = self._cats[category]
        vectors = _normalize_rows(vectors.astype(np.float32))

        if cat.vectors.size == 0:
            cat.vectors = vectors
        else:
            cat.vectors = np.vstack([cat.vectors, vectors])
        cat.ids.extend(ids)

        for _id, t, m in zip(ids, texts, metadata):
            self._meta[_id] = {"text": t, "category": category, "metadata": m}

        self._rebuild_index(category)
        return ids

    def remove(self, ids: Iterable[str]) -> int:
        """
        Remove items by ID. Returns the number removed.
        """
        ids = list(ids)
        if not ids:
            return 0
        removed = 0
        # group by category
        by_cat: Dict[str, List[str]] = {}
        for _id in ids:
            meta = self._meta.get(_id)
            if meta:
                by_cat.setdefault(meta["category"], []).append(_id)

        for category, ids_in_cat in by_cat.items():
            cat = self._cats.get(category)
            if not cat:
                continue
            keep_mask = np.ones(len(cat.ids), dtype=bool)
            id_to_idx = {i: idx for idx, i in enumerate(cat.ids)}
            for _id in ids_in_cat:
                idx = id_to_idx.get(_id)
                if idx is not None:
                    keep_mask[idx] = False
                    removed += 1
                    self._meta.pop(_id, None)
            # apply mask
            cat.ids = [i for i, keep in zip(cat.ids, keep_mask) if keep]
            if cat.vectors.size:
                cat.vectors = cat.vectors[keep_mask]
            self._rebuild_index(category)
        return removed

    def search(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        categories: Optional[Union[str, Iterable[str]]] = None,
        return_text: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Nearest neighbor search using cosine similarity within the specified categories.
        - query: text (will be embedded) or a precomputed vector of shape (D,)
        - categories: a single category, a list of categories, or None for all categories
        Returns a list of hits: {id, category, score, text, metadata}.
        """
        if isinstance(query, str):
            q = self._embedder.embed([query]).astype(np.float32)
        else:
            q = np.asarray(query, dtype=np.float32)[None, :]
        if q.ndim != 2 or q.shape[1] != (self._dim or q.shape[1]):
            self._maybe_set_dim(q.shape[1])
        if q.shape[1] != self._dim:
            raise ValueError(f"Query dim mismatch: got {q.shape[1]}, expected {self._dim}.")
        q = _normalize_rows(q)

        cats = list(self._cats.keys()) if categories is None else _as_list(categories)
        results: List[Tuple[str, str, float]] = []  # (id, category, score)

        for category in cats:
            cat = self._cats.get(category)
            if not cat or len(cat.ids) == 0:
                continue
            scores, idxs = self._search_in_category(cat, q, top_k)
            for s, idx in zip(scores, idxs):
                if idx == -1:
                    continue
                results.append((cat.ids[idx], category, float(s)))

        # sort and take top_k overall
        results.sort(key=lambda t: t[2], reverse=True)
        results = results[:top_k]

        # decorate with metadata
        out: List[Dict[str, Any]] = []
        for _id, category, score in results:
            m = self._meta.get(_id, {"text": None, "metadata": {}})
            out.append(
                {
                    "id": _id,
                    "category": category,
                    "score": score,
                    "text": m["text"] if return_text else None,
                    "metadata": m.get("metadata"),
                }
            )
        return out

    def list_categories(self) -> List[str]:
        return sorted(self._cats.keys())

    def count(self, category: Optional[str] = None) -> int:
        if category is None:
            return sum(len(c.ids) for c in self._cats.values())
        return len(self._cats.get(category, _Category([], np.empty((0, self._dim or 0), np.float32), None)).ids)

    # ----- Persistence -----

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """
        Save database to a directory:
          - meta.json: id -> {text, category, metadata}, plus dim
          - per-category arrays: {category}_vectors.npy, {category}_ids.json
        """
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta_path = path / "meta.json"
        meta_payload = {
            "dim": self._dim,
            "items": self._meta,  # large but explicit
            "categories": list(self._cats.keys()),
        }
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2))

        for category, cat in self._cats.items():
            # enforce stable, file-system-safe names
            safe = self._safe_category_name(category)
            np.save(path / f"{safe}_vectors.npy", cat.vectors)
            (path / f"{safe}_ids.json").write_text(json.dumps(cat.ids, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Union[str, pathlib.Path], embedder: EmbeddingFn) -> "VectorDB":
        path = pathlib.Path(path)
        meta_payload = json.loads((path / "meta.json").read_text())
        dim = meta_payload["dim"]
        inst = cls(embedder=embedder)
        inst._dim = dim
        inst._meta = meta_payload["items"]

        for category in meta_payload["categories"]:
            safe = inst._safe_category_name(category)
            vecs = np.load(path / f"{safe}_vectors.npy")
            ids = json.loads((path / f"{safe}_ids.json").read_text())
            inst._cats[category] = _Category(ids=ids, vectors=vecs, index=None)
            inst._rebuild_index(category)

        return inst

    # ----- Internals -----

    def _maybe_set_dim(self, d: int) -> None:
        if self._dim is None:
            self._dim = int(d)

    def _ensure_cat(self, category: str) -> None:
        if category not in self._cats:
            self._cats[category] = _Category(ids=[], vectors=np.empty((0, self._dim or 0), dtype=np.float32), index=None)

    def _rebuild_index(self, category: str) -> None:
        cat = self._cats[category]
        if not _FAISS_AVAILABLE or cat.vectors.size == 0:
            cat.index = None
            return
        # cosine via dot-product on normalized vectors
        index = faiss.IndexFlatIP(self._dim)
        index.add(cat.vectors.astype(np.float32))
        cat.index = index

    def _search_in_category(self, cat: _Category, q: np.ndarray, top_k: int) -> Tuple[List[float], List[int]]:
        n = len(cat.ids)
        if n == 0:
            return [], []
        k = min(top_k, n)
        if cat.index is not None:
            sims, idxs = cat.index.search(q.astype(np.float32), k)
            return sims[0].tolist(), idxs[0].tolist()
        # fallback: NumPy
        sims = (cat.vectors @ q[0]).astype(np.float32)  # (N,)
        idxs = np.argpartition(-sims, kth=k - 1)[:k]
        # sort top-k
        order = np.argsort(-sims[idxs])
        idxs = idxs[order]
        return sims[idxs].tolist(), idxs.tolist()

    @staticmethod
    def _safe_category_name(name: str) -> str:
        # Keep readable but filesystem-safe
        keep = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
        return keep or "category"

#
# ---------- Benchmark utilities ----------

def _generate_random_texts(
    n: int,
    avg_words: int = 50,
    std_words: int = 10,
    vocab_size: int = 5000,
    seed: int = 0,
) -> List[str]:
    """
    Generate n synthetic sentences made of random tokens. This avoids any
    external dependencies while giving reasonably realistic lengths.
    """
    rng = np.random.default_rng(seed)
    vocab = [f"tok{i}" for i in range(vocab_size)]
    texts: List[str] = []
    for _ in range(n):
        length = int(max(1, round(rng.normal(avg_words, std_words))))
        tokens = rng.integers(0, vocab_size, size=length)
        texts.append(" ".join(vocab[t] for t in tokens))
    return texts


def _estimate_token_counts(
    texts: List[str],
    method: Literal["whitespace", "chars_per_token"] = "whitespace",
) -> List[int]:
    """
    Crude token-count estimate when a model-specific tokenizer isn't available.
    - "whitespace": count words split on whitespace.
    - "chars_per_token": assume ~4 chars per token (very rough heuristic).
    """
    if method == "whitespace":
        return [len(t.split()) for t in texts]
    # fallback heuristic (~4 chars per token is a common back-of-the-envelope)
    return [max(1, len(t) // 4) for t in texts]


def benchmark_embedding_speed(
    embedder: EmbeddingFn,
    num_texts: int = 1000,
    avg_words: int = 50,
    std_words: int = 10,
    batch_size: int = 32,
    seed: int = 0,
    token_counter: Optional[Callable[[List[str]], List[int]]] = None,
    token_estimate_method: Literal["whitespace", "chars_per_token"] = "whitespace",
) -> Dict[str, Any]:
    """
    Measure throughput of an embedding implementation.

    Parameters
    ----------
    embedder : EmbeddingFn
        Object with `.embed(List[str]) -> np.ndarray`.
    num_texts : int
        Number of synthetic examples to embed.
    avg_words, std_words : int
        Distribution of words per synthetic example.
    batch_size : int
        Number of texts to send to `.embed` at once.
    seed : int
        RNG seed for reproducibility of synthetic data.
    token_counter : Optional[Callable[[List[str]], List[int]]]
        If provided, should return per-text token counts for the given texts.
        Use this if you have a real tokenizer (e.g., tiktoken). If not provided,
        we estimate tokens using `token_estimate_method`.
    token_estimate_method : Literal["whitespace", "chars_per_token"]
        Method used when `token_counter` is None.

    Returns
    -------
    Dict[str, Any]
        A dictionary with raw timings and derived throughput metrics.
    """
    # Prepare synthetic data
    texts = _generate_random_texts(
        n=num_texts, avg_words=avg_words, std_words=std_words, seed=seed
    )

    # Determine embedding dimensionality with a warmup call (also warms caches)
    warm_n = min(batch_size, max(1, num_texts // 50))  # small warmup batch
    warm_vecs = embedder.embed(texts[:warm_n])
    if warm_vecs.ndim != 2:
        raise ValueError("embedder must return a 2D array (N, D)")
    dim = int(warm_vecs.shape[1])

    # Count tokens (real or estimated)
    if token_counter is not None:
        token_counts = token_counter(texts)
    else:
        token_counts = _estimate_token_counts(texts, method=token_estimate_method)
    total_tokens = int(sum(int(x) for x in token_counts))

    # Timed run
    start = time.perf_counter()
    i = 0
    while i < num_texts:
        batch = texts[i : i + batch_size]
        embedder.embed(batch)
        i += len(batch)
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000.0
    texts_per_s = num_texts / elapsed if elapsed > 0 else float("inf")
    tokens_per_s = total_tokens / elapsed if elapsed > 0 else float("inf")
    ms_per_text = (elapsed_ms / num_texts) if num_texts else float("nan")
    ms_per_token = (elapsed_ms / total_tokens) if total_tokens else float("nan")

    return {
        "num_texts": num_texts,
        "avg_words": avg_words,
        "std_words": std_words,
        "batch_size": batch_size,
        "dim": dim,
        "elapsed_seconds": elapsed,
        "elapsed_ms": elapsed_ms,
        "total_tokens": total_tokens,
        "texts_per_second": texts_per_s,
        "tokens_per_second": tokens_per_s,
        "ms_per_text": ms_per_text,
        "ms_per_token": ms_per_token,
        "token_estimation": ("custom" if token_counter is not None else token_estimate_method),
    }


def print_benchmark(result: Dict[str, Any]) -> None:
    """Pretty-print the result from `benchmark_embedding_speed`."""
    print("\nEmbedding benchmark")
    print("-------------------")
    print(f"Dimension:           {result['dim']}")
    print(f"Samples:             {result['num_texts']} (batch={result['batch_size']})")
    print(f"Text length ~words:  {result['avg_words']} ± {result['std_words']}")
    print(f"Token method:        {result['token_estimation']}")
    print(f"Elapsed:             {result['elapsed_ms']:.1f} ms")
    print(f"Throughput (texts):  {result['texts_per_second']:.2f} / s")
    print(f"Throughput (tokens): {result['tokens_per_second']:.2f} / s (total={result['total_tokens']})")
    print(f"Latency per text:    {result['ms_per_text']:.3f} ms")
    print(f"Latency per token:   {result['ms_per_token']:.6f} ms")

# test run
if __name__ == "__main__":
    # Option A: use a local Ollama embedder if you have an embedding-capable model running
    emb = OllamaEmbedder(model="nomic-embed-text")


    Vec = VectorDB(emb)
