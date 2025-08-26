#!/usr/bin/env python3
"""
Build a vector DB from the HuggingFace dataset:
  lang-uk/recruitment-dataset-candidate-profiles-english

- Creates category: "total_CV"
- Each row becomes one document combining relevant fields into a CV-like block
- Saves a VectorDB directory (default: ./recruitment_vector_db)

The mapper uses the schema documented in the dataset card:
  Position, Moreinfo, Looking For, Highlights, Primary Keyword, English Level,
  Experience Years, CV, CV_lang, id

Prereqs:
  pip install datasets requests numpy tqdm
  ollama pull nomic-embed-text  (or any embedding-capable model)
Env overrides:
  OLLAMA_EMBED_MODEL=...   (default: nomic-embed-text)
  RECRUITMENT_VDB_OUT=...  (default: recruitment_vector_db)
"""

from  benchmark import getQuantTest

import os
import uuid
import pathlib
from typing import Any, Dict, List, Union, Iterable, Optional, Tuple
import argparse

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import math
import numpy as np
import threading
import time


# --- lightweight debug helpers (enabled if SELECTION_DEBUG env var is set) ---
def _dbg_enabled() -> bool:
    return os.environ.get("SELECTION_DEBUG", "").strip().lower() not in {"", "0", "false"}


def _dbg(*args: object) -> None:
    if _dbg_enabled():
        print("[DEBUG]", *args)


# Import your classes from FAISS.py (same folder)
from FAISS import VectorDB, OllamaEmbedder, EmbeddingFn  # type: ignore

# --- LLM benchmarking config (CLI-overridable) ---
_llm_debug = False
_llm_batch_timeout = 120   # seconds per batch before we move on
_llm_retries = 0           # retries per batch on timeout/error
_llm_batch_size = 20       # keywords per LLM call


# ---------- helpers to build a CV-like blob ----------

def _stringify(value: Any) -> str:
    """Render lists/dicts/scalars into readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            sv = _stringify(v)
            if sv:
                parts.append(f"{k.capitalize()}: {sv}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        parts = [_stringify(v) for v in value]
        return "; ".join(p for p in parts if p)
    return str(value)


def build_total_cv(record: Dict[str, Any]) -> str:
    """
    Combine relevant fields from a candidate profile into a single CV-like text block.
    Uses the official schema first (see HF README):
      Position, Moreinfo, Looking For, Highlights, Primary Keyword, English Level,
      Experience Years, CV, CV_lang, id
    Falls back to generic fields if any are missing.
    """
    lines: List[str] = []

    # Official fields from the dataset card
    position = record.get("Position")
    moreinfo = record.get("Moreinfo")
    looking_for = record.get("Looking For")
    highlights = record.get("Highlights")
    primary_kw = record.get("Primary Keyword")
    eng_level = record.get("English Level")
    exp_years = record.get("Experience Years")
    cv_text = record.get("CV")
    cv_lang = record.get("CV_lang")
    rid = record.get("id")

    # Header line: Position (and id if present)
    header_bits: List[str] = []
    if position:
        header_bits.append(_stringify(position))
    if rid:
        header_bits.append(f"ID: {_stringify(rid)}")
    if header_bits:
        lines.append(" | ".join(header_bits))

    # Key meta fields
    if looking_for:
        lines.append(f"Looking For: {_stringify(looking_for)}")
    if primary_kw:
        lines.append(f"Primary Keyword: {_stringify(primary_kw)}")
    if eng_level:
        lines.append(f"English Level: {_stringify(eng_level)}")
    if exp_years not in (None, ""):
        lines.append(f"Experience Years: {_stringify(exp_years)}")
    if moreinfo:
        lines.append(f"More info: {_stringify(moreinfo)}")
    if highlights:
        lines.append(f"Highlights: {_stringify(highlights)}")
    if cv_lang:
        lines.append(f"CV Language: {_stringify(cv_lang)}")

    # Main CV body last (so search hits find meta first but keep rich text)
    if cv_text:
        lines.append("\n" + _stringify(cv_text))

    # Fallbacks for unexpected keys (exclude known keys)
    known = {"Position","Moreinfo","Looking For","Highlights","Primary Keyword","English Level","Experience Years","CV","CV_lang","id","__index_level_0__"}
    for k, v in record.items():
        if k in known:
            continue
        sv = _stringify(v)
        if sv:
            lines.append(f"{k}: {sv}")

    return "\n".join([ln for ln in lines if ln])

# ---------- ESCO skills indexing & CV keyword attachment ----------
def index_esco_skills_to_vdb(
    vdb: VectorDB,
    esco_df_en: pd.DataFrame,
    category: str = "esco_skill",
    batch_size: int = 128,
    groups_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Index ESCO skills using a concise, minimal label per skill.

    Minimal attachment strategy:
      - Label = preferredLabel if present, else the *shortest* token from altLabel.
      - Representative keyword = immediate parent group label (if available).
      - Stored text is short only:
            "Skill: {label}\nCategory: {parent_label}\nType: {skillType}"
    """
    df = esco_df_en.copy()

    # Ensure useful columns exist
    for col in ["altLabel", "preferredLabel", "skillType", "conceptUri", "broader"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Pick up a parent label column if it already exists
    parent_label_col = None
    for cand in [
        "broader_pref_label", "broaderPrefLabel",
        "broaderLabel_en", "broaderLabel",
        "parentLabel", "groupLabel", "group_name",
    ]:
        if cand in df.columns:
            parent_label_col = cand
            break

    # Join groups_df to resolve parent labels if needed
    if parent_label_col:
        df["parentLabel"] = df[parent_label_col].fillna("").astype(str)
    elif "broader" in df.columns and groups_df is not None and not groups_df.empty:
        g_uri_col = next((c for c in ["conceptUri","uri","groupUri","id"] if c in groups_df.columns), None)
        g_label_col = next((c for c in ["preferredLabel","altLabel","label_en","label"] if c in groups_df.columns), None)
        if g_uri_col and g_label_col:
            groups_small = groups_df[[g_uri_col, g_label_col]].copy()
            groups_small.columns = ["broader", "parentLabel"]
            df = df.merge(groups_small, on="broader", how="left")
        else:
            df["parentLabel"] = ""
    else:
        df["parentLabel"] = ""

    # Helper to pick a minimal term from a delimited synonyms string
    def _minimal_term(s: str) -> str:
        if not s:
            return s
        seps = ["␞", "|", ",", ";", "/", "·", "•", "\n"]
        parts = [s]
        for sep in seps:
            tmp = []
            for p in parts:
                tmp.extend(p.split(sep))
            parts = tmp
        parts = [p.strip() for p in parts if p and p.strip()]
        if not parts:
            return s.strip()
        parts.sort(key=lambda p: (len(p.split()), len(p)))  # by fewest words, then shortest
        return parts[0]

    # Minimal display label: prefer preferredLabel; else shortest altLabel token
    df["displayLabel"] = df.apply(
        lambda r: (r.get("preferredLabel") or "").strip()
                  or _minimal_term((r.get("altLabel") or "").strip()),
        axis=1,
    )

    # Build concise formatted text (no descriptions!)
    def _format_row(r: pd.Series) -> str:
        s = (r.get("displayLabel") or "").strip()
        c = (r.get("parentLabel") or "").strip()
        t = (r.get("skillType") or "").strip()
        parts = [f"Skill: {s}"]
        if c:
            parts.append(f"Category: {c}")
        if t:
            parts.append(f"Type: {t}")
        return "\n".join(parts)

    df["formatted_skill"] = df.apply(_format_row, axis=1)

    # IDs
    ids = [str(x) if (isinstance(x, str) and x) else str(uuid.uuid4())
           for x in (df["conceptUri"].tolist() if "conceptUri" in df.columns else ["" for _ in range(len(df))])]

    # Texts + compact metadata
    texts = df["formatted_skill"].fillna("").astype(str).tolist()
    metas = []
    keep_cols = [c for c in [
        "displayLabel","preferredLabel","altLabel","conceptUri","skillType",
        "parentLabel","broader","narrower",
    ] if c in df.columns]
    for _, row in df.iterrows():
        meta = {c: row.get(c) for c in keep_cols}
        meta["attachedLabel"] = (row.get("displayLabel") or "").strip()
        meta["representativeKeyword"] = (row.get("parentLabel") or "").strip() or (row.get("skillType") or "").strip()
        metas.append(meta)

    # Batch insert
    inserted_ids: List[str] = []
    i = 0
    with tqdm(total=len(texts), desc="Indexing ESCO skills", unit="skills") as pbar:
        while i < len(texts):
            inserted_ids.extend(
                vdb.add_texts(category=category,
                              texts=texts[i:i+batch_size],
                              ids=ids[i:i+batch_size],
                              metadata=metas[i:i+batch_size])
            )
            pbar.update(min(batch_size, len(texts)-i))
            i += batch_size

    return inserted_ids
# ---------- adaptive selection helpers (elbow cutoff + optional MMR diversification) ----------

def _find_elbow_cutoff(
    scores: List[float],
) -> int:
    """Pick cutoff at the *largest consecutive drop* in the (descending) scores.

    To reduce outlier influence from the very top and very bottom values,
    we ignore the absolute max and min score when computing the differences.
    If fewer than 3 scores are provided, we default to keeping 1 item.

    Returns
    -------
    int
        Number of items to keep. If ``scores`` is non-empty, this is at least 1;
        if ``scores`` is empty, this returns 0.
    """
    if not scores:
        return 0

    # Ensure descending order and stable numeric type
    s = sorted((float(x) for x in scores), reverse=True)
    _dbg("scores(desc)=", s)

    n = len(s)
    if n == 1:
        return 1
    if n == 2:
        # Only one diff possible; keep the top one
        return 1

    # Remove the absolute max and min values to avoid extreme outliers dominating
    s_trim = s[1:-1]
    if len(s_trim) < 2:
        # Not enough to compute a trimmed diff; fall back to keeping one
        return 1

    # Find index of the maximum consecutive drop (the elbow) on the trimmed list
    diffs_trim = [s_trim[i] - s_trim[i + 1] for i in range(len(s_trim) - 1)]
    _dbg("diffs_trim (without max/min)=", diffs_trim)
    elbow_idx_trim = int(np.argmax(diffs_trim)) if diffs_trim else 0

    # Map back to the original index (shift by +1 because of the left trim)
    elbow_idx_orig = elbow_idx_trim + 1
    keep_n = elbow_idx_orig + 1
    _dbg(f"elbow_idx_orig={elbow_idx_orig} keep_n={keep_n}")

    return keep_n


def _mmr_select(query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lambda_mult: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance selection.
    Assumes `query_vec` is shape (D,) and `cand_vecs` is shape (N, D), both normalized.
    Returns a list of indices into `cand_vecs` in the chosen order (length <= k).
    """
    if cand_vecs.size == 0 or k <= 0:
        return []
    # Normalize defensively (VectorDB already stores normalized vectors, but callers may pass raw arrays)
    q = query_vec.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    C = cand_vecs.astype(np.float32)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    sim_q = C @ q  # (N,)
    selected: List[int] = []
    remaining = set(range(C.shape[0]))

    while remaining and len(selected) < k:
        best_i = None
        best_score = -1e9
        if not selected:
            # First pick: highest similarity to query
            best_i = int(np.argmax(sim_q))
            best_score = float(sim_q[best_i])
        else:
            # Precompute similarity to the selected set for all candidates
            S = C[selected]  # (m, D)
            # For each candidate i, diversity term is max_j sim(i, selected_j)
            # We'll compute on the fly for clarity (N is small, k is small)
            for i in list(remaining):
                div = float(np.max(C[i] @ S.T)) if S.size else 0.0
                score = float(lambda_mult * sim_q[i] - (1.0 - lambda_mult) * div)
                if score > best_score:
                    best_score = score
                    best_i = i
        selected.append(int(best_i))
        remaining.remove(int(best_i))

    return selected
def _find_knee_cutoff(scores: List[float]) -> int:
    """Knee via max perpendicular distance to the line joining endpoints.

    Steps:
      1) Sort scores descending.
      2) Normalize y to [0,1] using first and last values.
      3) For points (x=i, y[i]), pick the i with max distance to the chord.
    Returns number of items to keep (>=0).
    """
    if not scores:
        return 0
    s = sorted((float(x) for x in scores), reverse=True)
    n = len(s)
    if n <= 2:
        return 1

    y_first, y_last = s[0], s[-1]
    if abs(y_first - y_last) < 1e-12:
        return 1  # flat curve

    # y normalized so endpoints are (0,1) and (n-1,0)
    y = [(v - y_last) / (y_first - y_last) for v in s]

    # Line through endpoints (0,1) and (n-1,0): ax + by + c = 0
    a = -1.0               # y1 - y0
    b = -(n - 1.0)         # x0 - x1
    c = (n - 1.0)          # x1*y0 - y1*x0
    denom = math.sqrt(a*a + b*b) + 1e-12

    best_i, best_d = 1, -1.0
    for i in range(1, n - 1):  # exclude endpoints
        xi, yi = float(i), float(y[i])
        d = abs(a * xi + b * yi + c) / denom
        if d > best_d:
            best_d, best_i = d, i

    return best_i + 1  # keep through best_i

def select_relevant_skills_for_text(
    vdb: VectorDB,
    query_text: str,
    category_skill: str = "esco_skill",
    candidate_k: int = 100,
) -> List[Dict[str, Any]]:
    """Search ESCO skills for `query_text` and pick the most relevant ones
    by cutting the sorted candidate list at the largest score gap (elbow).

    Steps:
      1) Retrieve top `candidate_k` by cosine similarity.
      2) Determine cutoff with the largest consecutive drop (using `_find_elbow_cutoff`).

    Returns a list of dicts: {skill_id, label, score} in final order.
    """
    # Candidate pool
    hits = vdb.search(query=query_text, top_k=candidate_k, categories=category_skill, return_text=False)
    if not hits:
        return []

    # Decide how many to keep purely via the largest gap
    scores = [h["score"] for h in hits]
    keep_n = _find_knee_cutoff(scores)

    # Take the top slice only
    chosen = hits[:keep_n]

    # Materialize attachment items using minimal-label logic
    attached: List[Dict[str, Any]] = []
    for h in chosen:
        # Minimal label helper: pick the shortest clean token from a delimited string
        def _minimal_term(s: str) -> str:
            if not s:
                return s
            seps = ["␞", "|", ",", ";", "/", "·", "•", "\n"]
            parts = [s]
            for sep in seps:
                tmp = []
                for p in parts:
                    tmp.extend(p.split(sep))
                parts = tmp
            parts = [p.strip() for p in parts if p and p.strip()]
            if not parts:
                return s.strip()
            parts.sort(key=lambda p: (len(p.split()), len(p)))  # fewest words, then shortest length
            return parts[0]

        skill_meta = vdb._meta.get(h["id"], {}).get("metadata", {})
        # Prefer ESCO's canonical name; else the precomputed minimal label; else a minimized synonym
        label = (
            (skill_meta.get("preferredLabel") or "").strip()
            or (skill_meta.get("attachedLabel") or "").strip()
            or _minimal_term((skill_meta.get("altLabel") or "").strip())
            or (vdb._meta.get(h["id"], {}).get("text") or "").split("\n", 1)[0].replace("Skill: ", "").strip()
        )
        rep_kw = (skill_meta.get("representativeKeyword") or skill_meta.get("parentLabel") or "").strip()
        attached.append({
            "skill_id": h["id"],
            "label": label,
            "category": rep_kw,
            "score": float(h["score"]),
        })
    return attached


def attach_top_keywords_to_cvs(
    vdb: VectorDB,
    category_cv: str = "total_CV",
    category_skill: str = "esco_skill",
    candidate_k: int = 50,
) -> Dict[str, List[Dict[str, Any]]]:
    """For each CV in `category_cv`, attach the most relevant ESCO skills
    by taking a candidate pool and cutting at the largest score gap.

    Results are stored in `metadata.top_esco_skills`.
    Returns: mapping cv_id -> list of attached skill dicts.
    """
    if category_cv not in vdb._cats:
        raise ValueError(f"Category not found: {category_cv}")
    if category_skill not in vdb._cats:
        raise ValueError(f"Category not found: {category_skill}")

    results_by_cv: Dict[str, List[Dict[str, Any]]] = {}
    cv_cat = vdb._cats[category_cv]

    bar_fmt = "{l_bar}{bar} | {n_fmt}/{total_fmt} [elapsed {elapsed} < eta {remaining}]"
    for cv_id in tqdm(cv_cat.ids, desc="Attaching ESCO skills to CVs", unit="cv", dynamic_ncols=True, bar_format=bar_fmt):
        meta = vdb._meta.get(cv_id, {})
        cv_text = meta.get("text") or ""
        attached = select_relevant_skills_for_text(
            vdb,
            query_text=cv_text,
            category_skill=category_skill,
            candidate_k=candidate_k,
        )

        # LLM validation: keep only keywords validated as True
        labels = [item.get("label", "") for item in attached if item.get("label")]
        keyword_truths: Dict[str, bool] = run_keyword_benchmark(
            cv_text,
            labels,
            batch_size=_llm_batch_size,
        )
        true_attached = [item for item in attached if keyword_truths.get(item.get("label", ""), False)]

        # Persist only validated skills and store the truth map for auditing
        meta.setdefault("metadata", {})
        meta["metadata"]["top_esco_skills"] = true_attached
        meta["metadata"]["keyword_truths"] = keyword_truths
        vdb._meta[cv_id] = meta
        results_by_cv[cv_id] = true_attached

    return results_by_cv



# ---------- LLM keyword benchmark (batching) ----------

def run_keyword_benchmark(
    cv_text: str,
    keywords: List[str],
    batch_size: int = None,
    debug: Optional[bool] = None,
    batch_timeout: Optional[int] = None,
    retries: Optional[int] = None,
) -> Dict[str, bool]:
    """Run getQuantTest over all keywords in batches and return a merged mapping.

    Parameters
    ----------
    cv_text : str
        The raw CV text to evaluate against.
    keywords : List[str]
        List of keyword strings to check.
    batch_size : int or None
        Number of keywords per LLM call (to avoid oversized prompts). Defaults to global _llm_batch_size.
    debug : bool or None
        If True, print detailed per-batch diagnostics. Defaults to global _llm_debug.
    batch_timeout : int or None
        Seconds to wait per batch before skipping (best-effort). Defaults to global _llm_batch_timeout.
    retries : int or None
        Retries per batch when a timeout/error occurs. Defaults to global _llm_retries.

    Returns
    -------
    Dict[str, bool]
        Mapping of each provided keyword to a boolean truth value.
    """
    # Resolve defaults from module-level config
    if batch_size is None:
        batch_size = _llm_batch_size
    if debug is None:
        debug = _llm_debug
    if batch_timeout is None:
        batch_timeout = _llm_batch_timeout
    if retries is None:
        retries = _llm_retries

    def _log(*a: Any) -> None:
        if debug:
            print("[LLM]", *a)

    # Defensive cleanup & stable order
    seen = set()
    ordered_keywords: List[str] = []
    for k in keywords:
        if not isinstance(k, str):
            k = str(k)
        k2 = k.strip()
        if not k2:
            continue
        if k2 not in seen:
            seen.add(k2)
            ordered_keywords.append(k2)

    results: Dict[str, bool] = {}

    def _call_batch_with_timeout(batch: List[str]) -> Optional[Dict[str, Any]]:
        """Call getQuantTest in a background thread and wait up to batch_timeout seconds.
        Returns the dict on success, None on timeout. Exceptions are raised to caller.
        """
        out: Dict[str, Any] = {}
        exc: Dict[str, BaseException] = {}

        def _worker():
            try:
                out.update(getQuantTest(batch, cv_text))
            except BaseException as e:  # capture any error to re-raise in main thread
                exc["e"] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(batch_timeout)
        if t.is_alive():
            return None  # timed out
        if exc:
            raise exc["e"]
        return out

    total = len(ordered_keywords)
    start_all = time.monotonic()
    for i in range(0, total, batch_size):
        batch = ordered_keywords[i:i + batch_size]
        batch_idx = i // batch_size + 1
        n_batches = (total + batch_size - 1) // batch_size
        _log(f"Batch {batch_idx}/{n_batches}: size={len(batch)} | timeout={batch_timeout}s | retries={retries}")
        if debug:
            preview = ", ".join(batch[:5]) + (" ..." if len(batch) > 5 else "")
            _log(f"Keywords preview: {preview}")
        attempt = 0
        while True:
            attempt += 1
            t0 = time.monotonic()
            try:
                res = _call_batch_with_timeout(batch)
                if res is None:
                    _log(f"Batch {batch_idx} timed out after {batch_timeout}s (attempt {attempt}).")
                    if attempt <= retries:
                        _log("Retrying batch...")
                        continue
                    # mark as False and move on
                    for bk in batch:
                        results.setdefault(bk, False)
                    break
                # Merge results
                for bk in batch:
                    v = res.get(bk, False)
                    if isinstance(v, bool):
                        results.setdefault(bk, v)
                    elif isinstance(v, (int, float)):
                        results.setdefault(bk, bool(v))
                    elif isinstance(v, str):
                        results.setdefault(bk, v.strip().lower() in {"true", "t", "yes", "y", "1"})
                    else:
                        results.setdefault(bk, False)
                dt = time.monotonic() - t0
                _log(f"Batch {batch_idx} completed in {dt:.2f}s.")
                break
            except Exception as e:
                _log(f"Batch {batch_idx} error: {e!r}")
                if attempt <= retries:
                    _log("Retrying batch after error...")
                    continue
                for bk in batch:
                    results.setdefault(bk, False)
                break

    _log(f"All batches done in {time.monotonic() - start_all:.2f}s. Total keywords={total}.")
    return results

# ---------- utility: attach and print ESCO skills for a single CV file ----------

def attach_and_print_esco_skills_for_cv_file(
    vdb: VectorDB,
    filepath: Union[str, pathlib.Path],
    category_cv: str = "total_CV",
    category_skill: str = "esco_skill",
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Read a CV from a local .txt file, add it to the VectorDB, attach the
    closest ESCO skills to its metadata, and print them.

    Displays a progress bar with ETA across 6 steps.

    Returns the attached list of {skill_id, label, score}.
    """
    p = pathlib.Path(filepath)

    bar_fmt = "{l_bar}{bar} | {n_fmt}/{total_fmt} [elapsed {elapsed} < eta {remaining}]"
    with tqdm(total=7, desc=f"Processing {p.name}", unit="step", dynamic_ncols=True, bar_format=bar_fmt) as pbar:
        # 1) Validate file exists
        if not p.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        pbar.update(1)

        # 2) Ensure categories exist
        if category_skill not in vdb._cats:
            raise ValueError(
                f"Skills category '{category_skill}' not found. Index ESCO skills first (index_esco_skills_to_vdb)."
            )
        if category_cv not in vdb._cats:
            vdb._ensure_cat(category_cv)
        pbar.update(1)

        # 3) Load text
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise ValueError(f"CV file is empty: {filepath}")
        pbar.update(1)

        # 4) Add to the DB
        cv_id = f"cv::{p.name}::{uuid.uuid4()}"
        vdb.add_texts(
            category=category_cv,
            texts=[text],
            ids=[cv_id],
            metadata=[{"source_file": str(p)}],
        )
        pbar.update(1)

        # 5) Search & select ESCO skills
        attached: List[Dict[str, Any]] = select_relevant_skills_for_text(
            vdb,
            query_text=text,
            category_skill=category_skill,
            candidate_k=1000,
        )
        pbar.update(1)

        # 6) LLM benchmark for all selected keywords (batched)
        labels = [item.get("label", "") for item in attached if item.get("label")]
        if _llm_debug:
            print(f"[LLM] Starting benchmark: labels={len(labels)}, batch_size={_llm_batch_size}, timeout={_llm_batch_timeout}s, retries={_llm_retries}")
        keyword_truths: Dict[str, bool] = run_keyword_benchmark(
            text,
            labels,
            batch_size=_llm_batch_size,
        )
        pbar.update(1)

        # 7) Persist on CV metadata (ONLY LLM-true skills + full truth map)
        meta = vdb._meta.get(cv_id, {})
        meta.setdefault("metadata", {})
        # Filter attached skills to only those validated as True by the LLM
        true_attached = [item for item in attached if keyword_truths.get(item.get("label", ""), False)]
        meta["metadata"]["top_esco_skills"] = true_attached
        meta["metadata"]["keyword_truths"] = keyword_truths
        vdb._meta[cv_id] = meta
        pbar.update(1)

    # Retrieve keyword truths & the filtered (validated) skills
    meta = vdb._meta.get(cv_id, {})
    kt = meta.get("metadata", {}).get("keyword_truths", {})
    true_attached = meta.get("metadata", {}).get("top_esco_skills", [])

    total_selected = len(attached)
    total_true = len(true_attached)

    # Print only validated keywords
    print(f"Validated {total_true} ESCO skill(s) for CV file: {p.name}")
    print("-" * (36 + len(str(total_true))))

    for i, item in enumerate(true_attached, start=1):
        label = item.get("label", "")
        cat_kw = item.get("category") or ""
        cat_suffix = f" [{cat_kw}]" if cat_kw else ""
        print(f"{i:2d}. {label}{cat_suffix}\t[id={item['skill_id']}]")

    # If debugging is enabled, also show the full LLM truth map and summary
    if _llm_debug:
        try:
            import json as _json
            print("\n[DEBUG] LLM keyword benchmark (full results):")
            print(_json.dumps(kt, ensure_ascii=False, indent=2, sort_keys=True))
            true_count = sum(1 for v in kt.values() if v)
            total_count = len(kt)
            print(f"[DEBUG] Summary: {true_count} true / {total_count} total")
        except Exception as _e:
            print(f"[WARN] Could not pretty-print keyword truths: {_e}")

    return attached


# ---------- main pipeline ----------

def import_recruitment_dataset_and_build_db(
    embedder: EmbeddingFn,
    out_dir: Union[str, pathlib.Path] = "recruitment_vector_db",
    batch_size: int = 64,
) -> Tuple[VectorDB, str]:
    """
    Download dataset, build VectorDB with category 'total_CV', save to out_dir.
    Returns a tuple: (VectorDB instance, saved directory path as string).
    """
    bar_fmt = "{l_bar}{bar} | {n_fmt}/{total_fmt} [elapsed {elapsed} < eta {remaining}]"
    stepbar = tqdm(total=3, desc="Build CV VectorDB (steps)", unit="step", dynamic_ncols=True, bar_format=bar_fmt)
    try:
        ds = load_dataset("lang-uk/recruitment-dataset-candidate-profiles-english")
    except Exception as e:
        raise RuntimeError(
            "Failed to load 'lang-uk/recruitment-dataset-candidate-profiles-english'.\n"
            "Ensure `pip install datasets` and internet access.\n"
            f"Original error: {e}"
        )

    dataset = ds['train'].select(range(1000))
    stepbar.update(1)

    id_field = 'id' if 'id' in dataset.column_names else None

    vdb = VectorDB(embedder)

    texts: List[str] = []
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []

    # Collect rows
    for row in tqdm(dataset, total=len(dataset), desc="Building CV texts", unit="docs"):
        rid = str(row.get(id_field) or uuid.uuid4())
        ids.append(rid)
        metas.append({k: row.get(k) for k in dataset.column_names})
        texts.append(build_total_cv(row))

    # Add in batches
    i = 0
    n = len(texts)
    with tqdm(total=n, desc="Adding to VectorDB", unit="docs") as pbar:
        while i < n:
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_meta = metas[i:i+batch_size]
            vdb.add_texts(category="total_CV", texts=batch_texts, ids=batch_ids, metadata=batch_meta)
            i += len(batch_texts)
            pbar.update(len(batch_texts))

    stepbar.update(1)
    out_path = pathlib.Path(out_dir)
    vdb.save(out_path)
    stepbar.update(1)
    stepbar.close()
    return vdb, str(out_path)


def load_vector_db(embedder: EmbeddingFn, dir_path: Union[str, pathlib.Path]) -> VectorDB:
    """Load a previously saved VectorDB directory."""
    p = pathlib.Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(f"VectorDB directory not found: {dir_path}")
    print(f"Loading VectorDB from: {p}")
    return VectorDB.load(p, embedder)

def ensure_esco_skills_category(
    vdb: VectorDB,
    skills_tsv_path: Union[str, pathlib.Path],
    category: str = "esco_skill",
) -> None:
    """Ensure the ESCO skills category exists; if missing and TSV is present, index it.

    Looks for an optional groups TSV (env `ESCO_GROUPS_TSV`, default 'skillGroups.tsv')
    to resolve parent group labels used as representative keywords.
    """
    if category in vdb.list_categories():
        print(f"ESCO skills category '{category}' already present.")
        return
    p = pathlib.Path(skills_tsv_path)
    if not p.exists():
        print(f"ESCO TSV not found at {p}. Skipping skill indexing.")
        return
    print(f"Indexing ESCO skills from: {p}")
    esco_df = pd.read_csv(p, sep="\t")
    esco_df_en = esco_df[esco_df.get("language") == "en"].copy()
    if esco_df_en.empty:
        raise RuntimeError("No English ('en') rows found in skills.tsv")

    # Try to load groups TSV for parent labels
    groups_path = os.environ.get("ESCO_GROUPS_TSV", "skillGroups.tsv")
    groups_df: Optional[pd.DataFrame] = None
    gp = pathlib.Path(groups_path)
    if gp.exists():
        try:
            groups_df = pd.read_csv(gp, sep="\t")
            print(f"Loaded skill groups from: {gp}")
        except Exception as e:
            print(f"[WARN] Could not read groups TSV at {gp}: {e}")
            groups_df = None
    else:
        print(f"(Optional) groups TSV not found at {gp}; proceeding without parent label join.")

    inserted_skill_ids = index_esco_skills_to_vdb(
        vdb, esco_df_en, category=category, batch_size=256, groups_df=groups_df
    )
    print(f"Indexed {len(inserted_skill_ids)} ESCO skills into category '{category}'.")

def interactive_attach_loop(
    vdb: VectorDB,
    category_cv: str = "total_CV",
    category_skill: str = "esco_skill",
    top_k: int = 1000,
) -> None:
    """Interactive loop: type a path to a .txt CV and see attached ESCO skills."""
    print("\nInteractive test mode. Enter a path to a .txt CV (blank to quit).")
    while True:
        try:
            path_to_cv = input("CV file path (blank to quit): ").strip()
        except EOFError:
            break
        if not path_to_cv:
            print("Exiting.")
            break
        try:
            attach_and_print_esco_skills_for_cv_file(
                vdb,
                filepath=path_to_cv,
                category_cv=category_cv,
                category_skill=category_skill,
                top_k=top_k,
            )
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build or load a CV VectorDB and run interactive ESCO skill attachment tests."
        )
    )
    parser.add_argument(
        "--load",
        type=str,
        default=os.environ.get("RECRUITMENT_VDB_LOAD"),
        help="Path to an existing VectorDB directory to load.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force building the VectorDB from the dataset (ignores --load).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.environ.get("RECRUITMENT_VDB_OUT", "recruitment_vector_db"),
        help="Output directory when building a new VectorDB.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        help="Ollama embedding model to use.",
    )
    parser.add_argument(
        "--skills",
        type=str,
        default=os.environ.get("ESCO_SKILLS_TSV", "skills.tsv"),
        help=(
            "Path to ESCO skills TSV (used when building OR when the loaded DB is missing skills)."
        ),
    )
    parser.add_argument(
        "--skip-skill-index",
        action="store_true",
        help="Skip indexing ESCO skills even if missing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Candidate pool size for the interactive attachment helper.",
    )
    parser.add_argument(
        "--debug-elbow",
        type=str,
        default=None,
        help="Comma-separated scores to test the elbow cutoff and exit (e.g. '0.91,0.84,0.83,0.60,0.59').",
    )
    parser.add_argument(
        "--llm-debug",
        action="store_true",
        help="Verbose logging for LLM keyword batches.",
    )
    parser.add_argument(
        "--llm-batch-timeout",
        type=int,
        default=int(os.environ.get("LLM_BATCH_TIMEOUT", "120")),
        help="Seconds to wait per LLM batch before skipping (best-effort).",
    )
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=int(os.environ.get("LLM_RETRIES", "0")),
        help="Retries per batch after timeout/error.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=int(os.environ.get("LLM_BATCH_SIZE", "20")),
        help="Keywords per LLM call.",
    )
    args = parser.parse_args()

    global _llm_debug, _llm_batch_timeout, _llm_retries, _llm_batch_size
    _llm_debug = bool(args.llm_debug)
    _llm_batch_timeout = int(args.llm_batch_timeout)
    _llm_retries = int(args.llm_retries)
    _llm_batch_size = int(args.llm_batch_size)
    if _llm_debug:
        print(f"[LLM] Debug ON | batch_size={_llm_batch_size} timeout={_llm_batch_timeout}s retries={_llm_retries}")

    if args.debug_elbow:
        try:
            raw = [x.strip() for x in args.debug_elbow.split(",") if x.strip()]
            scores = [float(x) for x in raw]
        except ValueError:
            print("Could not parse --debug-elbow. Provide comma-separated numbers, e.g. 0.9,0.8,0.5")
            return
        n = _find_elbow_cutoff(scores)
        print(f"Input scores (desc sorted internally): {sorted([float(x) for x in scores], reverse=True)}")
        print(f"Computed keep_n = {n}")
        return

    emb = OllamaEmbedder(model=args.model)
    print(f"Using Ollama embedding model: {args.model}")

    # Build or load the VectorDB
    if args.load and not args.build:
        vdb = load_vector_db(emb, args.load)
        print("VectorDB loaded.")
    else:
        print("Importing dataset and building the vector database (category: 'total_CV')...")
        vdb, saved = import_recruitment_dataset_and_build_db(
            emb, out_dir=args.out, batch_size=64
        )
        print(f"\nDone. Vector database saved to: {saved}")
        print("Category created: ['total_CV']")

    # Ensure ESCO skills category exists (unless skipped), then save
    if not args.skip_skill_index:
        ensure_esco_skills_category(vdb, args.skills, category="esco_skill")
        save_dir = args.load or args.out
        vdb.save(save_dir)
        print(f"VectorDB saved to: {save_dir}")
        print("Categories now present:", vdb.list_categories())
        print(
            f"Counts — total_CV: {vdb.count('total_CV')}, esco_skill: {vdb.count('esco_skill')}"
        )
    else:
        print("Skipping ESCO skills indexing as requested.")

    # Interactive testing loop (same behavior as before, but works for both build/load modes)
    interactive_attach_loop(
        vdb, category_cv="total_CV", category_skill="esco_skill", top_k=args.top_k
    )

if __name__ == "__main__":
    main()