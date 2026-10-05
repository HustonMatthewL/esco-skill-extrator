# ESCO Keyword Extraction (FAISS + Ollama)

This repository is a small, self‑contained pipeline to **extract ESCO skill keywords** from free text (e.g., a CV or job description). The focus is on **retrieving the most relevant ESCO skills** via fast vector search and then (optionally) **validating** those skills with a local LLM.

> TL;DR: We embed ESCO skills → build a FAISS index → embed a CV → retrieve top‑k skills → (optionally) validate them with an LLM prompt.

---

## What’s here

- `FAISS.py` – Vector DB helpers (FAISS-based) + a simple **Ollama** embedder wrapper.
- `test.py` – End‑to‑end **ESCO keyword attachment** flow (build/load index, interactive attach loop).
- `benchmark.py` – **LLM validation** of keywords via `ollama.chat`, reading your labeling instructions from `QuantCheck.prompt` (or `quantCheck.prompt`). Returns a dict `{keyword: true/false}`.

> **Note:** The code intentionally keeps the ESCO flow simple: retrieval first, then optional LLM validation. You can run retrieval alone if you don’t want LLMs involved.

---

## Why ESCO?

[ESCO](https://ec.europa.eu/esco/) is the European multilingual classification of Skills, Competences, and Occupations. Using a shared taxonomy makes downstream tasks (CV parsing, matching, analytics) consistent and comparable across documents and languages.

---

## Pipeline overview (ESCO-focused)

1. **Prepare ESCO skills**
   - Download ESCO skills in your preferred language (CSV/JSON). Make a flat CSV with at least:
     - `id` (ESCO skill identifier)
     - `label` (preferred label / skill name)
     - (optional) `description`
   - Put it at `data/esco_skills.csv` (or wherever you like).

2. **Embed ESCO skills**
   - `FAISS.py` includes an `OllamaEmbedder` (uses `/api/embed`).
   - Run the **build step** in `test.py` to embed all ESCO rows and store them in a FAISS index.

3. **Attach skills to text (CV or JD)**
   - Embed the input text, run **top‑k similarity search** in FAISS, and **attach** the most relevant ESCO skills.

4. **(Optional) Validate skills with an LLM**
   - `benchmark.py` sends `{CV text + candidate skills}` to a local chat model via `ollama.chat` and expects a clean JSON verdict `{skill->true/false}`.
   - This helps remove false positives from the raw retrieval step.

---

## Requirements

- **Python** ≥ 3.9
- **Ollama** running locally (default `http://localhost:11434`)
  - **Embedding model** (choose one that supports `/api/embed`), e.g.:
    - `nomic-embed-text` (good default)
    - `bge-m3`, `mxbai-embed-large`, etc.
  - **Chat model** for validation (optional):
    - `deepseek-r1:8b` (as used in `benchmark.py`) or any capable model you prefer.
- Python packages (install with pip):
  ```bash
  pip install numpy faiss-cpu requests ollama pandas tqdm
  ```
  *(If you have CUDA and want GPU FAISS, use `faiss-gpu` instead of `faiss-cpu`.)*

> Make sure to `ollama pull` the models you intend to use, e.g. `ollama pull nomic-embed-text` and `ollama pull deepseek-r1:8b`.

---

## Quick start (ESCO retrieval)

### 1) Build the ESCO index
Use `test.py` to embed ESCO skills and persist a FAISS index. The script supports both **build** and **load** modes. Run the help to see the exact flags:

```bash
python test.py --help
```

Typical workflow (illustrative):
```bash
# Build the FAISS index from an ESCO CSV and save to a directory
python test.py \
  --esco data/esco_skills.csv \
  --build \
  --save-dir .vector_store \
  --top-k 20
```

What this does:
- Reads ESCO rows (`id`, `label`, optional `description`).
- Creates text to embed (usually `label` + maybe `description`).
- Embeds all skills with the Ollama embedder.
- Stores vectors + metadata in a FAISS index under `.vector_store`.

### 2) Attach ESCO skills to a CV (interactive)
Once the index exists, load it and start the interactive loop:
```bash
python test.py \
  --load-dir .vector_store \
  --top-k 20
```
Paste a CV or job text when prompted; you’ll get the **top‑k ESCO skills** ranked by similarity. This is the core **ESCO keyword extraction** step.

---

## Optional: LLM validation of skills

After retrieval, you can validate with a local LLM to reduce false positives.

1. Create a file named `QuantCheck.prompt` (or `quantCheck.prompt`) in the project root with your instructions. Example scaffold:

   ```text
   You are labeling whether a CV truly evidences specific skills.

   Return a single JSON object: 
   {
     "<skill>": true|false,
     ...
   }

   Label true only if the CV contains strong evidence (explicit mention or clearly implied experience). 
   Be strict.
   ```

2. Call the validator from Python or run `benchmark.py` directly.

   **Python API:**
   ```python
   from benchmark import getQuantTest

   cv_text = "Senior Software Engineer ... REST APIs with Django and Spring Boot. Deployed on AWS."
   kws = ["Python", "Java", "C++", "AWS", "Azure"]

   verdicts = getQuantTest(kws, cv_text)
   # e.g., {'Python': True, 'Java': True, 'C++': False, 'AWS': True, 'Azure': False}
   ```

   **CLI sanity check:**
   ```bash
   python benchmark.py
   ```

`test.py` integrates this idea as well: it first retrieves candidate ESCO skills, then (optionally) batches them through the validator and **keeps only the skills labeled `true`**.

---

## Configuration notes

- **Models**: `FAISS.py` looks for an embedding model; set the model name via code or environment variables as needed.
- **Ollama host**: The embedder uses `OLLAMA_HOST` if present, otherwise `http://localhost:11434`.
- **Batching**: For large ESCO sets (10k+ skills), embedding and FAISS indexing can take time; use batching and persist to disk.
- **Multilingual**: ESCO is multilingual; you may build one index per language or mix labels + descriptions depending on your use case.

---

## Outputs

- **Retrieval**: A ranked list of ESCO skills with similarity scores (top‑k).
- **Validation (optional)**: A boolean mask `{skill->true/false}`, used to filter to only **validated** ESCO skills.

These can be fed into downstream systems (profiling, matching, analytics, dashboards).

---

## Troubleshooting

- *I get empty results.* Make sure your ESCO CSV has useful text (skill labels, optional descriptions) and you actually built/loaded the index.
- *Embeddings fail.* Confirm your chosen Ollama model supports `/api/embed`. Many chat‑only models do **not**.
- *Validation JSON parsing fails.* Tighten your `QuantCheck.prompt` so the model outputs a **single JSON object** and nothing else.
- *Performance issues.* Consider smaller `top-k` or a faster embedder; persist the FAISS index so you don’t rebuild each run.

---

## File map

```
FAISS.py       # FAISS vector store + Ollama embedding client
test.py        # Build/load index + interactive ESCO attach loop (+ optional LLM validation)
benchmark.py   # Keyword validation via ollama.chat reading QuantCheck.prompt
```

---

## License

MIT.
