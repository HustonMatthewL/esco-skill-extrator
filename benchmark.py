"""
Complete example that:
1) Uses deepseek-r1:8b via ollama.chat
2) Reads your saved prompt from QuantCheck.prompt (or quantCheck.prompt)
3) Sends CV + keywords to the LLM
4) Parses the LLM response robustly and returns a dict {keyword: bool}
"""

import json
import os
from ollama import chat


def _load_prompt_text() -> str:
    """Load the saved labeling prompt from disk (case-robust)."""
    for fname in ("QuantCheck.prompt", "quantCheck.prompt"):
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError("Could not find 'QuantCheck.prompt' or 'quantCheck.prompt'.")


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_first_json_object(text: str):
    """
    Scan the text and return the first valid JSON object by balancing braces.
    This avoids issues if the model adds extra text before/after the JSON.
    """
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try the next candidate
                        pass
    raise ValueError("No valid JSON object found in model output.")


def getQuantTest(keywords, cv):
    """
    Run the quant check against the LLM and return a dict:
    { "keyword1": True/False, ... } for ALL provided keywords.
    """
    prompt_text = _load_prompt_text()

    # Ensure keywords are strings, deduped, and preserve order
    keywords = [str(k) for k in keywords]
    keywords_deduped = _dedupe_keep_order(keywords)

    # Build the user payload exactly as the prompt expects
    user_payload = f"""{prompt_text}

CV:
<<<
{cv}
>>>

KEYWORDS (JSON array):
{json.dumps(keywords_deduped, ensure_ascii=False)}
"""

    # Call the model
    resp = chat(
        model="deepseek-r1:1.5b",
        messages=[
            {
                "role": "system",
                "content": "Return ONLY a valid JSON object mapping each keyword to a boolean. No explanations."
            },
            {"role": "user", "content": user_payload},
        ],
        options={"temperature": 0},
    )

    raw = resp["message"]["content"]

    # Parse JSON from model output (robustly)
    try:
        data = _extract_first_json_object(raw)
    except Exception as e:
        # As a fallback, try to parse the whole thing (e.g., if it already is clean JSON)
        try:
            data = json.loads(raw.strip().strip("`"))
        except Exception:
            raise ValueError(f"LLM did not return valid JSON.\nRaw output:\n{raw}") from e

    # Normalize: keep ONLY the provided keywords and coerce to bools
    out = {}
    for k in keywords_deduped:
        v = data.get(k, False)
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = bool(v)
        elif isinstance(v, str):
            out[k] = v.strip().lower() in {"true", "t", "yes", "y", "1"}
        else:
            out[k] = False

    return out


# --- Example usage ---
if __name__ == "__main__":
    # quick sanity check of the model call


    # Example run (replace with your real CV and keywords)
    cv_text = "Senior Software Engineer with 6+ years working in Python and Java. Built REST APIs with Django and Spring Boot. Deployed on AWS."
    kws = ["Python", "Java", "C++", "AWS", "Azure"]

    result = getQuantTest(kws, cv_text)
    print(result)  # {'Python': True, 'Java': True, 'C++': False, 'AWS': True, 'Azure': False}