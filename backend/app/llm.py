"""Shared LLM + knowledge-base helpers for prediction-chat (Change 2).

Keys stay on the backend. Callers must supply grounded context; this module
does not invent coping classes or survey figures.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "knowledge_base"


def _load_dotenv_once() -> None:
    """Load project .env into process env if present (does not log secrets)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            # Always apply project LLM_* from .env so shell OPENAI_* cannot win.
            if key.startswith("LLM_") or key not in os.environ:
                os.environ[key] = val
    except OSError:
        return


_load_dotenv_once()

TARGETING_PATTERNS = re.compile(
    r"\b("
    r"who should (get|receive|be)|"
    r"target(ed|ing)?|"
    r"eligib|"
    r"rank (households|people)|"
    r"why did (this|that) household"
    r")\b",
    re.I,
)


def llm_configured() -> bool:
    return bool(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def load_kb_chunks() -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    if not KB.exists():
        return chunks
    for path in sorted(KB.glob("*.md")):
        chunks.append((path.name, path.read_text(encoding="utf-8")))
    return chunks


def retrieve(query: str, k: int = 4) -> list[tuple[str, str]]:
    q = query.lower()
    scored: list[tuple[int, str, str]] = []
    for name, text in load_kb_chunks():
        score = sum(1 for tok in re.findall(r"[a-z0-9]+", q) if tok in text.lower())
        scored.append((score, name, text))
    scored.sort(reverse=True)
    hits = [(n, t[:1400]) for s, n, t in scored[:k] if s > 0]
    if hits:
        return hits
    return [(n, t[:1400]) for _, n, t in scored[:k]]


def _extract_assistant_text(message: dict[str, Any]) -> str:
    """Prefer visible content; ignore model reasoning channels for user-facing text."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    # Some SDKs return content as a list of parts
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                parts.append(str(part.get("text") or ""))
        joined = "".join(parts).strip()
        if joined:
            return joined
    # Do not surface raw "reasoning" to users — it is internal and may invent detail.
    raise RuntimeError("LLM returned empty assistant content (reasoning-only or truncated)")


def call_chat_completion(
    *,
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout: float = 60.0,
    max_tokens: int = 2048,
) -> str:
    """Call OpenAI-compatible chat completions (Groq via LLM_API_*). Raises on failure."""
    # Prefer LLM_API_KEY so a stale OPENAI_API_KEY in the shell cannot override .env.
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("LLM API key not configured")

    import httpx

    base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")
    # Prefer max_completion_tokens (Groq). Reasoning models can exhaust the budget on
    # hidden tokens and return empty content — tune effort per family.
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    ml = model.lower()
    if "gpt-oss" in ml:
        payload["reasoning_effort"] = "low"
    elif "qwen3" in ml or "qwen/qwen3" in ml:
        # Free Groq Qwen 3.8: disable thinking so replies land in content.
        payload["reasoning_effort"] = "none"

    def _once(body: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"LLM HTTP {response.status_code}: {response.text[:300]}")
        data: dict[str, Any] = response.json()
        choice = data["choices"][0]
        finish = choice.get("finish_reason")
        msg = choice.get("message") or {}
        try:
            text = _extract_assistant_text(msg)
        except RuntimeError:
            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            raise RuntimeError(
                "LLM returned empty assistant content "
                f"(finish_reason={finish}, reasoning_tokens={details.get('reasoning_tokens')})"
            ) from None
        return text, finish, data

    try:
        text, finish, _ = _once(payload)
    except RuntimeError as exc:
        if "empty assistant content" not in str(exc):
            raise
        # One retry with more budget and the strongest low-reasoning setting.
        retry = dict(payload)
        retry["max_completion_tokens"] = max(max_tokens, 3072)
        if "gpt-oss" in model.lower():
            retry["reasoning_effort"] = "low"
        retry["messages"] = [
            {
                "role": "system",
                "content": system
                + " Put the entire user-facing answer in the normal content field; keep reasoning brief.",
            },
            {"role": "user", "content": user},
        ]
        text, finish, _ = _once(retry)

    if finish == "length":
        raise RuntimeError("LLM reply truncated by token limit (finish_reason=length)")
    return text
