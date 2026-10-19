"""Shared LLM calls (Groq or Gemini) used by every agent."""
import os
import re
from typing import Mapping

from src import llm_usage, usage_ledger
from src.config import LLM_PROVIDER, REASONING_EFFORT
from src.llm_retry import with_backoff


def track(model: str, prompt_tokens: int, completion_tokens: int, headers: Mapping[str, str] | None = None) -> None:
    """Record usage for the current question and in the local daily ledger."""
    llm_usage.record(prompt_tokens, completion_tokens, model)
    usage_ledger.add(model, prompt_tokens, completion_tokens, headers)


def call_groq(
    prompt: str, model: str | None = None, max_tokens: int | None = None, reasoning_effort: str | None = None,
) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    effort = reasoning_effort or REASONING_EFFORT
    extra = {"reasoning_effort": effort} if effort else {}
    if max_tokens:
        extra["max_completion_tokens"] = max_tokens
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # with_raw_response returns the rate-limit headers of this same call (no extra request).
    raw = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        **extra,
    )
    resp = raw.parse()
    if resp.usage:
        track(model, resp.usage.prompt_tokens, resp.usage.completion_tokens, raw.headers)
    return resp.choices[0].message.content or ""


def call_gemini(prompt: str, model: str | None = None, **_: object) -> str:
    """Gemini ignores per-agent Groq model names and always uses GEMINI_MODEL."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    meta = resp.usage_metadata
    if meta:
        track(model, meta.prompt_token_count, (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
    return resp.text or ""


def complete(prompt: str, model: str | None = None, **options: object) -> str:
    """Return the model's raw text reply, retrying on rate limits.

    Options (Groq only): max_tokens, reasoning_effort (overrides REASONING_EFFORT for this call).
    """
    call = call_gemini if LLM_PROVIDER == "gemini" else call_groq
    return with_backoff(lambda: call(prompt, model, **options))


class NotSqlError(ValueError):
    """The model replied without a SELECT/WITH query; `text` is what it said."""

    def __init__(self, text: str):
        super().__init__("Return only a single SQL query, no prose")
        self.text = text


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = (match.group(1) if match else text).strip().rstrip(";").strip()
    if not re.search(r"\b(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise NotSqlError(sql)
    return sql


def ask(prompt: str, model: str | None = None) -> str:
    """Return SQL extracted from the model's reply."""
    return extract_sql(complete(prompt, model))
