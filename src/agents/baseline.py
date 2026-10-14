"""Single-agent baseline: one LLM call turns a question into DuckDB SQL."""
import os
import re

from src import llm_usage
from src.config import LLM_PROVIDER, PROMPT_SCHEMA, SCHEMA_COMPACT_PATH, SCHEMA_PATH
from src.llm_retry import with_backoff

RULES = """\
- Write a single read-only DuckDB SELECT (or WITH) query. Never INSERT/UPDATE/DELETE/DROP/ALTER.
- Revenue = price + freight_value from order_items, delivered orders only (order_status = 'delivered').
- Count customers with customer_unique_id, not customer_id.
- Exclude 2016 and Sept-Oct 2018 from trend analysis (incomplete months).
- Label months as text 'YYYY-MM' using strftime(ts, '%Y-%m'); label years as integers.
- For questions asking for one value, return exactly one row with one numeric column (text labels OK).
- Return only the SQL, no explanation, no markdown fences."""


def build_prompt(question: str) -> str:
    path = SCHEMA_PATH if PROMPT_SCHEMA == "full" else SCHEMA_COMPACT_PATH
    schema = path.read_text(encoding="utf-8")
    return (
        "You are a data analyst writing SQL for the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{schema}\n\n# Rules\n{RULES}\n\n# Question\n{question}\n\nSQL:"
    )


def call_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    if resp.usage:
        llm_usage.record(resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return resp.choices[0].message.content or ""


def call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    meta = resp.usage_metadata
    if meta:
        llm_usage.record(meta.prompt_token_count, (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
    return resp.text or ""


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip().rstrip(";").strip()


def answer(question: str) -> str:
    call = call_gemini if LLM_PROVIDER == "gemini" else call_groq
    prompt = build_prompt(question)
    return extract_sql(with_backoff(lambda: call(prompt)))
