# analyst-crew

A multi-agent AI data analyst for the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), stored in DuckDB.
The planned agents are Planner, Data (SQL), Viz (Plotly), Critic and Reporter, orchestrated with LangGraph. The LLM is Groq or Gemini, chosen in `.env`.

## Progress

| # | Step | Status |
|---|------|--------|
| 0 | Foundation: scaffold, DuckDB loader, schema.md, read-only SQL tool, eval harness | ✅ Done |
| 1 | Single-agent baseline + eval score | ⬜ Next |
| 2 | Multi-agent LangGraph (Planner, Data, Viz, Reporter) | ⬜ |
| 3 | Critic loop with max 3 retries | ⬜ |
| 4 | Safety: timeouts, row limits, sandboxed code | ⬜ |
| 5 | MCP server for the database | ⬜ |
| 6 | Streamlit UI + Langfuse tracing | ⬜ |
| 7 | Deploy | ⬜ |

### Eval scores
| Date | Answerer | Score |
|------|----------|-------|
| _pending_ | oracle (gold SQL, sanity check) | expected 10/10 |

Every new feature must be measured with `evals/run_eval.py`. Add a row here after each run.

## What's built (Step 0: Foundation)

- **Data pipeline**
  - `scripts/download_data.py` downloads the dataset from Kaggle into `data/`.
  - `scripts/build_db.py` loads all 9 CSVs into `db/olist.duckdb`. The tables are orders, order_items, customers, payments, reviews, products, sellers, geolocation and category_translation.
  - `scripts/gen_schema.py` writes `schema.md` with tables, columns, types, row counts, join keys and business rules. Agents will read this file.
- **Read-only SQL tool**, `src/tools/sql_tool.py`
  - `is_read_only(sql)` allows only a single `SELECT`/`WITH` statement. It blocks INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, ATTACH, PRAGMA, multiple statements, and similar tricks.
  - `run_sql(sql, limit=1000)` opens DuckDB with `read_only=True` as a second layer of protection. It returns a `SqlResult` with columns, rows, a truncated flag and any error.
- **Eval harness**, `evals/`
  - `questions.json` holds 10 questions, each with a correct answer query and a check (single value or rows, with a tolerance).
  - `run_eval.py --answerer module:function` scores any `answer(question) -> sql` function and saves the results to `evals/results/`.
- **Tests**: `tests/test_sql_tool.py` has 21 SQL-safety tests, all passing.

## Business rules
- Revenue = `price + freight_value` from `order_items`, delivered orders only.
- Count customers with `customer_unique_id`, not `customer_id`.
- Trends exclude 2016 and Sept–Oct 2018 (incomplete months).

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # then add your API keys to .env

# Data (needs %USERPROFILE%\.kaggle\kaggle.json)
python scripts/download_data.py
python scripts/build_db.py
python scripts/gen_schema.py

# Check
pytest
python evals/run_eval.py          # oracle should score 10/10
```

## Project structure
```
data/            raw CSVs (gitignored)
db/              olist.duckdb (gitignored)
scripts/         download, build DB, generate schema
src/config.py    paths + env settings
src/tools/       sql_tool.py (read-only SQL)
src/agents/      (step 1+)
evals/           questions.json, run_eval.py, results/
tests/           pytest suite
schema.md        generated schema doc for agents
CLAUDE.md        project rules + roadmap
```
