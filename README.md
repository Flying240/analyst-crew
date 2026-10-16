# analyst-crew

A multi-agent AI data analyst for the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), stored in DuckDB.
The planned agents are Planner, Data (SQL), Viz (Plotly), Critic and Reporter, orchestrated with LangGraph. The LLM is Groq or Gemini, chosen in `.env`.

## Progress

| # | Step | Status |
|---|------|--------|
| 0 | Foundation: scaffold, DuckDB loader, schema.md, read-only SQL tool, eval harness | ✅ Done |
| 1 | Single-agent baseline + eval score | ✅ Done |
| 2 | Multi-agent LangGraph (Planner, Data, Viz, Reporter) | ⬜ Next |
| 3 | Critic loop with max 3 retries | ⬜ |
| 4 | Safety: timeouts, row limits, sandboxed code | ⬜ |
| 5 | MCP server for the database | ⬜ |
| 6 | Streamlit UI + Langfuse tracing | ⬜ |
| 7 | Deploy | ⬜ |

### Eval setup (frozen 2026-10-15)

Every eval row from 2026-10-15 on uses this setup, which is the default in `src/config.py` and `.env.example`:

- Model: Groq `openai/gpt-oss-120b`, temperature 0
- Schema: `PROMPT_SCHEMA=compact` (`schema_compact.md`)
- Reasoning effort: `REASONING_EFFORT=low`

Why: about 900 tokens per question (≈680 prompt + ≈220 completion) against ≈4,000 with the full schema and default reasoning effort, mostly from fewer reasoning tokens. A 10-question dev check scored 7/10 both ways. That leaves room for 3 full 50-question runs inside Groq's free 200K tokens/day. Rows dated before 2026-10-15 used the full schema and default effort.

### Eval scores
| Date | Answerer | Score |
|------|----------|-------|
| 2026-10-14 | oracle (gold SQL, sanity check) | 10/10 |
| 2026-10-14 | baseline, Groq `openai/gpt-oss-120b` (before) | 9/10 — e06 failed: months returned as DATE, not `'YYYY-MM'` |
| 2026-10-14 | baseline + month-label prompt rule | 10/10 |
| 2026-10-14 | oracle, 40-question set | 40/40 |
| 2026-10-14 | **baseline, 40-question set (real "before")** | **32/40** — easy 9/10, medium 15/15, hard 8/15 |
| 2026-10-14 | oracle, after eval-set corrections below | 40/40 |
| 2026-10-14 | baseline, corrected 40-question set (same prompt/model) | 34/40 — easy 10/10, medium 14/15, hard 10/15 |
| 2026-10-14 | baseline, `--runs 3` (temperature 0) | mean 33.7/40, range 33–34 — easy 10/10, medium 13.7/15 (13–15), hard 10/15 (9–11); 8 questions flip between runs; one run-1 failure (h15) was a Groq 429 rate limit |
| 2026-10-14 | oracle, 50 questions (+10 silent-wrong-answer) with dev/heldout split | 50/50 — dev 35/35, heldout 15/15 |
| 2026-10-14 | baseline, 50 questions, full schema, default effort (early experiment, not the baseline) | 43/50, 1 run |
| 2026-10-15 | **baseline, 50 questions, frozen eval setup, `--runs 3` (official "before")** | **mean 85.3% (42.7/50), range 42–44** — easy 100%; medium 88.9% (18–19/21); hard 73.7% (13–15/19); dev 82.9% (28–30/35); heldout 91.1% (13–14/15). Failed every run: m11, m19, h02, h06, h10, h15. Flipped: m04, h04, h07, h18. ≈840 tokens/question, ≈126K for 3 runs. Details: `evals/baseline_50q_runs.json` |
| 2026-10-15 | *Spot check, not an official score:* baseline `--retries 2` (error-feedback retry), 5 always-failing dev questions only (`--ids`) | 2/5 — fixed h10, h15; still failing h06 (DB error after 2 retries), m11 and m19 (SQL ran but answer wrong, so no retry) |
| 2026-10-16 | **baseline + retry (`--retries 2`), 50 questions, frozen eval setup, 3 runs (official "after" for error-feedback retry)** | **mean 94.0% (47.0/50), range 46–48** — easy 100%; medium 90.5% (18–20/21); hard 94.7% (18/19 every run); dev 91.4% (31–33/35); heldout 100% (15/15). Failed every run: m19. Flipped: m10, m11, h05, h15. ≈1,010 tokens/question, ≈150K for 3 runs. Run 1 on 2026-10-15, runs 2–3 on 2026-10-16. Details: `evals/retry_50q_runs.json` |
| 2026-10-16 | *Spot check, not an official score:* 5 questions (`--ids` m11, m19, h15, m09, e10), `--retries 2`, 1 run each: baseline vs baseline + DuckDB tips vs multi-agent graph | baseline 3/5 (m11, m19 fail), ≈1,440 tokens/q; **+ tips 4/5** (m19 fails), ≈1,525 tokens/q; graph 3/5 (m11, e10 fail; m19 fixed by the Planner), ≈2,230 tokens/q. Graph failures come from the Planner: m11 checks for a missing translation instead of a NULL category, and e10 measures delivered minus estimated date instead of delivered minus purchase date |
| 2026-10-16 | *Spot check, not an official score:* graph after Planner fix (Planner on gpt-oss-120b, logic-only plans with project rules, Data told to follow the question over the plan), same 5 ids, `--retries 2` | **4/5** (was 3/5): e10 now passes; m19 still passes; m11 still fails (the Planner still left-joins to `category_translation` instead of checking `product_category_name IS NULL`). ≈2,315 tokens/q (was ≈2,230). Plans and attempts are now saved in each eval result under `trace` |

Every new feature must be measured with `evals/run_eval.py`. Add a row here after each run.

Heldout caveat: the 10 older heldout questions were seen during baseline work, so only the 5 new heldout questions (m16, m17, m20, h17, h19) are truly unseen.

### Eval set corrections

Reviewing the 8 baseline failures turned up three defects in the eval set itself, fixed in `evals/questions.json`. The oracle still scores 40/40 and no question's expected answer changed.

How far the 32/40 baseline stays comparable depends on the question:

- **38 of 40 questions** have byte-identical text, h04 among them — only its gold SQL changed, and it returns the same category as before, so it is fully comparable.
- **h09 and h08** had their question text reworded, so a later run may answer either one differently. That is partly the point: both rewrites state a scope the agent previously had to guess. h09 was one of the 8 failures and h08 was already passing.

Treat a change in the h09 or h08 result as a change in the question, not as evidence about the agent. A run-to-run comparison that needs to be airtight should be scored over the 38 unchanged questions.

- **h09_best_review_month** — the question asked for the best month by average review score without saying which orders to count, and the baseline filtered to `order_status = 'delivered'` while the gold SQL did not. CLAUDE.md's `delivered orders only` is scoped to the revenue definition, so it does not apply to a reviews question, but the question never said so. The non-delivered reviews it turns on are few (~2,800 of ~99,000 in the window) and harsh (averaging under 2.0 against 4.16 for delivered), and the top four months sit within 0.012 points of each other, so the filter choice decides the answer. The question now says **"across all orders, regardless of order status"**; the gold SQL is unchanged and still returns `2018-06`.
- **h04_drop_category** — the gold SQL compared each category's revenue in the worst month against its *previous row* via `LAG(rev) OVER (PARTITION BY cat ORDER BY month)`, which silently reaches past a month in which the category sold nothing, and a category with zero revenue in the worst month had no row at all, so the biggest possible drop was invisible. Rewritten to pin the comparison to the worst month and the calendar month before it, treating a missing month as 0 revenue. Both flaws are real at the current data point but too small to move the result: four categories vanish between 2017-11 and 2017-12, the largest worth R$259.94 against the winner's R$45,585.09 decline. Old and new SQL both return `bed_bath_table`, with an identical top eight.
- **h08_worst_late_state** — the question said "at least 1000 delivered orders" while the gold SQL applied `order_delivered_customer_date IS NOT NULL`, counting only delivered orders whose delivery date is known. The question now says **"at least 1000 delivered orders with a known delivery date"**; the gold SQL is unchanged.

## What's built (Step 0: Foundation)

- **Data pipeline**
  - `scripts/download_data.py` downloads the dataset from Kaggle into `data/`.
  - `scripts/build_db.py` loads all 9 CSVs into `db/olist.duckdb`. The tables are orders, order_items, customers, payments, reviews, products, sellers, geolocation and category_translation.
  - `scripts/gen_schema.py` writes `schema.md` with tables, columns, types, row counts, join keys and business rules. Agents will read this file.
- **Read-only SQL tool**, `src/tools/sql_tool.py`
  - `is_read_only(sql)` allows only a single `SELECT`/`WITH` statement. It blocks INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, COPY, ATTACH, PRAGMA, multiple statements, and similar tricks.
  - `run_sql(sql, limit=1000)` opens DuckDB with `read_only=True` as a second layer of protection. It returns a `SqlResult` with columns, rows, a truncated flag and any error.
- **Eval harness**, `evals/`
  - `questions.json` holds 40 questions (easy/medium/hard), each with a correct answer query and a check (single value or rows, with a tolerance).
  - `run_eval.py --answerer module:function` scores any `answer(question) -> sql` function and saves the results to `evals/results/` as `YYYYMMDD_HHMMSS_<answerer>.json` (e.g. `20261014_145208_oracle.json`, `20261014_012927_baseline.json`), where the suffix is the last segment of the answerer's module.
- **Tests**: `tests/test_sql_tool.py` has 21 SQL-safety tests, all passing.

## What's built (Step 1: Single-agent baseline)

- `src/agents/baseline.py`: `answer(question) -> sql` makes one LLM call at temperature 0. The prompt contains `schema.md` plus the project rules: read-only SQL, the revenue definition, `customer_unique_id`, excluded months, one value means one row, and `'YYYY-MM'` month labels.
- The provider is chosen by `LLM_PROVIDER` (`groq` or `gemini`), and the model by `GROQ_MODEL` or `GEMINI_MODEL`. There are no retries and no self-correction; this is the "before" number for the multi-agent steps.
- Run it with `python evals/run_eval.py --answerer src.agents.baseline:answer`.
- Gemini's free tier (5 requests/min) hits rate limits during a full eval run, so Groq is the default.
- The eval set has 40 questions, each with a `difficulty` field (10 easy, 15 medium, 15 hard), and `run_eval.py` reports scores per difficulty.

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
python evals/run_eval.py          # oracle should score 40/40
```

## Project structure
```
data/            raw CSVs (gitignored)
db/              olist.duckdb (gitignored)
scripts/         download, build DB, generate schema
src/config.py    paths + env settings
src/tools/       sql_tool.py (read-only SQL)
src/agents/      baseline.py (single-agent SQL)
evals/           questions.json, run_eval.py, results/
tests/           pytest suite
schema.md        generated schema doc for agents
CLAUDE.md        project rules + roadmap
```
