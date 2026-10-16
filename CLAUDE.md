# analyst-crew

Multi-agent AI data analyst over the Olist e-commerce dataset (DuckDB).
Agents: Planner, Data (SQL), Viz (Plotly), Critic, Reporter. Orchestrated with LangGraph.

## Rules
- Python 3.11, type hints, small functions
- All SQL must be read-only; block INSERT/UPDATE/DELETE/DROP/ALTER
- LLM: Gemini or Groq via env vars in .env (never hardcode keys)
- Revenue = price + freight_value from order_items, delivered orders only
- Count customers with customer_unique_id, not customer_id
- Exclude 2016 and Sept–Oct 2018 from trend analysis (incomplete months)
- Late = order_delivered_customer_date > order_estimated_delivery_date (timestamp comparison, not calendar date)
- Every new feature must be measurable with evals/run_eval.py
- Eval scalar check is strict: agent prompts must say "For questions asking for one value, return exactly one row with one numeric column" (text labels OK)

## Roadmap
- [x] 0. Foundation: scaffold, DuckDB loader, schema.md, read-only SQL tool, eval harness
- [x] 1. Single-agent baseline + eval score
- [ ] 2. Multi-agent LangGraph (Planner, Data, Viz, Reporter)
- [ ] 3. Critic loop with max 3 retries
- [ ] 4. Safety: timeouts, row limits, sandboxed code
- [ ] 5. MCP server for the database
- [ ] 6. Streamlit UI + Langfuse tracing
- [ ] 7. Deploy (must get db/olist.duckdb onto the host; it is gitignored)

## Commands
- Setup: `pip install -r requirements.txt`, copy `.env.example` to `.env`
- Data: `python scripts/download_data.py` → `python scripts/build_db.py` → `python scripts/gen_schema.py`
- Tests: `pytest`
- Eval: `python evals/run_eval.py [--answerer module:function]` (default `oracle` = gold SQL)

## Structure
data/ raw CSVs (gitignored) | db/ olist.duckdb | src/agents/ | src/tools/
src/graph.py | evals/ | app.py + app_pages/ | examples/ (committed eval traces) | .cache/ (live runs, token ledger)