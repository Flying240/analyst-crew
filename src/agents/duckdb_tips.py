"""Short DuckDB cheat sheet appended to SQL-writing prompts."""

DUCKDB_TIPS = """\
- Days between timestamps: date_diff('day', a, b). Use epoch(a - b) / 86400.0 only when the question asks for fractional days.
- Never use reserved words as aliases (do, order, group, desc, to); use short aliases like o, oi, c.
- Month labels: strftime(ts, '%Y-%m').
- Late = order_delivered_customer_date > order_estimated_delivery_date, compared as timestamps."""
