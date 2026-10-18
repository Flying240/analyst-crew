"""Short DuckDB cheat sheet appended to SQL-writing prompts."""

DUCKDB_TIPS = """\
- Days between timestamps: date_diff('day', a, b). Use epoch(a - b) / 86400.0 only when the question asks for fractional days.
- Never use reserved words as aliases (do, order, group, desc, to); use short aliases like o, oi, c.
- Filter date ranges on raw timestamps (ts >= '2017-01-01' AND ts < '2017-09-01'); use strftime only for output labels and GROUP BY.
- For previous/next month use LAG/LEAD over the full series before filtering, or ts - INTERVAL 1 MONTH; DuckDB has no add_months.
- Always parenthesize OR conditions mixed with AND.
- Late = order_delivered_customer_date > order_estimated_delivery_date, compared as timestamps."""
