"""Streamlit entry point: `streamlit run app.py`."""
import streamlit as st

from src.ui import footer

st.set_page_config(page_title="Analyst crew", page_icon=":material/query_stats:", layout="wide")


@st.cache_resource(show_spinner="Building the database from Parquet (first start only)…")
def database_ready() -> bool:
    from src.db_bootstrap import ensure_db

    ensure_db()
    return True


database_ready()

page = st.navigation(
    [
        st.Page("app_pages/ask.py", title="Ask", icon=":material/chat:", default=True),
        st.Page("app_pages/about.py", title="About", icon=":material/info:"),
        st.Page("app_pages/eval_dashboard.py", title="Eval dashboard", icon=":material/monitoring:"),
        st.Page("app_pages/schema_explorer.py", title="Schema explorer", icon=":material/table:"),
    ],
    position="top",
)
st.title(page.title, anchor=False)
page.run()
footer.render()
