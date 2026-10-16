"""Streamlit entry point: `streamlit run app.py`."""
import streamlit as st

st.set_page_config(page_title="Analyst crew", page_icon=":material/query_stats:", layout="wide")

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
