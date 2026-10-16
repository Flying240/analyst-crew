"""Footer shown under every page. Fill in the placeholders."""
import streamlit as st

AUTHOR = "<your name>"
GITHUB_URL = "https://github.com/<your-github>"
LINKEDIN_URL = "https://www.linkedin.com/in/<your-linkedin>"


def footer_text() -> str:
    return f"Built by {AUTHOR} · [GitHub]({GITHUB_URL}) · [LinkedIn]({LINKEDIN_URL})"


def render() -> None:
    st.divider()
    st.caption(footer_text())
