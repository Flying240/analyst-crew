"""Footer shown under every page."""
import streamlit as st

AUTHOR = "D R Vikas"
GITHUB_URL = "https://github.com/Flying240"
LINKEDIN_URL = "https://www.linkedin.com/in/d-r-vikas-b1605b292/"


def footer_text() -> str:
    return f"Built by {AUTHOR} · [GitHub]({GITHUB_URL}) · [LinkedIn]({LINKEDIN_URL})"


def render() -> None:
    st.divider()
    st.caption(footer_text())
