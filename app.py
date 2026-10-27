"""FinPilot - GenAI-Powered Financial Advisor. Run with: streamlit run app.py"""
import streamlit as st

import config
from src import advisor, dashboard, knowledge_base_page
from src.gemini_client import is_configured

st.set_page_config(page_title="FinPilot", page_icon="\U0001f4b9", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #faf8ff 0%, #f0fbff 55%, #fff6f0 100%);}
    .block-container {padding-top: 3.4rem; max-width: 1200px;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #ede9fe 0%, #e0f2fe 55%, #ccfbf1 100%);}
    [data-testid="stHeader"] {background: transparent;}

    /* Hero banner */
    .fp-hero {position: relative; overflow: hidden; color: #fff; border-radius: 22px;
        padding: 1.7rem 2.1rem; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 48%, #ec4899 100%);
        box-shadow: 0 14px 34px rgba(124, 58, 237, 0.30);}
    .fp-hero:before {content: ""; position: absolute; right: -70px; top: -90px; width: 260px; height: 260px;
        border-radius: 50%; background: rgba(255,255,255,0.14);}
    .fp-hero:after {content: ""; position: absolute; right: 120px; bottom: -110px; width: 200px; height: 200px;
        border-radius: 50%; background: rgba(255,255,255,0.10);}
    .fp-title {font-size: 2.4rem; font-weight: 800; margin: 0; line-height: 1.1; letter-spacing: -0.5px; position: relative;}
    .fp-sub {margin: 0.35rem 0 0 0; font-size: 1.05rem; opacity: 0.93; position: relative;}
    .fp-tags {margin-top: 0.9rem; position: relative;}
    .fp-tag {display: inline-block; font-size: 0.78rem; font-weight: 600; padding: 0.2rem 0.7rem; margin-right: 0.4rem;
        border-radius: 999px; background: rgba(255,255,255,0.20); border: 1px solid rgba(255,255,255,0.35);}

    /* Headings */
    .stApp h2 {font-weight: 800; background: linear-gradient(90deg, #4f46e5, #ec4899);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; width: fit-content;}
    .stApp h3 {font-weight: 700; border-left: 5px solid #7c3aed; padding-left: 0.7rem; margin-top: 1.8rem;}

    /* Gradient KPI cards (one colour per column) */
    [data-testid="stMetric"] {border-radius: 18px; padding: 1.1rem 1.3rem; color: #fff;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); box-shadow: 0 8px 22px rgba(99, 102, 241, 0.25);}
    [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%); box-shadow: 0 8px 22px rgba(244, 63, 94, 0.25);}
    [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d9488 0%, #34d399 100%); box-shadow: 0 8px 22px rgba(13, 148, 136, 0.25);}
    [data-testid="stMetric"] * {color: #ffffff !important;}
    [data-testid="stMetricLabel"] p {font-weight: 600; opacity: 0.92; letter-spacing: 0.2px;}
    [data-testid="stMetricValue"] {font-weight: 800;}

    /* Buttons */
    button[kind="primary"] {background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899); border: none; color: #fff;
        font-weight: 600; box-shadow: 0 6px 16px rgba(139, 92, 246, 0.35); transition: transform .15s ease;}
    button[kind="primary"]:hover {transform: translateY(-1px); filter: brightness(1.06); color: #fff;}
    button[kind="secondary"] {border-radius: 10px;}

    /* Panels */
    [data-testid="stExpander"] {background: rgba(255,255,255,0.75); border-radius: 14px; border: 1px solid #e9e5ff;}
    [data-testid="stFileUploaderDropzone"] {background: rgba(255,255,255,0.8); border: 2px dashed #c4b5fd; border-radius: 14px;}
    [data-testid="stPlotlyChart"] {background: rgba(255,255,255,0.85); border-radius: 18px; padding: 0.4rem;
        border: 1px solid #ece8ff; box-shadow: 0 6px 18px rgba(99, 102, 241, 0.08);}
    [data-baseweb="tab-list"] {gap: 0.4rem;}
    [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 0.4rem 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _header(page_title: str) -> None:
    st.markdown(
        """
        <div class="fp-hero">
          <p class="fp-title">FinPilot</p>
          <p class="fp-sub">GenAI-Powered Financial Advisor</p>
          <div class="fp-tags"><span class="fp-tag">Gemini</span><span class="fp-tag">RAG</span>
          <span class="fp-tag">FAISS</span><span class="fp-tag">Local-first</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header(page_title)


def dashboard_page() -> None:
    _header("Financial Dashboard")
    dashboard.render()


def advisor_page() -> None:
    _header("AI Financial Advisor")
    advisor.render()


def knowledge_base_page_view() -> None:
    _header("RAG Knowledge Base")
    knowledge_base_page.render()


pages = [
    st.Page(dashboard_page, title="Financial Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    st.Page(advisor_page, title="AI Financial Advisor", icon=":material/chat:", url_path="advisor"),
    st.Page(knowledge_base_page_view, title="RAG Knowledge Base", icon=":material/menu_book:", url_path="knowledge-base"),
]
navigation = st.navigation(pages)

with st.sidebar:
    st.divider()
    if is_configured():
        st.success(f"Gemini ready\n\nModel: {config.get_model_name()}", icon=":material/check_circle:")
    else:
        st.warning("Gemini API key not set. Dashboard and document search still work; AI answers are disabled.",
                   icon=":material/key_off:")
    with st.expander("Privacy"):
        st.caption(
            "Your CSV files, documents, embeddings and vector index stay on this computer. When you ask Gemini a "
            "question, your question, summary figures and/or retrieved text passages are sent to Google's Gemini API "
            "and processed by Google. FinPilot never connects to bank accounts. Sample data is fictional."
        )
    st.caption("Educational project. Not financial advice.")

navigation.run()
