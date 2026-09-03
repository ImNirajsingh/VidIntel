import html
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summerized import summerizes, generate_title
from core.extractor import extract_action_item, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="VidMind AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

:root {
    --cream: #f7f4ee;
    --cream-2: #f1ece4;
    --white: #ffffff;
    --navy: #111827;
    --navy-2: #273244;
    --muted: #667085;
    --soft: #8b93a1;
    --border: #e3ded5;
    --border-strong: #d4cec4;
    --peach: #efd9cf;
    --peach-2: #f4e6df;
    --green-bg: #e6f0e6;
    --lavender-bg: #ebe6f3;
    --amber-bg: #f4ead5;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif !important;
    color: var(--navy) !important;
}

[data-testid="stAppViewContainer"],
.main,
.stApp {
    background:
        radial-gradient(700px 360px at 50% -180px, rgba(239,217,207,.55), transparent 70%),
        linear-gradient(180deg, #fbfaf8 0%, var(--cream) 100%) !important;
    color: var(--navy) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

.block-container {
    max-width: 1160px !important;
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
}

h1, h2, h3, h4, p, label, span, div {
    color: inherit;
}

h1, h2, h3, h4 {
    font-family: "DM Sans", sans-serif !important;
    color: var(--navy) !important;
    letter-spacing: -.035em;
}

.topbar {
    min-height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,.92);
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 9px 12px 9px 14px;
    margin-bottom: 55px;
    box-shadow: 0 10px 30px rgba(17,24,39,.055);
}

.brand {
    display: flex;
    align-items: center;
    gap: 9px;
    font-weight: 700;
    font-size: 18px;
    color: var(--navy) !important;
}

.brandmark {
    width: 33px;
    height: 33px;
    border-radius: 10px;
    display: grid;
    place-items: center;
    background: var(--peach);
    color: var(--navy) !important;
    font-weight: 700;
}

.brand-ai {
    font-size: 9px;
    color: #9a8278 !important;
    align-self: flex-start;
    margin-top: 2px;
    letter-spacing: .08em;
}

.top-status {
    color: #7b8491 !important;
    font-size: 11px;
    display: flex;
    gap: 8px;
    align-items: center;
}

.live-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #78a878;
}

.hero {
    text-align: center;
    max-width: 900px;
    margin: 0 auto 52px;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 10px;
    letter-spacing: .18em;
    font-weight: 700;
    color: #8c766d !important;
    background: var(--peach-2);
    border: 1px solid #e7d4ca;
    border-radius: 999px;
    padding: 7px 12px;
}

.hero h1 {
    font-family: "Playfair Display", Georgia, serif !important;
    font-size: clamp(44px, 5.8vw, 68px) !important;
    line-height: 1.12 !important;
    font-weight: 500 !important;
    letter-spacing: -2px;
    margin: 19px 0 18px !important;
    color: var(--navy) !important;
}

.hero h1 span {
    background: var(--peach);
    color: var(--navy) !important;
    padding: 0 11px 3px;
    border-radius: 12px;
    box-decoration-break: clone;
    -webkit-box-decoration-break: clone;
}

.hero-copy {
    color: var(--muted) !important;
    font-size: 16px;
    line-height: 1.7;
    max-width: 670px;
    margin: 0 auto 28px;
}

.feature-pills {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 16px 0 0;
}

.pill {
    border: 1px solid var(--border);
    background: rgba(255,255,255,.72);
    color: #667085 !important;
    border-radius: 999px;
    padding: 7px 11px;
    font-size: 11px;
    box-shadow: 0 4px 14px rgba(17,24,39,.025);
}

.pill b { color: var(--navy) !important; font-weight: 600; }

.upload-shell {
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 25px;
    background: rgba(255,255,255,.66);
    box-shadow: 0 20px 65px rgba(17,24,39,.07);
}

.upload-zone {
    border: 1.5px dashed #cfc4ba;
    border-radius: 18px;
    padding: 29px 25px;
    background:
        radial-gradient(circle at 50% 0, rgba(239,217,207,.30), transparent 65%),
        #fffdfa;
}

.upload-top { display: flex; align-items: center; gap: 15px; }

.upload-icon {
    width: 47px;
    height: 47px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: var(--peach-2);
    border: 1px solid #e5d1c7;
    color: #806b63 !important;
    font-size: 21px;
}

.upload-zone h3 {
    margin: 0 0 3px !important;
    font-size: 17px !important;
    color: var(--navy) !important;
}

.upload-zone p { margin: 0; color: #7b8491 !important; font-size: 12px; }
.upload-hint { color: #969da7 !important; font-size: 10px; margin-top: 15px; }

.divider {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #9b948c !important;
    font-size: 9px;
    letter-spacing: .18em;
    margin: 15px 4px;
}

.divider:before, .divider:after {
    content: "";
    height: 1px;
    background: var(--border);
    flex: 1;
}

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea {
    background: #fff !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--navy) !important;
    border-radius: 12px !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea textarea::placeholder {
    color: #a0a6af !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #b99b8e !important;
    box-shadow: 0 0 0 3px rgba(239,217,207,.35) !important;
}

.stSelectbox [data-baseweb="select"] > div {
    background: #fff !important;
    color: var(--navy) !important;
    border-color: var(--border-strong) !important;
    border-radius: 12px !important;
}

.stSelectbox [data-baseweb="select"] span {
    color: var(--navy) !important;
}

label {
    color: #667085 !important;
    font-size: 11px !important;
}

.stButton > button {
    min-height: 45px;
    border-radius: 12px !important;
    border: 1px solid var(--border-strong) !important;
    background: #fff !important;
    color: var(--navy) !important;
    font-weight: 600 !important;
    transition: all .18s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #b8aaa0 !important;
    box-shadow: 0 9px 22px rgba(17,24,39,.08);
}

.primary > div > button {
    background: var(--navy) !important;
    color: #fff !important;
    border: 0 !important;
    box-shadow: 0 10px 24px rgba(17,24,39,.14) !important;
}

.primary > div > button:hover {
    background: #273244 !important;
}

.section-label {
    font-size: 10px;
    letter-spacing: .16em;
    color: #918981 !important;
    font-weight: 700;
    margin-bottom: 10px;
}

.session-title {
    font: 700 38px/1.08 "DM Sans";
    color: var(--navy) !important;
}

.session-meta { color: #7c8490 !important; font-size: 12px; margin-top: 7px; }

.card {
    background: rgba(255,255,255,.88);
    border: 1px solid var(--border);
    border-radius: 19px;
    padding: 23px;
    height: 100%;
    box-shadow: 0 12px 38px rgba(17,24,39,.055);
}

.card:hover { border-color: #d2c9c0; }

.card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.card-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .15em;
    color: #8a918c !important;
}

.card-icon {
    width: 30px;
    height: 30px;
    border-radius: 9px;
    display: grid;
    place-items: center;
    background: var(--peach-2);
    color: #806b63 !important;
    font-size: 13px;
}

.card-text {
    color: #475467 !important;
    font-size: 13px;
    line-height: 1.78;
    white-space: pre-wrap;
}

.muted {
    color: var(--muted) !important;
    font-size: 13px;
    line-height: 1.65;
}

.kicker {
    color: #939a9f !important;
    font-size: 10px;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.insight-card { min-height: 215px; }

.insight-icon {
    width: 36px;
    height: 36px;
    border-radius: 11px;
    display: grid;
    place-items: center;
    margin-bottom: 18px;
    background: var(--peach-2);
}

.green { color: #52755b !important; }
.purple { color: #76658d !important; }
.amber { color: #967542 !important; }

.chat-wrap {
    border: 1px solid var(--border);
    border-radius: 20px;
    overflow: hidden;
    background: rgba(255,255,255,.82);
    box-shadow: 0 15px 50px rgba(17,24,39,.06);
}

.chat-header {
    padding: 17px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.ai-orb {
    width: 34px;
    height: 34px;
    border-radius: 11px;
    display: grid;
    place-items: center;
    background: var(--peach);
    color: var(--navy) !important;
    margin-right: 10px;
}

.chat-title { font-weight: 700; font-size: 13px; color: var(--navy) !important; }
.chat-sub { font-size: 10px; color: #8b939e !important; margin-top: 2px; }
.chat-ready { color: #63806a !important; font-size: 10px; }

.chat-area { padding: 20px; }

.chat-user {
    max-width: 72%;
    margin: 9px 0 9px auto;
    padding: 11px 14px;
    border-radius: 14px 14px 4px 14px;
    background: var(--navy);
    font-size: 12px;
    line-height: 1.6;
    color: #fff !important;
}

.chat-ai {
    max-width: 78%;
    margin: 9px 0;
    padding: 11px 14px;
    border-radius: 14px 14px 14px 4px;
    background: #f3f0eb;
    border: 1px solid var(--border);
    font-size: 12px;
    line-height: 1.65;
    color: #475467 !important;
}

.chat-suggestions { display: flex; gap: 7px; flex-wrap: wrap; margin: 6px 0 15px; }
.chat-suggestion { color: #667085 !important; font-size: 10px; border: 1px solid var(--border); border-radius: 999px; padding: 7px 10px; background: #fff; }

.processing-card {
    max-width: 700px;
    margin: 50px auto;
    padding: 32px;
    border: 1px solid var(--border);
    border-radius: 20px;
    background: rgba(255,255,255,.9);
    text-align: center;
    box-shadow: 0 15px 50px rgba(17,24,39,.06);
}

.spinner {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    border: 3px solid #e8e3dc;
    border-top-color: #a4877c;
    animation: spin 1s linear infinite;
    margin: 0 auto 17px;
}

@keyframes spin { to { transform: rotate(360deg); } }

.progress-track {
    height: 4px;
    border-radius: 9px;
    background: #e7e2db;
    overflow: hidden;
    margin: 20px 0 10px;
}

.progress-fill {
    height: 100%;
    width: 60%;
    background: linear-gradient(90deg, #c6a99c, #e4c8bd);
    animation: progress 5s ease-in-out infinite;
}

@keyframes progress {
    0% { width: 5%; }
    50% { width: 68%; }
    100% { width: 93%; }
}


/* ---------------------------------------------------------
   VIDMIND INTERACTIVE VIDEO CHAT
--------------------------------------------------------- */

.chat-panel {
    background: rgba(255,255,255,.94);
    border: 1px solid var(--border);
    border-radius: 22px;
    box-shadow: 0 16px 48px rgba(17,24,39,.07);
    overflow: hidden;
    margin-top: 8px;
}

.chat-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 19px 21px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #fff 0%, #fcfaf7 100%);
}

.chat-panel-title-wrap {
    display: flex;
    align-items: center;
    gap: 11px;
}

.chat-panel-orb {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    background: var(--peach);
    border: 1px solid #e4cfc5;
    color: var(--navy) !important;
    font-weight: 700;
}

.chat-panel-title {
    color: var(--navy) !important;
    font-size: 15px;
    font-weight: 700;
}

.chat-panel-subtitle {
    color: #8b939e !important;
    font-size: 10px;
    margin-top: 2px;
}

.chat-ready-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 9px;
    border-radius: 999px;
    background: #edf4ed;
    border: 1px solid #dbe8db;
    color: #58725e !important;
    font-size: 10px;
    font-weight: 600;
}

.chat-messages {
    padding: 20px 21px 7px;
    min-height: 105px;
    max-height: 420px;
    overflow-y: auto;
}

.chat-empty {
    text-align: center;
    padding: 16px 10px 11px;
}

.chat-empty-title {
    color: var(--navy) !important;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 5px;
}

.chat-empty-text {
    color: #8a929d !important;
    font-size: 11px;
    line-height: 1.6;
}

.chat-bubble {
    max-width: 78%;
    padding: 11px 14px;
    border-radius: 15px;
    margin: 8px 0;
    font-size: 12px;
    line-height: 1.65;
    white-space: pre-wrap;
}

.chat-bubble.user {
    margin-left: auto;
    color: #fff !important;
    background: var(--navy);
    border-bottom-right-radius: 5px;
}

.chat-bubble.ai {
    margin-right: auto;
    color: #475467 !important;
    background: #f4f1ec;
    border: 1px solid var(--border);
    border-bottom-left-radius: 5px;
}

.chat-suggestion-title {
    color: #939a9f !important;
    font-size: 9px;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 7px 0 7px;
}

.chat-composer {
    padding: 14px 17px 17px;
    border-top: 1px solid var(--border);
    background: #fbfaf8;
}

.chat-input-label {
    color: #747d89 !important;
    font-size: 10px;
    margin-bottom: 6px;
}

.chat-panel + div .stTextInput input {
    min-height: 45px;
}

.chat-send > div > button {
    min-height: 45px !important;
    background: var(--navy) !important;
    color: #fff !important;
    border: 0 !important;
}

.chat-send > div > button:hover {
    background: #273244 !important;
}

.transcript-box {
    max-height: 430px;
    overflow: auto;
    white-space: pre-wrap;
    padding: 17px;
    background: #fbfaf8;
    border: 1px solid var(--border);
    border-radius: 13px;
    color: #667085 !important;
    font: 12px/1.8 "DM Sans";
}

[data-testid="stFileUploader"] section {
    background: #fff !important;
    border: 1px dashed #cfc4ba !important;
    border-radius: 13px !important;
}

[data-testid="stFileUploader"] section > div { padding: 10px !important; }

[data-testid="stFileUploader"] button {
    background: var(--navy) !important;
    color: #fff !important;
    border: 0 !important;
    border-radius: 9px !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #7b8491 !important;
}

.stExpander {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: #fff !important;
}

.stExpander summary,
.stExpander summary p {
    color: var(--navy) !important;
}

.stAlert {
    border-radius: 12px !important;
}

hr { border-color: var(--border) !important; }
footer { visibility: hidden; }

.empty-feature {
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 19px;
    background: rgba(255,255,255,.65);
    transition: all .2s ease;
}

.empty-feature:hover {
    background: #fff;
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(17,24,39,.05);
}

.empty-feature .num { color: #a09a93 !important; font-size: 10px; }
.empty-feature h4 { font-size: 15px; margin: 12px 0 6px; color: var(--navy) !important; }
.empty-feature p { font-size: 12px; line-height: 1.6; color: #7b8491 !important; margin: 0; }

@media(max-width:800px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .topbar { margin-bottom: 35px; }
    .top-status { display: none; }
    .hero h1 { font-size: 46px !important; letter-spacing: -2px; }
    .session-title { font-size: 29px; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "result": None,
    "chat_history": [],
    "analysis_source": "",
    "analysis_language": "english",
    "processing": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def safe(value):
    return html.escape(str(value if value is not None else ""))

def display_value(value):
    if value is None:
        return "Nothing was extracted."
    if isinstance(value, str):
        return value
    return str(value)

def run_analysis(source, language):
    progress = st.progress(0, text="Initializing AI pipeline…")
    status = st.empty()

    status.markdown("**01 · Processing media**  — extracting audio")
    progress.progress(12, text="Processing media…")
    chunks = process_input(source)

    status.markdown("**02 · Transcribing**  — converting speech to text")
    progress.progress(31, text="Transcribing…")
    transcript = transcribe_all(chunks, language)

    status.markdown("**03 · Understanding**  — generating title and summary")
    progress.progress(52, text="Generating understanding…")
    title = generate_title(transcript)
    summary = summerizes(transcript)

    status.markdown("**04 · Extracting insights**  — decisions, tasks and questions")
    progress.progress(72, text="Extracting insights…")
    action_items = extract_action_item(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    status.markdown("**05 · Building knowledge base**  — enabling AI chat")
    progress.progress(90, text="Building RAG knowledge base…")
    rag_chain = build_rag_chain(transcript)

    progress.progress(100, text="Analysis complete")
    status.empty()
    progress.empty()

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="brand">
    <span class="brandmark">✦</span><span>VidMind</span><span class="brand-ai">AI</span>
  </div>
  <div class="top-status"><span class="live-dot"></span> Intelligence workspace</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# EMPTY / CREATE VIEW
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.result:

    st.markdown("""
    <section class="hero">
      <div class="eyebrow">AI VIDEO INTELLIGENCE</div>
      <h1>Turn long videos into<br><span>useful knowledge.</span></h1>
      <p class="hero-copy">
        Turn meetings, lectures, interviews and YouTube videos into clear,
        actionable knowledge — without watching everything twice.
      </p>
      <div class="feature-pills">
        <div class="pill">✦ <b>Transcribe</b></div>
        <div class="pill">◈ <b>Summarize</b></div>
        <div class="pill">✓ <b>Extract</b></div>
        <div class="pill">⌁ <b>Ask with RAG</b></div>
      </div>
    </section>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
      <div class="upload-top">
        <div class="upload-icon">↑</div>
        <div>
          <h3>Start with a video</h3>
          <p>Upload a local file or paste a YouTube URL below.</p>
        </div>
      </div>
      <div class="upload-hint">MP4 · MOV · MP3 · WAV · YouTube</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload media",
        type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "mpeg"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"><span>OR USE A LINK</span></div>', unsafe_allow_html=True)
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")

    c1, c2, c3 = st.columns([1.2, 1, 1.2], gap="small")
    with c1:
        language = st.selectbox("Language", ["english", "hinglish"], label_visibility="collapsed")
    with c2:
        st.markdown("")
    with c3:
        st.markdown('<div class="primary">', unsafe_allow_html=True)
        analyse = st.button("✦  Analyze video", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:30px">
      <div class="empty-feature"><div class="num">01</div><h4>Understand</h4><p>Turn long-form speech into searchable knowledge.</p></div>
      <div class="empty-feature"><div class="num">02</div><h4>Extract</h4><p>Surface decisions, action items and open questions.</p></div>
      <div class="empty-feature"><div class="num">03</div><h4>Ask</h4><p>Chat directly with the content using your RAG engine.</p></div>
    </div>
    """, unsafe_allow_html=True)

    if analyse:
        if not uploaded and not url.strip():
            st.error("Add a video file or paste a YouTube URL first.")
        else:
            source = url.strip()
            if uploaded:
                temp_dir = Path(".vidmind_uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)
                suffix = Path(uploaded.name).suffix or ".mp4"
                temp_path = temp_dir / f"upload{suffix}"
                temp_path.write_bytes(uploaded.getbuffer())
                source = str(temp_path)

            try:
                st.session_state.processing = True
                with st.container():
                    st.markdown("""
                    <div class="processing-card">
                      <div class="spinner"></div>
                      <div class="eyebrow">VIDMIND ENGINE</div>
                      <h2 style="margin:9px 0 4px">Understanding your video</h2>
                      <p class="muted">Transcribing, extracting insights and building your AI knowledge base.</p>
                      <div class="progress-track"><div class="progress-fill"></div></div>
                    </div>
                    """, unsafe_allow_html=True)
                result = run_analysis(source, language)
                st.session_state.result = result
                st.session_state.analysis_source = uploaded.name if uploaded else url
                st.session_state.analysis_language = language
                st.session_state.chat_history = []
                st.session_state.processing = False
                try:
                    if uploaded and temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                st.rerun()
            except Exception as e:
                st.session_state.processing = False
                st.error(f"Analysis failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS VIEW
# ──────────────────────────────────────────────────────────────────────────────
else:
    r = st.session_state.result

    top1, top2 = st.columns([4,1])
    with top1:
        st.markdown('<div class="section-label">✦ ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="session-title">{safe(r["title"])}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="session-meta">{safe(st.session_state.analysis_source)} · '
            f'{safe(st.session_state.analysis_language.title())} · AI knowledge base ready</div>',
            unsafe_allow_html=True
        )
    with top2:
        st.markdown("")
        if st.button("＋ New analysis", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Summary + transcript
    a,b = st.columns([1.65,.75], gap="medium")
    with a:
        st.markdown(f"""
        <div class="card">
          <div class="card-head"><div class="card-label">✦ AI SUMMARY</div><div class="card-icon">✦</div></div>
          <div class="card-text">{safe(display_value(r["summary"]))}</div>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="card">
          <div class="card-head"><div class="card-label">▤ SOURCE</div><div class="card-icon">↗</div></div>
          <div class="kicker">Knowledge base</div>
          <p class="muted">Your transcript has been indexed and is ready for semantic questions.</p>
        """, unsafe_allow_html=True)
        if st.button("View full transcript →", use_container_width=True):
            st.session_state.show_transcript = True
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:6px">✦ VIDEO INSIGHTS</div>', unsafe_allow_html=True)
    # Insights
    c1,c2,c3 = st.columns(3, gap="medium")
    cards = [
        (c1, "✓", "green", "ACTION ITEMS", r["action_items"]),
        (c2, "◆", "purple", "KEY DECISIONS", r["key_decisions"]),
        (c3, "?", "amber", "OPEN QUESTIONS", r["open_questions"]),
    ]
    for col, icon, color, label, value in cards:
        with col:
            st.markdown(f"""
            <div class="card insight-card">
              <div class="insight-icon {color}">{icon}</div>
              <div class="card-label">{label}</div>
              <div class="card-text">{safe(display_value(value))}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Interactive AI Chat — the core feature
    st.markdown("""
    <div class="chat-panel">
      <div class="chat-panel-header">
        <div class="chat-panel-title-wrap">
          <div class="chat-panel-orb">✦</div>
          <div>
            <div class="chat-panel-title">Chat with your video</div>
            <div class="chat-panel-subtitle">Ask questions and get answers grounded in the transcript</div>
          </div>
        </div>
        <div class="chat-ready-badge"><span>●</span> AI ready</div>
      </div>
      <div class="chat-messages">
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="chat-empty">
          <div class="chat-empty-title">What would you like to know?</div>
          <div class="chat-empty-text">
            Ask about ideas, decisions, people, examples, action items,
            or anything discussed in the video.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            role_class = "user" if msg["role"] == "user" else "ai"
            role_label = "You" if msg["role"] == "user" else "VidMind AI"
            st.markdown(
                f'<div style="color:#98a2b3;font-size:9px;margin-top:8px;'
                f'{"text-align:right;" if role_class == "user" else ""}">{role_label}</div>'
                f'<div class="chat-bubble {role_class}">{safe(msg["content"])}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown('<div class="chat-suggestion-title">Try asking</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3, gap="small")
        suggestions = [
            "What are the main takeaways?",
            "What decisions were made?",
            "What action items were mentioned?",
        ]
        for col, q in zip([s1, s2, s3], suggestions):
            with col:
                if st.button(q, key="suggest_" + q, use_container_width=True):
                    st.session_state.pending_question = q

    st.markdown('<div class="chat-composer"><div class="chat-input-label">ASK YOUR VIDEO</div>', unsafe_allow_html=True)
    if st.session_state.pop("clear_chat_input", False):
        st.session_state.chat_input_box = ""

    input_col, send_col = st.columns([5, 1], gap="small")
    with input_col:
        chat_question = st.text_input(
            "Question",
            placeholder="e.g. What did the speaker say about AI pricing?",
            label_visibility="collapsed",
            key="chat_input_box",
        )
    with send_col:
        st.markdown('<div class="chat-send">', unsafe_allow_html=True)
        send_question = st.button("Send  ↗", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    pending = st.session_state.pop("pending_question", "")
    question = (chat_question.strip() if chat_question else "") if send_question else pending

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Searching your video…"):
            try:
                answer = ask_question(r["rag_chain"], question)
            except Exception as e:
                answer = f"I couldn't answer that: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.clear_chat_input = True
        st.rerun()

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Transcript
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.expander("▤  Full transcript", expanded=False):
        st.markdown(f'<div class="transcript-box">{safe(display_value(r["transcript"]))}</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("""
    <div style="text-align:center;color:#55555d;font-size:10px;letter-spacing:.08em;margin-top:70px">
      VIDMIND AI · VIDEO → KNOWLEDGE
    </div>
    """, unsafe_allow_html=True)
