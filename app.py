#!/usr/bin/env python
"""
Streamlit AI-Persona Matcher  (Big-Five edition)
-----------------------------------------------
• Upload or paste two ChatGPT histories
• Summarise → persona
• Big-Five trait ratings (JSON)
• 12-line dialogue
• Compatibility score 0-100
"""

import json, os, textwrap, numpy as np, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ─────────── Config ───────────
MAX_WORDS   = 60
LINES       = 12
MODEL       = "gpt-4o-mini"
# ──────────────────────────────

# 1) try server-side key
load_dotenv()
SERVER_KEY = os.getenv("OPENAI_API_KEY")

# 2) or let user paste a key (only stored in session_state)
if SERVER_KEY:
    client = OpenAI(api_key=SERVER_KEY)
else:
    st.sidebar.subheader("🔑 OpenAI key required")
    USER_KEY = st.sidebar.text_input(
        "Paste your key here (kept only in this tab)",
        type="password",
        key="user_api_key",
    )
    if not USER_KEY:
        st.stop()
    client = OpenAI(api_key=USER_KEY)

# ─────────── LLM helpers ───────────
@st.cache_data(show_spinner=False)
def ask_llm(messages, temp=0.7, max_tokens=256):
    return client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=temp, max_tokens=max_tokens,
    ).choices[0].message.content.strip()

def persona_of(history: str) -> str:
    prompt = (
        f"Summarise the user's personality, interests, communication style "
        f"in ≤{MAX_WORDS} words, third-person, no bullet lists.\n\n"
        f"=== HISTORY ===\n{history}\n=== END ===\n\nPERSONA:"
    )
    return ask_llm([{"role": "user", "content": prompt}], 0.7, 120)

def big_five(persona: str) -> np.ndarray:
    sys = (
        "Return ONLY valid JSON like "
        '{"O":float,"C":float,"E":float,"A":float,"N":float} '
        "rated 1-5 for the description."
    )
    raw = ask_llm(
        [{"role": "system", "content": sys},
         {"role": "user", "content": persona}],
        0.0, 80)
    data = json.loads(raw)
    return np.array([data[k] for k in ("O","C","E","A","N")], np.float32)

def chat_lines(p1, p2) -> str:
    sys = textwrap.dedent(f"""
        Persona-A: {p1}
        Persona-B: {p2}
        Write exactly {LINES} lines, alternating 'A:' then 'B:'.
        Each line ≤ 25 words, friendly, no narration.
    """).strip()
    return ask_llm([{"role":"system","content":sys}], 0.8, 500)

def score(v1, v2) -> float:
    dist = np.linalg.norm(v1 - v2)              # 0 … ~9
    max_dist = np.sqrt((4**2)*5)                # 4 per trait
    return round((1 - dist / max_dist) * 100, 1)

# ─────────── UI ───────────
st.title("🗣️ AI Persona Matcher")

col1, col2 = st.columns(2)
def history_input(col, tag):
    upl = col.file_uploader(f"History {tag} (txt)", type=["txt"], key=f"file{tag}")
    return (upl.read().decode() if upl else
            col.text_area(f"…or paste History {tag}", height=180, key=f"txt{tag}"))

histA = history_input(col1, "A")
histB = history_input(col2, "B")

run = st.button("▶️ Run Match", disabled=not(histA and histB))

if run:
    with st.spinner("Thinking …"):
        pA, pB = persona_of(histA), persona_of(histB)
        tA, tB = big_five(pA), big_five(pB)
        chat   = chat_lines(pA, pB)
        match  = score(tA, tB)

    st.subheader("Persona A"); st.info(pA)
    st.subheader("Persona B"); st.info(pB)
    st.subheader("Dialogue");  st.code(chat, language="text")
    st.subheader("Compatibility"); st.metric("Match Score", f"{match} / 100")
