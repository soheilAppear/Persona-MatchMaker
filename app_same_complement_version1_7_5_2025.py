# ==============================================================
#  app.py  —  Streamlit Persona-Matcher v4
# --------------------------------------------------------------
#  • User pastes a bio.
#  • GPT returns (1) a 1-sentence summary  (2) MBTI type.
#  • Compute score =  50 % MBTI compatibility
#                    50 % semantic similarity (embeddings).
#  • Switch button:
#        - "Similar":  best score when letters are the same.
#        - "Complement": best when letters are opposite.
#  • Show top 15 matches.  Best 5 get a 6-line mini-dialogue.
# ==============================================================

import json, os, textwrap, numpy as np, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from math import sqrt

# ---------- 0.  Keys & constants ------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL  = "text-embedding-3-small"
CHAT_MODEL   = "gpt-4o-mini"

TOP_K        = 15               # characters to show
CHAT_TOP     = 5                # mini-chat only for best 5
CHAT_LINES   = 6
TOK_CHAT     = 180

# ---------- 1.  Helpers ----------------------------------------------------
@st.cache_data(show_spinner=False)
def embed(text: str) -> np.ndarray:
    """Return 1536-D embedding (cached)."""
    v = client.embeddings.create(model=EMBED_MODEL, input=text
         ).data[0].embedding
    return np.array(v, dtype=np.float32)

def cosine(a, b) -> float:
    return max(0.0, float(np.dot(a, b) /
                ((np.linalg.norm(a)*np.linalg.norm(b))+1e-8)))

def gpt_chat(prompt, temp=0.7, maxtok=TOK_CHAT):
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":prompt}],
        temperature=temp,
        max_tokens=maxtok
    ).choices[0].message.content.strip()

def classify_mbti(text: str) -> tuple[str, str]:
    """
    Ask GPT for MBTI + 1-sentence summary.
    Returns (mbti, summary).
    """
    prompt = (
      "Return JSON with keys mbti (4-letter type) and summary "
      "(ONE 20-word sentence) that describes the writer."
    )
    rsp = gpt_chat(prompt + "\n\nTEXT:\n" + text[:800], 0.4, 120)
    data = json.loads(rsp)
    return data["mbti"].upper().strip(), data["summary"].strip()

# ---------- 2.  MBTI scoring ----------------------------------------------
def mbti_similarity(user: str, char: str, mode: str) -> float:
    """
    user, char: 4-letter MBTI codes
    mode: 'similar' or 'complement'
    Returns value 0-1 over 4 letters.
    """
    score = 0
    for u, c in zip(user, char):
        same = u == c
        if mode == "similar":
            score += 1 if same else 0
        else:                              # complement mode
            score += 1 if not same else 0
    return score / 4                       # 0-1

# ---------- 3.  Load DB + pre-embed summaries ------------------------------
@st.cache_data(show_spinner=False)
def load_db(path="characters.json"):
    data = json.load(open(path, encoding="utf-8"))
    for rec in data:
        rec["embed"] = embed(rec["summary"])
    return data

# ---------- 4.  UI ---------------------------------------------------------
st.set_page_config("Persona-Matcher v4", "🤖")
st.title("⚡ Persona-Matcher  |  MBTI × Semantic Hybrid")

bio = st.text_area("Paste a short bio or chat snippet about you", height=160)

mode = st.radio("Personality preference",
                ("similar", "complement"),
                captions=("Match my letters", "Opposite letters complement me"))

if st.button("Find matches") and bio.strip():
    with st.spinner("Analysing…"):
        user_mbti, user_summary = classify_mbti(bio)
        vec_summary  = embed(user_summary)
        vec_paragraph = embed(bio)
        db = load_db()

        # ----- score each character ---------------------------------------
        ranked = []
        for ch in db:
            # MBTI part
            s_mbti = mbti_similarity(user_mbti, ch["mbti"], mode)   # 0-1
            # semantic part = avg of summary-summary & paragraph-summary
            sem1 = cosine(vec_summary,   ch["embed"])
            sem2 = cosine(vec_paragraph, ch["embed"])
            s_sem = 0.5 * (sem1 + sem2)
            score = 0.5 * s_mbti + 0.5 * s_sem
            ranked.append((ch, score))
        ranked.sort(key=lambda t: t[1], reverse=True)
        ranked = ranked[:TOP_K]

    # ---------- show user summary ----------------------------------------
    st.write(f"**Your MBTI**: `{user_mbti}`")
    st.write(f"**Your summary**: {user_summary}")

    # ---------- display matches -----------------------------------------
    lo = ranked[-1][1]
    hi = ranked[0][1]
    def pct(x):                          # rescale best→100, worst→30
        return (x - lo) / (hi - lo + 1e-8) * 70 + 30

    for idx, (ch, sc) in enumerate(ranked, 1):
        st.markdown(f"### {idx}. {ch['name']} ({ch['category']}) "
                    f"`{ch['mbti']}` — {pct(sc):.1f}%")
        if ch["image_url"]:
            st.image(ch["image_url"], width=120)
        st.write(ch["summary"])

        if idx <= CHAT_TOP:
            prompt = textwrap.dedent(f"""
                Imagine a brief conversation.
                The user summary: "{user_summary}"
                Character summary: "{ch['summary']}"
                Write {CHAT_LINES} lines alternating User / {ch['name']}.
                Keep each line ≤16 words.
            """)
            st.code(gpt_chat(prompt, 0.8), language="text")
