#!/usr/bin/env python3
# ======================================================================
#  app.py — Persona-Matcher v5
# ----------------------------------------------------------------------
#  • 30 %  values alignment         (summary ↔ summary cosine)
#  • 10 %  shared interests         (full bio ↔ summary cosine)
#  • 25 %  E/I match or contrast
#  • 20 %  T/F match or contrast
#  • 15 %  J/P match or contrast
#  --------------------------------------------
#  TOP 15 shown, best 5 get a 6-line mini-chat.
# ======================================================================

import json, os, textwrap, numpy as np, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── 0. keys & models ────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

TOP_K     = 25    # cards shown
CHAT_TOP  = 10     # mini-dialogs
CHAT_LINES = 100
TOK_CHAT   = 500

# ── 1.-- helper: cached embedding ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def embed(text: str) -> np.ndarray:
    vec = client.embeddings.create(model=EMBED_MODEL, input=text
          ).data[0].embedding
    return np.array(vec, dtype=np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, float(np.dot(a, b) /
        ((np.linalg.norm(a)*np.linalg.norm(b))+1e-8)))

# ── 2.-- helper: cheap chat completions ─────────────────────────────────
def gpt_chat(prompt: str, temp=0.5, maxtok=120) -> str:
    return client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system", "content": prompt}],
        temperature=temp,
        max_tokens=maxtok
    ).choices[0].message.content.strip()

# ── 3.-- classify user MBTI + summary sentence ──────────────────────────
def classify_user(text: str) -> tuple[str, str]:
    ask = ("Respond ONLY with JSON {\"mbti\":\"....\",\"summary\":\"one 20-word sentence\"}")
    data = json.loads(gpt_chat(ask + "\n\nTEXT:\n" + text[:800], 0.4, 100))
    return data["mbti"].upper(), data["summary"].strip()

# ── 4.-- MBTI letter score (similar or complement) ──────────────────────
def letter_score(u: str, c: str, mode: str, idx: int, weight: float) -> float:
    """Return weighted 0/weight depending on letter match/opposite."""
    same = u[idx] == c[idx]
    good = same if mode == "similar" else (not same)
    return weight if good else 0.0

# ── 5.-- load characters & pre-embed summaries ──────────────────────────
@st.cache_data(show_spinner=False)
def load_db(path="characters_10.json"):
    rows = json.load(open(path, encoding="utf-8"))
    for r in rows:
        r["embed"] = embed(r["summary"])
    return rows

# ── 6.-- mini-chat generator (flavour only) ─────────────────────────────
def mini_chat(user_sum: str, name: str, char_sum: str):
    p = textwrap.dedent(f"""
        Write {CHAT_LINES} lines alternating User / {name}.
        User summary: "{user_sum}"
        Character summary: "{char_sum}"
        ≤16 words per line.
    """)
    return gpt_chat(p, 0.8, TOK_CHAT)

# ── 7. Streamlit UI ─────────────────────────────────────────────────────
st.set_page_config("Persona-Matcher v5", "🤖")
st.title("⚡ Persona-Matcher | Complement or Similar?")

bio = st.text_area("Paste a short bio or chat snippet about you", height=170)

mode = st.radio(
    "Personality preference",
    options=("similar", "complement"),
    captions=("Match my letters", "Opposites attract")
)

if st.button("Find matches") and bio.strip():
    with st.spinner("Analysing…"):
        user_mbti, user_sum = classify_user(bio)
        vec_sum  = embed(user_sum)  # values vector
        vec_bio  = embed(bio)       # interests vector
        db       = load_db()

        ranked = []
        for ch in db:
            # 1) values alignment (30 %)
            val_sim = cosine(vec_sum, ch["embed"]) * 0.30

            # 2) interests overlap (10 %)
            int_sim = cosine(vec_bio, ch["embed"]) * 0.10

            # 3) MBTI-letter complement or match (total 60 %)
            mbti_part = (
                letter_score(user_mbti, ch["mbti"], mode, 0, 0.25) +  # E/I
                letter_score(user_mbti, ch["mbti"], mode, 2, 0.20) +  # T/F
                letter_score(user_mbti, ch["mbti"], mode, 3, 0.15)    # J/P
            )

            score = val_sim + int_sim + mbti_part   # already sums to 1.0 max
            ranked.append((ch, score))

        ranked.sort(key=lambda t: t[1], reverse=True)
        ranked = ranked[:TOP_K]

    # ── display your self-info ──────────────────────────────────────────
    st.write(f"**Your MBTI:** `{user_mbti}`")
    st.write(f"**Your summary:** {user_sum}")

    # ── rescale best→100, worst→30 for nicer UX ─────────────────────────
    hi, lo = ranked[0][1], ranked[-1][1]
    pct = lambda x: (x - lo) / (hi - lo + 1e-8) * 70 + 30

    # ── show matches ────────────────────────────────────────────────────
    for i, (ch, raw) in enumerate(ranked, 1):
        st.markdown(
            f"### {i}. {ch['name']} ({ch['category']}) "
            f"`{ch['mbti']}` — {pct(raw):.1f}%"
        )
        if ch["image_url"]:
            st.image(ch["image_url"], width=120)
        st.write(ch["summary"])

        if i <= CHAT_TOP:
            st.code(mini_chat(user_sum, ch["name"], ch["summary"]),
                    language="text")
