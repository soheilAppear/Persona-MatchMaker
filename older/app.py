#!/usr/bin/env python3
"""
app.py — Simple Persona‑Matcher (v3)
------------------------------------
Embedding‑only matcher with blended similarity and friendlier score scale.

Changes v3
~~~~~~~~~~
1. **Two similarities blended**
   • `sim_short`  – cosine between ONE‑sentence user summary and character blurb.
   • `sim_long`   – cosine between full user paragraph and character blurb.
   • Score  = 0.6 × sim_short + 0.4 × sim_long.
2. **Pretty‑score mapping**
   Raw cosine 0.15 → 0 %   |   0.75 → 100 % (clamped).
3. Still shows top 20 and a 10‑line mini chat for each.

Cost: ≈ $0.002 per click (one extra embedding call).
"""

# ── Imports ─────────────────────────────────────────────────────────────
import json, os, textwrap, numpy as np, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
EMBED_MODEL  = "text-embedding-3-small"
CHAT_MODEL   = "gpt-4o-mini"
TOP_K        = 20      # characters to show
CHAT_LINES   = 10      # lines in the mini chat
TOK_CHAT     = 250     # token cap per chat

# ── OpenAI client ───────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Caching helpers ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def embed(text: str) -> np.ndarray:
    """Return a 1536‑D embedding (cached)."""
    vec = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    return np.array(vec, dtype=np.float32)

@st.cache_data(show_spinner=False)
def load_db(path: str = "characters.json"):
    """Load characters and attach an embedding to each."""
    db = json.load(open(path, encoding="utf-8"))
    for c in db:
        seed = c["description"] or c["name"]
        if "embed" not in c:
            c["embed"] = embed(seed)
    return db

# ── User‑side helpers ───────────────────────────────────────────────────

def make_summary(text: str) -> str:
    """One vivid ≤25‑word sentence describing the writer of *text*."""
    prompt = (
        "Write ONE vivid sentence (≤25 words) that captures the personality, "
        "tone and core values of the writer.")
    return client.chat.completions.create(
        model       = CHAT_MODEL,
        messages    = [{"role":"user","content": prompt + "\n\n" + text[:800]}],
        temperature = 0.5,
        max_tokens  = 60
    ).choices[0].message.content.strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return max(0.0, num / den)


def pretty(raw: float) -> float:
    """Map raw cosine 0.15–0.75 → 0‑100 %."""
    return max(0.0, min(1.0, (raw - 0.15) / 0.60)) * 100


def mini_chat(user_summary: str, char_name: str, char_desc: str) -> str:
    """Return a 10‑line back‑and‑forth chat."""
    prompt = textwrap.dedent(f"""
        You are staging a playful chat.
        User summary  : {user_summary}
        Character     : {char_name} — {char_desc}
        Write {CHAT_LINES} alternating lines (User:, {char_name}:).
        Keep each line ≤18 words.
    """)
    return client.chat.completions.create(
        model       = CHAT_MODEL,
        messages    = [{"role":"system","content": prompt}],
        temperature = 0.8,
        max_tokens  = TOK_CHAT
    ).choices[0].message.content.strip()

# ── Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config("Persona‑Matcher v3", "🤖")
st.title("⚡ Simple Persona‑Matcher (v3)")

bio = st.text_area("Paste a short bio or chat snippet", height=170)

if st.button("Match me!") and bio.strip():
    with st.spinner("Embedding & ranking …"):
        summary   = make_summary(bio)   # one crisp sentence
        vec_short = embed(summary)      # summary vector
        vec_long  = embed(bio)          # paragraph vector

        db = load_db()
        ranked = sorted(
            (
                (
                    c,
                    0.6 * cosine(vec_short, c["embed"]) +
                    0.4 * cosine(vec_long,  c["embed"])
                ) for c in db
            ),
            key=lambda x: x[1], reverse=True
        )[:TOP_K]

    st.info(f"**Your summary:** {summary}")

    for i, (char, score_raw) in enumerate(ranked, 1):
        pct = pretty(score_raw)
        st.markdown(f"### {i}. {char['name']} *({char['category']})* — {pct:.1f}%")
        if char["image_url"]:
            st.image(char["image_url"], width=120)
        st.write(char["description"])
        st.code(mini_chat(summary, char['name'], char['description']), language="text")



# #!/usr/bin/env python3
# """
# app.py — Streamlit front-end (embedding-only)

# • Embeds your text and every character description with text-embedding-3-small.
# • Cosine similarity → % match.
# • Shows top 20 matches with a 10-line chat.

# Cost: ~0.2¢ per 1 000 matches.
# """

# import json, os, textwrap, numpy as np, streamlit as st
# from dotenv import load_dotenv
# from openai import OpenAI

# # ── Config ────────────────────────────────────────────────────────────────
# EMBED_MODEL  = "text-embedding-3-small"
# CHAT_MODEL   = "gpt-4o-mini"
# TOP_K        = 20      # characters to show
# CHAT_LINES   = 10
# TOK_CHAT     = 250

# # ── Init OpenAI ────────────────────────────────────────────────────────────
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# @st.cache_data(show_spinner=False)
# def embed(text: str) -> np.ndarray:
#     vec = client.embeddings.create(model=EMBED_MODEL, input=text
#            ).data[0].embedding
#     return np.array(vec, dtype=np.float32)

# @st.cache_data(show_spinner=False)
# def load_db(path="characters.json"):
#     ds = json.load(open(path, encoding="utf-8"))
#     for c in ds:
#         seed = c["description"] or c["name"]
#         c["embed"] = embed(seed)
#     return ds

# def cosine(a: np.ndarray, b: np.ndarray) -> float:
#     return max(0.0, float(np.dot(a, b) /
#                ((np.linalg.norm(a)*np.linalg.norm(b)) + 1e-8)))

# def mini_chat(user_text, char_name, char_desc):
#     prompt = textwrap.dedent(f"""
#         You are staging a fun chat.
#         User text: «{user_text[:800]}»
#         Character: {char_name} — {char_desc}
#         Write {CHAT_LINES} lines alternating "User:" and "{char_name}:".
#         Keep each line ≤18 words.
#     """).strip()
#     return client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[{"role":"system","content":prompt}],
#         temperature=0.8,
#         max_tokens=TOK_CHAT
#     ).choices[0].message.content.strip()

# # ── Streamlit UI ──────────────────────────────────────────────────────────
# st.set_page_config("Simple Matcher", "🤖")
# st.title("⚡ Simple Persona-Matcher (Embedding Only)")

# txt = st.text_area("Paste a short bio or chat example", height=160)

# if st.button("Match me!") and txt.strip():
#     with st.spinner("Finding matches …"):
#         u_vec  = embed(txt)
#         db     = load_db()
#         ranked = sorted(
#             ((c, cosine(u_vec, c["embed"])) for c in db),
#             key=lambda x: x[1], reverse=True
#         )[:TOP_K]

#     for i, (char, sim) in enumerate(ranked, 1):
#         st.markdown(f"### {i}. {char['name']} *({char['category']})* — {sim*100:.1f}%")
#         if char["image_url"]:
#             st.image(char["image_url"], width=120)
#         st.write(char["description"])
#         st.code(mini_chat(txt, char["name"], char["description"]), language="text")
