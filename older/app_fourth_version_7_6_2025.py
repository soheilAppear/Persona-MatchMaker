

#!/usr/bin/env python3
"""
app.py  –  Streamlit front-end for Persona-Matcher (hybrid similarity version)

Author: you
Created: 2025-07-05
--------------------------------------------------
• Reads characters.json  (must contain 'personality_vector' for each record)
• Extracts the user's persona → Big-Five vector (O,C,E,A,N)
• Creates an embedding of the user's text with text-embedding-3-small
• Computes for every character:
      0.6 × personality_similarity  +  0.4 × semantic_similarity
• Shows the top-POOL most compatible characters, then samples TOP_K so results
  look fresh every click.
• Generates a 50-line back-and-forth chat for each match.
Required env  :  OPENAI_API_KEY
External libs :  streamlit, numpy, python-dotenv, openai
"""

# ── Imports ───────────────────────────────────────────────────────────────
import json                         # read characters.json
import os, random, textwrap         # misc std-lib helpers
from math import sqrt               # for cosine similarity
import numpy as np                  # vector math
import streamlit as st              # web UI
from dotenv import load_dotenv      # pull key from .env
from openai import OpenAI           # OpenAI SDK

# ── Config constants (tweak at will) ──────────────────────────────────────
CHAT_MODEL     = "gpt-4o-mini"      # cheap, fast chat model
EMBED_MODEL    = "text-embedding-3-small"
TOP_K          = 50                 # how many characters to show
POOL           = 40                 # take best 40, then sample 10
LINES          = 50                 # dialogue lines per character
MAX_TOK_CHAT   = 900                # budget for 50-line chat

PERSONALITY_W  = 0.6               # 60 % personality
SEMANTIC_W     = 0.4               # 40 % embedding similarity

# ── Initialise OpenAI client ──────────────────────────────────────────────
load_dotenv()                       # loads .env if present
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Helper: generic call wrapper for chat completions ─────────────────────
def ask_llm(msgs, temp=0.7, max_tokens=MAX_TOK_CHAT):
    """
    One-shot call to an OpenAI chat model.

    msgs : list[dict]  –  [{'role':'user'/'system', 'content':'...'}, …]
    temp : float       –  sampling temperature
    returns : str      –  model reply (stripped)
    """
    return client.chat.completions.create(
        model       = CHAT_MODEL,
        messages    = msgs,
        temperature = temp,
        max_tokens  = max_tokens,
    ).choices[0].message.content.strip()

# ── Persona extraction helpers ────────────────────────────────────────────
def persona_of(user_text: str) -> str:
    """Return a ≤60-word summary of the user in third-person."""
    prompt  = ("Summarise the user’s personality, interests, communication "
               "style in ≤60 words, third-person, plain sentences.")
    return ask_llm([{"role": "user",
                     "content": prompt + "\n\n" + user_text}],
                   temp=0.7, max_tokens=120)

def big_five(persona: str) -> np.ndarray:
    """Convert persona text → numpy array [O,C,E,A,N] (floats)."""
    sys_msg = 'Return ONLY JSON {"O":f,"C":f,"E":f,"A":f,"N":f} using 1-5 scale.'
    data    = json.loads(ask_llm(
               [{"role": "system", "content": sys_msg},
                {"role": "user",   "content": persona}],
               temp=0.0, max_tokens=80))
    return np.array([data[k] for k in "OCEAN"], float)

# ── Embedding utilities ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def embed_text(text: str) -> np.ndarray:
    """
    Get a 1536-D embedding for text using text-embedding-3-small.
    Cached so repeated calls with the same string are free.
    """
    emb = client.embeddings.create(
        model = EMBED_MODEL,
        input = text
    ).data[0].embedding
    return np.array(emb, dtype=np.float32)

@st.cache_data(show_spinner=False)
def load_db(path: str = "characters.json"):
    """Load characters + prepare an embedding for each record once."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for rec in raw:
        # Use description if meaningful; else fall back to name only
        embed_source = rec["description"] if rec["description"] != "Personality placeholder." \
                                           else rec["name"]
        rec["embedding"] = embed_text(embed_source)
    return raw

# ── Similarity functions ──────────────────────────────────────────────────
def personality_sim(u_vec: np.ndarray, v_vec: np.ndarray) -> float:
    """
    Euclidean-to-[0,1] similarity over Big-Five vectors (max distance = 4 per axis).
    """
    return 1.0 - np.linalg.norm(u_vec - v_vec) / sqrt(5 * 4 ** 2)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1,1]; clamp negatives to 0 for our purpose."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return max(0.0, float(np.dot(a, b) / denom))

def hybrid_score(u_per: np.ndarray, u_emb: np.ndarray, char: dict) -> float:
    """
    Combine personality and semantic similarities into one score.
    """
    pers_sim = personality_sim(u_per, np.array(
                 list(char["personality_vector"].values()), float))
    sem_sim  = cosine(u_emb, char["embedding"])
    return PERSONALITY_W * pers_sim + SEMANTIC_W * sem_sim   # weighted sum

def choose_matches(user_pvec, user_emb, db, pool=POOL, k=TOP_K):
    """
    Score every character → take the top 'pool' → random sample 'k'.
    Returns list[(character_dict, score)].
    """
    scored = [(char, hybrid_score(user_pvec, user_emb, char)) for char in db]
    scored.sort(key=lambda x: x[1], reverse=True)      # best first
    top_pool = scored[:pool]
    return random.sample(top_pool, min(k, len(top_pool)))

# ── Streamlit UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Persona-Matcher", page_icon="🤖")
st.title("Persona-Matcher  |  Hybrid Similarity Demo")

user_bio = st.text_area("Paste some chat or a short bio about yourself", height=180)

if st.button("Match!") and user_bio.strip():
    with st.spinner("Analysing your persona …"):
        user_persona   = persona_of(user_bio)            # 60-word summary
        user_pvec      = big_five(user_persona)          # 5-D vector
        user_emb       = embed_text(user_bio)            # 1536-D embedding
        database       = load_db()                       # characters + cached embeds
        selections     = choose_matches(user_pvec, user_emb, database)

    # ── Display user summary ──────────────────────────────────────────────
    st.subheader("Your Persona")
    st.info(user_persona)

    # ── Show each chosen character ───────────────────────────────────────
    for idx, (char, score) in enumerate(selections, 1):
        name, cat   = char["name"], char["category"]
        desc, img   = char["description"], char["image_url"]

        st.markdown(f"### {idx}. {name} *({cat})* — {score*100:.1f}% match")
        if img:
            st.image(img, width=120)
        st.write(desc)

        # Build a prompt for a 50-line chat
        chat_prompt = textwrap.dedent(f"""
            User persona: {user_persona}
            Character persona: {desc}
            Write {LINES} lines of playful, back-and-forth dialogue.
            Alternate "User:" then "Character:".
            Keep each line ≤18 words.
        """).strip()

        dialogue = ask_llm([{"role": "system", "content": chat_prompt}], temp=0.8)
        st.code(dialogue, language="text")

