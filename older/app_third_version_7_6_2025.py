
#!/usr/bin/env python3
# app.py — Streamlit front-end for Persona-Matcher
import json, os, textwrap, random, numpy as np, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────
MODEL  = "gpt-4o-mini"
TOP_K  = 10      # how many matches to display
POOL   = 40      # consider top-40 then sample 10
LINES  = 50      # dialogue lines per character
MAXTOK = 900     # token budget for the 50-line chat

# ─── Initialise OpenAI ────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(messages, temp=0.7, max_tokens=MAXTOK):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
    ).choices[0].message.content.strip()

# ─── Persona extraction helpers ───────────────────────────────────────────
def persona_of(user_text: str) -> str:
    prompt = ("Summarise the user’s personality, interests, communication style "
              "in ≤60 words, third-person, plain sentences.")
    return ask_llm([{"role": "user", "content": prompt + "\n\n" + user_text}],
                   temp=0.7, max_tokens=120)

def big_five(persona: str) -> np.ndarray:
    sys_msg = 'Return ONLY JSON {"O":f,"C":f,"E":f,"A":f,"N":f} using 1-5 scale.'
    data = json.loads(ask_llm(
        [{"role": "system", "content": sys_msg},
         {"role": "user",   "content": persona}],
        temp=0.0, max_tokens=80))
    return np.array([data[k] for k in "OCEAN"], float)

# ─── DB & similarity ──────────────────────────────────────────────────────
def load_db(path: str = "characters.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def similarity(u: np.ndarray, v: np.ndarray) -> float:
    # Euclidean → convert to 0-1 where 1 is identical, 0 is farthest (4 per dimension)
    return 1.0 - np.linalg.norm(u - v) / np.sqrt(5 * 4 ** 2)

def choose_matches(user_vec: np.ndarray, db: list,
                   pool: int = POOL, k: int = TOP_K):
    # score everyone
    scored = [
        (char, similarity(
            user_vec,
            np.array(list(char["personality_vector"].values()), float)
        ))
        for char in db
    ]
    scored.sort(key=lambda x: x[1], reverse=True)  # best first
    top_pool = scored[:pool]                       # pool size
    return random.sample(top_pool, min(k, len(top_pool)))  # list of (char, score)

# ─── Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Persona-Matcher", page_icon="🤖")
st.title("Persona-Matcher: Find your famous twin")

bio = st.text_area("Paste some chat or a short bio about yourself", height=180)

if st.button("Match!") and bio.strip():
    with st.spinner("Analysing your persona …"):
        user_persona  = persona_of(bio)
        user_vector   = big_five(user_persona)
        database      = load_db()
        selections    = choose_matches(user_vector, database)

    st.subheader("Your Persona")
    st.info(user_persona)

    for idx, (char, score) in enumerate(selections, 1):
        name, cat = char["name"], char["category"]
        desc, img = char["description"], char["image_url"]

        st.markdown(f"### {idx}. {name} *({cat})* — {score*100:.1f}% match")
        if img:
            st.image(img, width=120)
        st.write(desc)

        chat_prompt = textwrap.dedent(f"""
            User persona: {user_persona}
            Character persona: {desc}
            Write {LINES} lines of playful, back-and-forth dialogue.
            Alternate "User:" then "Character:".
            Keep each line ≤18 words.
        """).strip()

        dialogue = ask_llm([{"role": "system", "content": chat_prompt}], temp=0.8)
        st.code(dialogue, language="text")







# from dotenv import load_dotenv
# import json, os, textwrap, random, numpy as np, streamlit as st
# from openai import OpenAI

# MODEL   = "gpt-4o-mini"
# TOP_K   = 10     # show 10 matches instead of 5
# POOL    = 40     # consider top-40 similar, then sample 10
# from dotenv import load_dotenv
# LINES   = 50     # lines per mini-chat
# MAXTOK  = 900    # generous budget for 50 lines

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def ask_llm(messages, temp=0.7, max_tokens=MAXTOK):
#     return client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         temperature=temp,
#         max_tokens=max_tokens,
#     ).choices[0].message.content.strip()

# # ─── Persona + Big-Five helpers ────────────────────────────────────────────
# def persona_of(text):
#     prompt = ("Summarise the user’s personality, interests, communication style "
#               "in ≤60 words, third-person, plain sentences.")
#     return ask_llm([{"role": "user", "content": prompt + "\n\n" + text}],
#                    temp=0.7, max_tokens=120)

# def big_five(persona):
#     sys = 'Return ONLY JSON {"O":f,"C":f,"E":f,"A":f,"N":f} 1-5.'
#     data = json.loads(ask_llm(
#         [{"role":"system","content":sys},
#          {"role":"user","content":persona}], temp=0.0, max_tokens=80))
#     return np.array([data[k] for k in "OCEAN"], float)

# # ─── DB & matching ────────────────────────────────────────────────────────
# def load_db(path="characters.json"):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def sim(u, v):
#     return 1 - np.linalg.norm(u - v) / np.sqrt(5*4**2)

# def pick_matches(user_v, db, k=TOP_K, pool=POOL):
#     scored = [(c, sim(
#         user_v, np.array(list(c["personality_vector"].values()), float)))
#         for c in db]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     top_pool  = [c for c, _ in scored[:pool]]
#     return random.sample(top_pool, min(k, len(top_pool)))

# # ─── Streamlit UI ─────────────────────────────────────────────────────────
# st.set_page_config("Persona-Matcher", "🤖")
# st.title("Persona-Matcher: Find your famous twin")

# bio = st.text_area("Paste some chat/bio about you", height=180)

# if st.button("Match!") and bio.strip():
#     with st.spinner("Analysing …"):
#         persona = persona_of(bio)
#         user_v  = big_five(persona)
#         db      = load_db()
#         picks   = pick_matches(user_v, db)

#     st.subheader("Your Persona")
#     st.info(persona)

#     for idx, char in enumerate(picks, 1):
#         name, cat = char["name"], char["category"]
#         desc, img = char["description"], char["image_url"]
#         st.markdown(f"### {idx}. {name}  *({cat})*")
#         if img: st.image(img, width=120)
#         st.write(desc)

#         chat_prompt = textwrap.dedent(f"""
#             User persona: {persona}
#             Character persona: {desc}
#             Write {LINES} lines of playful, back-and-forth dialogue.
#             Alternate "User:" then "Character:" on each line.
#             Keep each line ≤18 words.
#         """).strip()

#         dialogue = ask_llm([{"role":"system","content":chat_prompt}], temp=0.8)
#         st.code(dialogue, language="text")
