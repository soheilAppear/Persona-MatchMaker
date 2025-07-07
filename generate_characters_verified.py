#!/usr/bin/env python3
"""generate_characters_verified.py
--------------------------------------------------
* Produce **N unique, well‑known public figures** with fields
  {name, gender, category, mbti, summary}.
* Safety belts against hallucination:
  1. Candidate must have an **English Wikipedia article**.
  2. Gender is fetched from **Wikidata (P21)**; skip if missing.
  3. MBTI must be 4 letters and is cross‑checked locally.
* Prompts kept minimal; verification is done with public APIs.
* Output: UTF‑8 JSON array → characters.json
--------------------------------------------------
"""

# ───────────────────────── standard libs ─────────────────────────────────────
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

# ───────────────────────── third‑party libs ──────────────────────────────────
import requests  # for Wikipedia & Wikidata checks
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ───────────────────────── 0. client setup ───────────────────────────────────
load_dotenv()                                       # expects OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"                               # higher factuality vs 3.5

# ───────────────────────── 1. tweakables ─────────────────────────────────────
CATEGORIES = [
    "Actor", "Scientist", "Economist", "Politician", "Athlete",
    "Musician", "Author", "Entrepreneur"
]
EXCLUDE_WINDOW = 80        # send only the last 80 names in prompt
NAME_RETRIES = 3           # GPT retries
API_RETRIES = 3            # Wikipedia/Wikidata retries
SLEEP = 1.0                # seconds between OpenAI calls (CLI override)

# ───────────────────────── 2. GPT prompt templates ──────────────────────────
GET_NAME_SYS = (
    "You are a strict JSON generator. Respond ONLY with JSON."
)
GET_NAME_USER = (
    "Return ONE *widely‑known* public figure whose English Wikipedia page *exists*, "
    "not listed here: {taken}. Respond as {{\"name\":str, \"category\":str}} "
    "where category ∈ {cats}. *Do not explain or add fields*."
)

GET_PERS_SYS = (
    "You are a data generator. Respond ONLY with JSON matching "
    "{\\\"mbti\\\":\"XXXX\",\\\"summary\\\":\"...\"}."
)
GET_PERS_USER = (
    "Provide:\n• mbti → four‑letter MBTI type\n• summary → 2 paragraphs, 80‑120 words total.\n\nName: {name}\nField: {field}"
)

# ───────────────────────── 3. helper: OpenAI chat call ───────────────────────

def chat(msgs, *, temperature=0.3, max_tokens=200, response_format=None):
    """Send a chat completion and return *content* of first choice."""
    return client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format or {"type": "json_object"},
    ).choices[0].message.content.strip()

# ───────────────────────── 4. verification helpers ──────────────────────────

WIKI_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"


def wikipedia_exists(title: str) -> bool:
    """True iff an English Wikipedia page for *title* exists (exact match)."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
    }
    for _ in range(API_RETRIES):
        try:
            resp = requests.get(WIKI_SEARCH, params=params, timeout=5).json()
            pages = resp.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            return "missing" not in page
        except requests.RequestException:
            time.sleep(0.8)
    return False


def wikidata_gender(title: str) -> str | None:
    """Return 'male'/'female'/'other' using Wikidata P21, or None."""
    # 1) Get Wikibase item id
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "pageprops",
        "ppprop": "wikibase_item",
    }
    try:
        resp = requests.get(WIKI_SEARCH, params=params, timeout=5).json()
        pages = resp["query"]["pages"]
        qid = next(iter(pages.values()))["pageprops"]["wikibase_item"]
    except Exception:
        return None

    # 2) Fetch entity data
    url = WIKIDATA_ENTITY.format(qid=qid)
    try:
        data = requests.get(url, timeout=5).json()
        claims = data["entities"][qid]["claims"].get("P21")
        if not claims:
            return None
        target_id = claims[0]["mainsnak"]["datavalue"]["value"]["id"]
        # Human genders: Q6581097 male, Q6581072 female, else → other
        if target_id == "Q6581097":
            return "male"
        if target_id == "Q6581072":
            return "female"
        return "other"
    except Exception:
        return None


MBTI_RE = re.compile(r"^[EINFSNTJP]{4}$", re.I)


# ───────────────────────── 5. GPT wrapper: get ONE candidate ────────────────

def fetch_one_name(exclude: List[str]) -> Dict[str, str]:
    """Ask GPT for one candidate not in *exclude*; enforce JSON output."""
    taken_fragment = ", ".join(exclude) if exclude else "NONE"
    user_msg = GET_NAME_USER.format(taken=taken_fragment, cats=", ".join(CATEGORIES))
    raw = chat([
        {"role": "system", "content": GET_NAME_SYS},
        {"role": "user", "content": user_msg},
    ], max_tokens=120)
    return json.loads(raw)


# ───────────────────────── 6. GPT wrapper: get persona data ─────────────────

def fetch_persona(name: str, cat: str) -> Dict[str, str]:
    """Return mbti+summary (validated) or raise."""
    raw = chat([
        {"role": "system", "content": GET_PERS_SYS},
        {"role": "user", "content": GET_PERS_USER.format(name=name, field=cat)},
    ], max_tokens=220)
    data = json.loads(raw)
    mbti = data.get("mbti", "").upper()
    if not MBTI_RE.match(mbti):
        raise ValueError("bad MBTI format")
    return {"mbti": mbti, "summary": data["summary"].strip()}


# ───────────────────────── 7. main routine ──────────────────────────────────

def generate(outfile: str, n: int, pause: float):
    """Generate *n* unique, verified personas and dump JSON."""
    print(f"▶ Generating {n} verified personas …")

    collected: Dict[str, Dict[str, str]] = {}  # name → {'category':..,'gender':..}

    # 7‑A) acquire unique, verified names
    while len(collected) < n:
        window = list(collected.keys())[-EXCLUDE_WINDOW:]
        for attempt in range(1, NAME_RETRIES + 1):
            try:
                cand = fetch_one_name(window)
                name, cat = cand.get("name"), cand.get("category")
                if not name or cat not in CATEGORIES:
                    raise ValueError("invalid JSON fields")
                if name in collected:
                    raise ValueError("duplicate name")

                # verify Wikipedia presence
                if not wikipedia_exists(name):
                    raise ValueError("no Wikipedia page")

                # get gender via Wikidata
                gender = wikidata_gender(name)
                if gender is None:
                    raise ValueError("gender unknown")

                # store and break out of retry loop
                collected[name] = {"category": cat, "gender": gender}
                print(f"  + {name} ✓ ({len(collected)}/{n})")
                break
            except Exception as e:
                print(f"[name retry {attempt}] {e}")
                time.sleep(1.5 * attempt)
        time.sleep(pause)

    # 7‑B) fetch persona details
    rows: List[Dict[str, str]] = []
    for idx, (name, meta) in enumerate(collected.items(), 1):
        for attempt in range(1, 4):
            try:
                pdata = fetch_persona(name, meta["category"])
                row = {
                    "name": name,
                    "gender": meta["gender"],
                    "category": meta["category"],
                    "mbti": pdata["mbti"],
                    "summary": pdata["summary"],
                    "image_url": ""  # placeholder for future work
                }
                rows.append(row)
                if idx % 10 == 0 or idx == n:
                    print(f"  fetched {idx}/{n}")
                break
            except Exception as e:
                print(f"[persona retry {attempt}] {name}: {e}")
                time.sleep(2 * attempt)
        time.sleep(pause)

    # 7‑C) write UTF‑8 JSON
    Path(outfile).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✔ Saved {n} personas → {outfile}")


# ───────────────────────── 8. CLI wrapper ───────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="characters.json", help="output file")
    parser.add_argument("--n", type=int, default=50, help="number of personas")
    parser.add_argument("--pause", type=float, default=SLEEP, help="sleep between API calls")
    args = parser.parse_args()

    generate(args.out, args.n, args.pause)



####################################################################

#!/usr/bin/env python3
# =============================================================================
# # generate_characters_mbti.py       (robust, no-dup, any N)
# # -----------------------------------------------------------------------------
# # • Loop until we have N UNIQUE famous figures {name, category}
# #   – ask GPT-3.5-turbo for ONE new figure each time
# #   – pass only the last 100 collected names as “exclude” to keep prompt tiny
# # • For each figure fetch {"mbti","summary"} in strict-JSON mode
# # • Write characters.json (UTF-8)
# # =============================================================================

# # -------- standard libs ------------------------------------------------------
# import argparse, json, os, time
# from pathlib import Path
# from typing  import Dict, List

# # -------- third-party --------------------------------------------------------
# from dotenv import load_dotenv
# from openai import OpenAI, OpenAIError

# # -------- 0. client & cheap model -------------------------------------------
# load_dotenv()                                           # reads OPENAI_API_KEY
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))    # one client object
# MODEL  = "gpt-3.5-turbo-0125"                           # cost-effective

# # -------- 1. constants you can tweak ----------------------------------------
# CATEGORIES       = ["Anime", "Movie", "Scientist", "Economist", "Politician"]
# EXCLUDE_WINDOW   = 100       # send at most the last 100 names to GPT
# PERSONA_RETRIES  = 3
# NAME_RETRIES     = 3
# SLEEP            = 1.0       # seconds between API calls (overridden by CLI)

# # -------- 2. prompt templates -----------------------------------------------
# GET_NAME_SYS = "Return ONLY valid JSON."
# GET_NAME_USER = (
#     "Return ONE famous person not already listed here: {taken}.\n"
#     "Respond as {{\"name\": str, \"category\": str}} where category ∈ {cats}."
# )

# GET_PERS_SYS = (
#     "You are a data generator. Respond ONLY with JSON "
#     "matching {{\"mbti\":\"XXXX\",\"summary\":\"...\"}}."
# )
# GET_PERS_USER = (
#     "Give me:\n"
#     "• mbti    → MBTI four-letter type\n"
#     "• summary → TWO paragraphs (80-120 words TOTAL) "
#     "describing their public personality, values, background.\n\n"
#     "Name: {name}\nField: {field}"
# )

# # -------- 3. tiny wrapper for chat calls ------------------------------------
# def chat(msgs, **kw) -> str:
#     """Returns content of first choice (stripped)."""
#     return client.chat.completions.create(
#         model       = MODEL,
#         messages    = msgs,
#         temperature = kw.get("temperature", 0.35),
#         max_tokens  = kw.get("max_tokens", 400),
#         response_format = kw.get("response_format")
#     ).choices[0].message.content.strip()

# # -------- 4. fetch ONE unique name ------------------------------------------
# def fetch_one_name(exclude: List[str]) -> Dict[str, str]:
#     """Ask GPT for a name/category not in exclude list."""
#     taken_fragment = ", ".join(exclude) if exclude else "NONE"
#     prompt_user = GET_NAME_USER.format(
#         taken=taken_fragment,
#         cats=", ".join(CATEGORIES)
#     )
#     raw = chat(
#         [{"role":"system","content":GET_NAME_SYS},
#          {"role":"user", "content":prompt_user}],
#         max_tokens=120
#     )
#     return json.loads(raw)    # may raise JSONDecodeError

# # -------- 5. fetch mbti + summary for ONE person ----------------------------
# def fetch_persona(name: str, cat: str) -> Dict[str, str]:
#     """Retry PERSONA_RETRIES times then fallback."""
#     delay = 1.5
#     for attempt in range(1, PERSONA_RETRIES+1):
#         try:
#             raw = chat(
#                 [{"role":"system","content":GET_PERS_SYS},
#                  {"role":"user",  "content":GET_PERS_USER.format(name=name, field=cat)}],
#                 max_tokens=220,
#                 response_format={"type":"json_object"}
#             )
#             js = json.loads(raw)
#             if len(js.get("mbti","")) == 4 and len(js.get("summary","")) > 30:
#                 return js
#             raise ValueError("missing keys")
#         except (json.JSONDecodeError, OpenAIError, ValueError) as e:
#             print(f"[persona retry {attempt}] {name}: {e}")
#             time.sleep(delay); delay *= 2
#     # generic fallback
#     return {"mbti":"INTP","summary":"Interesting public figure."}

# # -------- 6. main loop ------------------------------------------------------
# def main(outfile: str, n: int, pause: float):
#     print(f"▶ Generating {n} unique personas with {MODEL} …")

#     collected: Dict[str, str] = {}      # name → category
#     # 6-A  get unique names
#     while len(collected) < n:
#         window = list(collected.keys())[-EXCLUDE_WINDOW:]
#         for attempt in range(1, NAME_RETRIES+1):
#             try:
#                 obj = fetch_one_name(window)
#                 # validate obj
#                 if (isinstance(obj, dict) and
#                     obj.get("category") in CATEGORIES and
#                     obj.get("name")):
#                     if obj["name"] not in collected:
#                         collected[obj["name"]] = obj["category"]
#                         print(f"  + {obj['name']} ({len(collected)}/{n})")
#                         break
#                 raise ValueError("invalid or duplicate")
#             except Exception as e:
#                 print(f"[name retry {attempt}] {e}")
#                 time.sleep(1.5 * (2**(attempt-1)))
#         time.sleep(pause)

#     # 6-B  for each name get mbti+summary
#     rows = []
#     for idx, (name, cat) in enumerate(collected.items(), 1):
#         pdata = fetch_persona(name, cat)
#         rows.append({
#             "name":      name,
#             "category":  cat,
#             "mbti":      pdata["mbti"].upper(),
#             "summary":   pdata["summary"].strip(),
#             "image_url": ""
#         })
#         if idx % 10 == 0 or idx == n:
#             print(f"  done {idx}/{n}")
#         time.sleep(pause)

#     # 6-C  write UTF-8 JSON
#     Path(outfile).write_text(
#         json.dumps(rows, indent=2, ensure_ascii=False),
#         encoding="utf-8"
#     )
#     print(f"✔ Saved {n} personas → {outfile}")

# # -------- 7. CLI wrapper ----------------------------------------------------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out",   default="characters.json", help="output file")
#     ap.add_argument("--n",     type=int, default=60,      help="how many personas")
#     ap.add_argument("--pause", type=float, default=SLEEP, help="seconds between API calls")
#     args = ap.parse_args()
#     main(args.out, args.n, args.pause)