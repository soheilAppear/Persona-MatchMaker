#!/usr/bin/env python3
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  generate_characters_from_csv.py  ·  v3.4                                 ║
# ║  Reads your CSV → calls GPT-3.5 *only* for missing info → writes JSON.    ║
# ║                                                                           ║
# ║  Usage example (PowerShell / CMD):                                        ║
# ║    python generate_characters_from_csv.py ^                               ║
# ║           --csv "Unique_Famous_People_with_Personality_Types.csv" ^       ║
# ║           --out characters.json ^                                         ║
# ║           --pause 1.2                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ────────────────────────────── 1. std-lib imports ──────────────────────────
import csv, json, os, sys, time          # CSV reading, JSON writing, timing
from pathlib import Path                 # convenient file-path handling
from typing import Dict, List            # type hints (optional but useful)

# ───────────────────────────── 2. third-party imports ───────────────────────
from dotenv import load_dotenv           # pip install python-dotenv
from openai import OpenAI, OpenAIError    # pip install openai

# ───────────────────────────── 3. OpenAI client setup ───────────────────────
load_dotenv()                            # reads .env so OPENAI_API_KEY is in env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = "gpt-3.5-turbo-0125"            # cost-effective chat model

# ───────────────────────────── 4. GPT helper with retries ───────────────────
def ask_gpt(system: str,
            user: str,
            max_toks: int = 220,
            temp: float = 0.4) -> str:
    """
    One call to GPT with **JSON-only** output enforced.
    Retries up to 3 times with exponential back-off on errors or bad JSON.
    """
    backoff = 1.5                        # initial sleep if we need to retry
    for attempt in range(1, 4):          # 1, 2, 3
        try:
            reply = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role":"system", "content": system},
                    {"role":"user",   "content": user}
                ],
                temperature=temp,
                max_tokens=max_toks,
                response_format={"type": "json_object"}  # guarantees JSON
            ).choices[0].message.content
            return reply                # success
        except (OpenAIError, ValueError) as e:
            print(f"[gpt-retry {attempt}] {e}")
            time.sleep(backoff)          # wait before next try
            backoff *= 2                # exponential back-off
    raise RuntimeError("GPT failed 3 attempts in a row")

# ───────────────────────────── 5. Prompt templates ──────────────────────────
SYS_MSG = ("You are a data generator. Respond ONLY with JSON like "
           '{"mbti":"XXXX","summary":"..."} .')

def build_user_prompt(name: str, gender: str, need_mbti: bool) -> str:
    """
    Creates the *user* prompt. If we already know MBTI, we ask only for summary.
    """
    if need_mbti:
        req = ("Return:\n"
               "  mbti    → 4-letter MBTI type (caps)\n"
               "  summary → TWO paragraphs, 80-120 words TOTAL.")
    else:
        req = ('MBTI already known. Return ONLY '
               '{"summary":"TWO paragraphs, 80-120 words TOTAL"}')
    return f"{req}\n\nName: {name}\nGender: {gender}"

# ───────────────────────────── 6. CSV reader (your headers) ─────────────────
def read_csv(csv_path: str) -> List[Dict[str, str]]:
    """
    Reads a CSV whose header line is:
        Name, Gender, PersonalityType
    • Whitespace is stripped.
    • Duplicate names (case-insensitive) are dropped (first occurrence wins).
    Returns: list of dicts → {"name", "category", "mbti"}.
    """
    rows: List[Dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        # auto-detect delimiter just in case (comma / semicolon / tab)
        dialect = csv.Sniffer().sniff(f.read(1024))
        f.seek(0)
        for raw in csv.DictReader(f, dialect=dialect):
            rows.append({
                "name":     raw.get("Name", "").strip(),
                "category": raw.get("Gender", "").strip() or "Unknown",
                "mbti":     (raw.get("PersonalityType", "") or "").upper().strip()
            })
    # deduplicate by lower-case name
    uniq: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row["name"]:                  # skip completely empty rows
            uniq.setdefault(row["name"].lower(), row)
    return list(uniq.values())

# ───────────────────────────── 7. JSON reader (existing data) ───────────────
def read_existing(json_path: str = "characters.json") -> Dict[str, Dict]:
    """
    Loads existing characters.json if it exists and returns a dict keyed
    by lower-case name for quick look-up / de-duplication.
    """
    if Path(json_path).exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return {d["name"].lower(): d for d in data}
    return {}

# ───────────────────────────── 8. Main generation routine ───────────────────
def generate(csv_path: str,
             out_path: str,
             pause: float):
    """
    • Reads CSV → list of seeds
    • Skips any names already present in characters.json
    • Calls GPT only for the missing data
    • Writes / updates characters.json
    """
    seeds      = read_csv(csv_path)
    existing   = read_existing(out_path)
    todo       = [s for s in seeds if s["name"].lower() not in existing]

    print(f"▶ CSV unique names : {len(seeds)}")
    print(f"▶ Need GPT calls   : {len(todo)}")

    new_rows: List[Dict[str, str]] = []

    for idx, person in enumerate(todo, 1):
        name, gender, mbti = person["name"], person["category"], person["mbti"]
        need_mbti          = (mbti == "")

        # ── GPT call (cheap, with retries) ─────────────────────────────────
        try:
            raw_json = ask_gpt(
                SYS_MSG,
                build_user_prompt(name, gender, need_mbti)
            )
            data = json.loads(raw_json)
        except Exception as e:
            # Fallback if GPT fails or JSON parse error
            print(f"[warn] {name}: {e} → using fallback")
            data = {"mbti": mbti or "INTP",
                    "summary": "Interesting public figure."}

        # Final MBTI: prefer CSV value; else from GPT; default INTP
        mbti_final = (mbti or data.get("mbti", "INTP")).upper()

        new_rows.append({
            "name":      name,
            "category":  gender,
            "mbti":      mbti_final,
            "summary":   data["summary"].strip(),
            "image_url": ""
        })

        print(f"  ✓ {idx}/{len(todo)} – {name}")
        time.sleep(pause)               # gentle on rate-limits

    # ── Merge old + new, sort alphabetically for readability ───────────────
    merged = list(existing.values()) + new_rows
    merged.sort(key=lambda r: r["name"])

    # ── Save JSON in UTF-8 (no ASCII escapes) ──────────────────────────────
    Path(out_path).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✔ Saved {len(merged)} total records → {out_path}")

# ───────────────────────────── 9. CLI wrapper ───────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate / update characters.json from a Name/Gender/MBTI CSV")
    ap.add_argument(
        "--csv",
        default="Unique_Famous_People_with_Personality_Types.csv",
        help="input CSV path (default: file in current directory)")
    ap.add_argument(
        "--out", default="characters.json",
        help="output JSON file (default: characters.json)")
    ap.add_argument(
        "--pause", type=float, default=1.2,
        help="seconds to sleep between GPT calls (default 1.2)")
    args = ap.parse_args()

    try:
        generate(args.csv, args.out, args.pause)
    except KeyboardInterrupt:
        sys.exit("\n⏹️  Interrupted by user")






###########################################################################################

# #!/usr/bin/env python3
# # =============================================================================
# # generate_characters_mbti.py   ·   UNIQUE-NAME JSON GENERATOR v2
# # -----------------------------------------------------------------------------
# # 1. Repeatedly asks GPT-4o-mini for CHUNK-sized batches of famous figures
# #    (name + category) until we collect n UNIQUE entries.
# # 2. For each entry fetches {"mbti": ..., "summary": two-paragraphs}.
# # 3. Writes UTF-8 characters.json with no duplicates.
# # =============================================================================

# import argparse, json, os, time
# from pathlib import Path
# from typing import List, Dict
# from dotenv import load_dotenv
# from openai import OpenAI, OpenAIError

# # ─────────────────────────────── 0.  OpenAI client ───────────────────────────
# load_dotenv()                                         # expects .env with OPENAI_API_KEY
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ─────────────────────────────── 1. Categories list ──────────────────────────
# CATEGORIES: List[str] = ["Anime", "Movie", "Scientist", "Economist", "Politician"]

# # Size of each seed request; keep small so reply fits comfortably in 600 tokens
# CHUNK = 50

# # ─────────────────────────────── 2. Prompt templates ─────────────────────────
# SEED_PROMPT = (
#     "Return ONLY a JSON array of length {k}. "
#     "Each element must be {{\"name\": str, \"category\": str}}. "
#     "Category MUST be one of: {cats}. "
#     "All names must be famous, internationally recognisable, and **distinct**."
# )

# SYSTEM_PERSONA = (
#     "You are a data generator. Respond only with strict JSON "
#     "matching {\"mbti\":\"XXXX\",\"summary\":\"...\"}"
# )

# USER_PERSONA = (
#     "Produce:\n"
#     "• mbti    → MBTI four-letter type\n"
#     "• summary → TWO paragraphs, 80-120 words TOTAL describing public personality\n\n"
#     "Name: {name}\nField: {field}"
# )

# # ───────────────────── 3. Fetch one chunk of seed names ──────────────────────
# def fetch_seed_chunk(k: int) -> List[Dict[str, str]]:
#     """
#     Ask GPT for k unique {name, category} objects.
#     *No retries here* – higher-level function handles retries.
#     """
#     prompt = SEED_PROMPT.format(k=k, cats=", ".join(CATEGORIES))
#     rsp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.4,
#         max_tokens=max(200, k * 12)      # room for JSON text
#     ).choices[0].message.content.strip()
#     return json.loads(rsp)               # may raise JSONDecodeError

# # ──────────────────── 4. Collect n unique seeds via chunks ───────────────────
# def collect_unique_seeds(n: int, tries: int = 3) -> List[Dict[str, str]]:
#     """
#     Keep requesting CHUNK-sized lists until we have n DISTINCT names.
#     Retries each chunk up to `tries` times on format errors.
#     """
#     collected: Dict[str, Dict] = {}      # key = name → {name,category}
#     while len(collected) < n:
#         need = min(CHUNK, n - len(collected))
#         for attempt in range(1, tries + 1):
#             try:
#                 chunk = fetch_seed_chunk(need)
#                 # Basic validation
#                 assert isinstance(chunk, list) and len(chunk) == need
#                 for obj in chunk:
#                     assert (
#                         isinstance(obj, dict)
#                         and "name" in obj
#                         and "category" in obj
#                         and obj["category"] in CATEGORIES
#                     )
#                     collected.setdefault(obj["name"], obj)   # dedupe by name
#                 break                                        # chunk accepted
#             except Exception as e:
#                 print(f"[seed-chunk retry {attempt}] {e}")
#                 time.sleep(1.5 * (2 ** (attempt - 1)))
#         else:
#             raise RuntimeError("failed to get valid seed chunk")
#     return list(collected.values())[:n]                      # truncate if over

# # ──────────────────── 5. Fetch MBTI + summary for one figure ─────────────────
# def fetch_persona(name: str, field: str, tries: int = 3) -> Dict[str, str]:
#     """Retries up to `tries` until GPT returns valid JSON."""
#     backoff = 1.5
#     for attempt in range(1, tries + 1):
#         try:
#             rsp = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": SYSTEM_PERSONA},
#                     {"role": "user",   "content": USER_PERSONA.format(name=name, field=field)}
#                 ],
#                 temperature=0.4,
#                 max_tokens=220,
#                 response_format={"type": "json_object"}
#             ).choices[0].message.content
#             data = json.loads(rsp)
#             if len(data.get("mbti", "")) == 4 and len(data.get("summary", "")) > 30:
#                 return data
#             raise ValueError("missing fields")
#         except (json.JSONDecodeError, OpenAIError, ValueError) as e:
#             print(f"[persona retry {attempt}] {name}: {e}")
#             time.sleep(backoff); backoff *= 2
#     raise RuntimeError(f"no valid persona for {name}")

# # ───────────────────────────── 6. Main driver ────────────────────────────────
# def main(out_file: str, n: int, pause: float):
#     print(f"Collecting {n} unique famous figures …")
#     seeds = collect_unique_seeds(n)                      # step 1 complete

#     rows = []                                           # final records
#     for idx, entry in enumerate(seeds, 1):
#         name, cat = entry["name"], entry["category"]
#         try:
#             persona = fetch_persona(name, cat)
#         except Exception as err:
#             print(f"[warn] {name}: {err} — fallback")
#             persona = {"mbti": "INTP", "summary": "Interesting public figure."}

#         rows.append({
#             "name":       name,
#             "category":   cat,
#             "mbti":       persona["mbti"].upper(),
#             "summary":    persona["summary"].strip(),
#             "image_url":  ""
#         })

#         if idx % 10 == 0 or idx == n:
#             print(f"  completed {idx}/{n}")
#         time.sleep(pause)                               # avoid rate spikes

#     # Write UTF-8 JSON so Streamlit reads without codec errors
#     Path(out_file).write_text(
#         json.dumps(rows, indent=2, ensure_ascii=False),
#         encoding="utf-8"
#     )
#     print(f"✔ Saved {n} unique personas → {out_file}")

# # ───────────────────────────── 7. CLI wrapper ────────────────────────────────
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--out",   default="characters.json", help="output file")
#     parser.add_argument("--n",     type=int, default=60,      help="number of personas")
#     parser.add_argument("--pause", type=float, default=1.2,   help="pause between GPT calls")
#     args = parser.parse_args()
#     main(args.out, args.n, args.pause)

######################################################################

# #!/usr/bin/env python3
# """
# generate_characters_mbti.py  (robust JSON version)
# --------------------------------------------------
# Creates characters.json with MBTI + 2-paragraph summaries.

# Run:
#     python generate_characters_mbti.py --out characters.json --n 60
# """

# import argparse, json, os, random, time
# from pathlib import Path
# from typing import Dict, List
# from dotenv import load_dotenv
# from openai import OpenAI, OpenAIError

# # ── 0. API key ───────────────────────────────────────────────────────────
# load_dotenv()                                       # expects .env with OPENAI_API_KEY
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ── 1. Seed names ────────────────────────────────────────────────────────
# CATEGORIES: Dict[str, List[str]] = {
#     "Anime":    ["Light Yagami", "Naruto Uzumaki", "Mikasa Ackerman"],
#     "Movie":    ["Luke Skywalker", "Hermione Granger", "Tony Stark"],
#     "Scientist": ["Albert Einstein", "Marie Curie", "Alan Turing"],
#     "Economist": ["Adam Smith", "John Maynard Keynes", "Milton Friedman"],
#     "Politician": ["Barack Obama", "Nelson Mandela", "Winston Churchill"],
# }

# # ── 2. Prompt parts ──────────────────────────────────────────────────────
# SYSTEM_MSG = (
#     "You are a data generator. "
#     "Always respond with STRICT JSON matching the schema "
#     '{"mbti": "...", "summary": "..."}'
# )
# USER_TMPL = (
#     "Give me:\n"
#     "  mbti    → one of the 16 MBTI types (4 letters, all caps)\n"
#     "  summary → TWO paragraphs, 80-120 words TOTAL, "
#     "describing public personality, values & background.\n\n"
#     "Name: {name}\nField: {field}"
# )

# # ── 3. Helper to call GPT with retries ────────────────────────────────────
# def fetch_persona(name: str, field: str, tries: int = 3) -> dict:
#     delay = 1.5
#     for attempt in range(tries):
#         try:
#             rsp = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": SYSTEM_MSG},
#                     {"role": "user",   "content": USER_TMPL.format(name=name, field=field)}
#                 ],
#                 temperature=0.4,
#                 max_tokens=220,
#                 response_format={"type": "json_object"},  # <- force JSON
#             ).choices[0].message.content
#             data = json.loads(rsp)
#             if len(data["mbti"]) == 4 and len(data["summary"]) > 30:
#                 return data
#         except (json.JSONDecodeError, KeyError, OpenAIError) as e:
#             print(f"[retry {attempt+1}] {name}: {e}")
#             time.sleep(delay)
#             delay *= 2          # exponential back-off
#     raise RuntimeError("No valid JSON after retries")

# # ── 4. Main ---------------------------------------------------------------
# def main(out: str, n: int, pause: float):
#     pool = [{"name": nm, "category": cat}
#             for cat, lst in CATEGORIES.items() for nm in lst]
#     if n > len(pool):
#         pool *= (n + len(pool) - 1) // len(pool)
#     pool = pool[:n]
#     random.shuffle(pool)

#     rows = []
#     for idx, row in enumerate(pool, 1):
#         try:
#             data = fetch_persona(row["name"], row["category"])
#         except Exception as err:
#             print(f"[warn] {row['name']} fallback → {err}")
#             data = {"mbti": "INTP", "summary": "Interesting public figure."}

#         rows.append({
#             **row,
#             "mbti": data["mbti"].upper(),
#             "summary": data["summary"].strip(),
#             "image_url": ""
#         })
#         if idx % 10 == 0:
#             print(f"generated {idx}/{n}")
#         time.sleep(pause)               # stay under rate-limit

#     Path(out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
#     print("✔ Saved →", out)

# # ── 5. CLI ---------------------------------------------------------------
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--out", default="characters.json")
#     p.add_argument("--n", type=int, default=60)
#     p.add_argument("--pause", type=float, default=1.2)
#     main(**vars(p.parse_args()))
