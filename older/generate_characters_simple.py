#!/usr/bin/env python3
"""
Generate characters.json with a richer 3-sentence blurb for each figure.
Cost: 3×40 ≈ 120 tokens per name ≈ $0.00007 using gpt-4o-mini.
"""

import argparse, itertools, json, random, time, os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CATEGORIES = {
    "Anime": ["Light Yagami", "Naruto Uzumaki", "Mikasa Ackerman"],
    "Movie": ["Luke Skywalker", "Hermione Granger", "Tony Stark"],
    "Scientist": ["Albert Einstein", "Marie Curie", "Alan Turing"],
    # … add more names or read from a CSV
}

PROMPT = ("Write THREE sentences (≤70 words total) that capture the public "
          "personality of {name}, a famous {field} figure. Respond with the "
          "three sentences only.")

def fetch_blurb(name, field):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":PROMPT.format(name=name, field=field)}],
        temperature=0.5,
        max_tokens=90
    ).choices[0].message.content.strip()

def main(out, n):
    pool = [dict(name=n, category=c) for c,L in CATEGORIES.items() for n in L]
    random.shuffle(pool)
    data=[]
    for item in pool[:n]:
        try:
            desc = fetch_blurb(item["name"], item["category"])
        except Exception as e:
            print("warn:",e); desc="Interesting public figure."
        data.append({**item, "description":desc, "image_url":""})
        time.sleep(0.8)           # stay well under rate limit
    Path(out).write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print("Saved", out)


    

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="characters.json")
    ap.add_argument("--n",type=int,default=60)
    main(**vars(ap.parse_args()))



# #!/usr/bin/env python3
# """
# Generate a lightweight characters.json for the simple Persona-Matcher.

# • For each (name, category) it asks GPT for ONE concise personality sentence.
# • Outputs: [{name, category, description, image_url:""}]

# Usage
# -----
# export OPENAI_API_KEY="sk-..."
# python generate_characters_simple.py --out characters.json --n 200
# """
# import argparse, itertools, json, os, random, time
# from pathlib import Path
# from typing import List, Dict
# from openai import OpenAI

# from dotenv import load_dotenv   # ← ADD
# load_dotenv()                    # ← ADD (right after imports)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # ── Seed lists ───────────────────────────────────────────────────────────
# CATEGORIES: Dict[str, List[str]] = {
#     "Anime": [
#         "Naruto Uzumaki", "Monkey D. Luffy", "Goku", "Sailor Moon",
#         "Light Yagami", "Mikasa Ackerman", "Tanjiro Kamado", "Spike Spiegel",
#         "Edward Elric", "Levi Ackerman"
#     ],
#     "Movie": [
#         "Luke Skywalker", "Indiana Jones", "Ellen Ripley", "Forrest Gump",
#         "Darth Vader", "Hermione Granger", "Tony Stark", "Black Panther",
#         "Wonder Woman", "Rocky Balboa"
#     ],
#     "Scientist": [
#         "Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin",
#         "Nikola Tesla", "Alan Turing", "Ada Lovelace", "Carl Sagan",
#         "Rosalind Franklin", "Katherine Johnson"
#     ],
#     "Economist": [
#         "Adam Smith", "John Maynard Keynes", "Milton Friedman", "Elinor Ostrom",
#         "Amartya Sen", "Paul Krugman", "Janet Yellen", "Thomas Piketty",
#         "Friedrich Hayek", "David Ricardo"
#     ],
#     "Politician": [
#         "Barack Obama", "Angela Merkel", "Nelson Mandela", "Winston Churchill",
#         "Abraham Lincoln", "Theodore Roosevelt", "Mahatma Gandhi",
#         "Margaret Thatcher", "Franklin D. Roosevelt", "Jacinda Ardern"
#     ],
# }

# # Flatten into [{name, category}, …]
# ALL_NAMES = [
#     {"name": n, "category": cat} for cat, names in CATEGORIES.items() for n in names
# ]

# # ── GPT helper ────────────────────────────────────────────────────────────
# PROMPT = (
#     "Write ONE vivid sentence (≤25 words) that captures the public personality "
#     "of the following famous figure.\n\nName: {name}\nField: {field}"
# )

# client = OpenAI()

# def fetch_description(name: str, field: str) -> str:
#     """Ask GPT-4o-mini for a single personality sentence."""
#     out = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": PROMPT.format(name=name, field=field)}],
#         temperature=0.4,
#         max_tokens=50
#     ).choices[0].message.content.strip()
#     return out.rstrip(".") + "."  # ensure period

# # ── CLI ───────────────────────────────────────────────────────────────────
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", default="characters.json")
#     ap.add_argument("--n", type=int, default=200)
#     ap.add_argument("--delay", type=float, default=0.8,
#                     help="seconds between API calls")
#     args = ap.parse_args()

#     pool = list(itertools.islice(itertools.cycle(ALL_NAMES), args.n))
#     random.shuffle(pool)

#     results = []
#     for idx, item in enumerate(pool, 1):
#         try:
#             desc = fetch_description(item["name"], item["category"])
#         except Exception as e:
#             print("[warn]", e, "→ using placeholder")
#             desc = "Interesting public figure."
#         results.append({
#             "name":        item["name"],
#             "category":    item["category"],
#             "description": desc,
#             "image_url":   ""
#         })
#         if idx % 10 == 0:
#             print(f"Built {idx}/{args.n}")
#         time.sleep(args.delay)

#     Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
#     print("Saved ⇒", args.out)

# if __name__ == "__main__":
#     main()
