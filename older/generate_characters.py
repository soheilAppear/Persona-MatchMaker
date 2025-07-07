"""Generate a 1 000‑character personality database for Persona‑Matcher.

Prerequisites
-------------
1.  `pip install -r requirements.txt`
2.  Set `OPENAI_API_KEY` env var (or pass via CLI).

Usage
-----
python generate_characters.py --out characters.json --n 1000
"""
import argparse, json, os, random, time, itertools
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

# ─── Helpers ──────────────────────────────────────────────────────────────
CATEGORIES = {
    "Anime": [
        "Naruto Uzumaki", "Monkey D. Luffy", "Goku", "Sailor Moon", "Light Yagami",
        "Mikasa Ackerman", "Tanjiro Kamado", "Spike Spiegel", "Edward Elric", "Levi Ackerman",
    ],
    "Movie": [
        "Luke Skywalker", "Indiana Jones", "Ellen Ripley", "Forrest Gump", "Darth Vader",
        "Hermione Granger", "Tony Stark", "Black Panther", "Wonder Woman", "Rocky Balboa",
    ],
    "Scientist": [
        "Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin", "Nikola Tesla",
        "Alan Turing", "Ada Lovelace", "Carl Sagan", "Rosalind Franklin", "Katherine Johnson",
    ],
    "Economist": [
        "Adam Smith", "John Maynard Keynes", "Milton Friedman", "Elinor Ostrom", "Amartya Sen",
        "Paul Krugman", "Janet Yellen", "Thomas Piketty", "Friedrich Hayek", "David Ricardo",
    ],
    "Politician": [
        "Barack Obama", "Angela Merkel", "Nelson Mandela", "Winston Churchill", "Abraham Lincoln",
        "Theodore Roosevelt", "Mahatma Gandhi", "Margaret Thatcher", "Franklin D. Roosevelt", "Jacinda Ardern",
    ],
}

# Expand lists to reach N names by repeating & shuffling
ALL_NAMES: List[Dict] = []
for cat, names in CATEGORIES.items():
    ALL_NAMES += [{"name": n, "category": cat} for n in names]

# ─── OpenAI utilities ─────────────────────────────────────────────────────
client: OpenAI | None = None
SYSTEM_PROMPT = (
    "You are an expert psychologist. Given a famous figure's name and their field, "
    "return only JSON with their Big‑Five personality scores (keys: O,C,E,A,N, 1‑5 floats), "
    "and a one‑sentence description in 'bio'."
)


def fetch_vector(full_name: str, category: str) -> Dict:
    global client
    if client is None:
        client = OpenAI()
    user_msg = f"Name: {full_name}\nField: {category}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_msg}],
        temperature=0.2,
        max_tokens=250,
    ).choices[0].message.content.strip()
    data = json.loads(resp)
    return {
        "name": full_name,
        "category": category,
        "description": data.pop("bio"),
        "image_url": "",  # you can fill later
        "personality_vector": data,
    }

# ─── Main CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="characters.json")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=1.1, help="seconds between requests")
    args = parser.parse_args()

    # sample or cycle names to reach N
    pool = list(itertools.islice(itertools.cycle(ALL_NAMES), args.n))
    random.shuffle(pool)

    results = []
    for idx, entry in enumerate(pool, 1):
        try:
            results.append(fetch_vector(entry["name"], entry["category"]))
        except Exception as e:
            print("[warn]", e, "-> using placeholder scores")
            results.append({
                "name": entry["name"],
                "category": entry["category"],
                "description": "Personality placeholder.",
                "image_url": "",
                "personality_vector": {k: round(random.uniform(2,4),1) for k in "OCEAN"}
            })
        if idx % 10 == 0:
            print(f"Built {idx}/{args.n}")
        time.sleep(args.delay)

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("Saved", args.out)

if __name__ == "__main__":
    main()