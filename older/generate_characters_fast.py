# generate_characters_fast.py
"""
Create N character entries quickly with batched, concurrent GPT-4o-mini calls.
Usage:
  export OPENAI_API_KEY=...
  python generate_characters_fast.py --n 1000 --batch 8 --out characters.json
"""
import asyncio, json, random, argparse, itertools, os
from pathlib import Path
from openai import AsyncOpenAI

CATEGORIES = {
    "Anime": ["Naruto Uzumaki", "Goku", "Mikasa Ackerman", "Light Yagami", "Sailor Moon"],
    "Movie": ["Luke Skywalker", "Hermione Granger", "Indiana Jones", "Ellen Ripley", "Tony Stark"],
    "Scientist": ["Albert Einstein", "Marie Curie", "Alan Turing", "Isaac Newton", "Nikola Tesla"],
    "Economist": ["Adam Smith", "Elinor Ostrom", "Milton Friedman", "Amartya Sen", "Janet Yellen"],
    "Politician": ["Nelson Mandela", "Angela Merkel", "Barack Obama", "Winston Churchill", "Abraham Lincoln"],
}

SYS = (
    "Return ONLY valid JSON array. "
    "For each entry include name, short bio (≤25 words), "
    "and Big-Five scores 1–5 with keys O,C,E,A,N."
)

client = AsyncOpenAI()

def all_names(n: int):
    pool = [dict(name=n, category=c) for c, L in CATEGORIES.items() for n in L]
    return list(itertools.islice(itertools.cycle(pool), n))

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

async def fetch_batch(batch):
    prompt = "\n".join(f"{i+1}. {p['name']} ({p['category']})"
                       for i, p in enumerate(batch))
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": prompt},
    ]
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=350,
    )
    return json.loads(resp.choices[0].message.content)

async def main(n, batch_size, out):
    names = all_names(n)
    results = []
    sem = asyncio.Semaphore(8)          # limit concurrent calls

    async def worker(b):
        async with sem:
            data = await fetch_batch(b)
            for d in data:
                results.append({
                    "name": d["name"],
                    "category": next(p["category"] for p in b if p["name"] == d["name"]),
                    "description": d["bio"],
                    "image_url": "",
                    "personality_vector": {k: float(d[k]) for k in "OCEAN"},
                })

    await asyncio.gather(*(worker(b) for b in chunk(names, batch_size)))
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} characters → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="characters.json")
    args = ap.parse_args()
    asyncio.run(main(args.n, args.batch, args.out))
