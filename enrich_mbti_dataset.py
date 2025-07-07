#!/usr/bin/env python3
"""enrich_mbti_dataset.py
───────────────────────────────────────────────────────────────────────────────
Adds *gender*, *category*, and *subcategory* to rows where they are "unknown"
using Wikidata look‑ups.  Designed for the previously merged file produced in
this session (combined_mbti_unique.csv).

Usage:
    python enrich_mbti_dataset.py --infile combined_mbti_unique.csv \
                                   --outfile combined_mbti_enriched.csv

Requirements:
    pip install requests pandas tqdm
"""

# ───────────────────────────── standard libs ────────────────────────────────
import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

# ───────────────────────────── third‑party ──────────────────────────────────
import pandas as pd
import requests
from tqdm import tqdm  # progress bar

# ───────────────────────────── API endpoints ────────────────────────────────
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_ENTITY = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

# ───────────────────────────── helpers ──────────────────────────────────────

def get_wikidata_qid(title: str) -> Optional[str]:
    """Return the Wikidata Q‑ID for a given *exact* Wikipedia title."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "pageprops",
        "ppprop": "wikibase_item",
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=5).json()
        page = next(iter(r["query"]["pages"].values()))
        return page["pageprops"]["wikibase_item"]
    except Exception:
        return None


def get_entity_json(qid: str) -> Optional[dict]:
    """Fetch entity JSON from Wikidata."""
    try:
        url = WIKIDATA_ENTITY.format(qid=qid)
        return requests.get(url, timeout=5).json()["entities"][qid]
    except Exception:
        return None


# Q‑IDs → gender/category mapping
GENDER_MAP = {
    "Q6581097": "male",
    "Q6581072": "female",
}

# Some common occupations (P106) mapped to broad categories.
OCCUPATION_TO_CATEGORY: Dict[str, Dict[str, str]] = {
    "Q33999": {  # actor
        "category": "Actor",
        "subcategory": "Film/TV"
    },
    "Q947873": {  # film actor
        "category": "Actor",
        "subcategory": "Film"
    },
    "Q49757": {  # musician
        "category": "Musician",
        "subcategory": "General"
    },
    "Q937857": {  # singer‑songwriter
        "category": "Musician",
        "subcategory": "Singer‑Songwriter"
    },
    "Q901": {  # politician
        "category": "Politician",
        "subcategory": "General"
    },
    "Q28389": {  # scientist
        "category": "Scientist",
        "subcategory": "General"
    },
    "Q36180": {  # economist
        "category": "Economist",
        "subcategory": "General"
    },
    "Q937794": {  # entrepreneur
        "category": "Entrepreneur",
        "subcategory": "General"
    },
    # add more Q‑IDs as desired
}


OCCUPATION_RE = re.compile(r"P106")


def enrich_row(row: pd.Series, throttle: float = 0.4) -> pd.Series:
    """Fill unknown gender/category/subcat for a single DataFrame row."""
    if all(row[col] != "unknown" for col in ("gender", "category", "subcategory")):
        return row  # already enriched

    qid = get_wikidata_qid(row["name"])
    if not qid:
        return row

    entity = get_entity_json(qid)
    if not entity:
        return row

    # ── gender (P21) ────────────────────────────────────────────────────────
    if row["gender"] == "unknown":
        claims = entity["claims"].get("P21")
        if claims:
            target = claims[0]["mainsnak"]["datavalue"]["value"]["id"]
            row["gender"] = GENDER_MAP.get(target, "other")

    # ── occupation → category/subcategory ───────────────────────────────────
    if row["category"] == "unknown" or row["subcategory"] == "unknown":
        claims = entity["claims"].get("P106", [])  # occupation(s)
        for c in claims:
            occ_qid = c["mainsnak"]["datavalue"]["value"]["id"]
            if occ_qid in OCCUPATION_TO_CATEGORY:
                mapping = OCCUPATION_TO_CATEGORY[occ_qid]
                if row["category"] == "unknown":
                    row["category"] = mapping["category"]
                if row["subcategory"] == "unknown":
                    row["subcategory"] = mapping["subcategory"]
                break

    time.sleep(throttle)  # be kind to the API
    return row


# ───────────────────────────── main routine ────────────────────────────────

def main(infile: str, outfile: str):
    df = pd.read_csv(infile)

    # make sure required columns exist
    for col in ("name", "gender", "mbti", "category", "subcategory"):
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")

    print(f"▶ Enriching {len(df)} rows …")
    df_enriched = df.apply(enrich_row, axis=1)

    # write output
    df_enriched.to_csv(outfile, index=False, encoding="utf-8")
    print(f"✔ Saved updated dataset → {outfile}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  default="combined_mbti_unique.csv",  help="input CSV path")
    ap.add_argument("--outfile", default="combined_mbti_enriched.csv", help="output CSV path")
    args = ap.parse_args()

    main(args.infile, args.outfile)
