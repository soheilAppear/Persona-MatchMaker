
#!/usr/bin/env python3
"""rank_top250_by_pageviews.py — updated for reliability
───────────────────────────────────────────────────────────────────────────────
Improved version that fixes unreliability from missing or malformed Wikipedia
page titles. Uses the Wikipedia Search API to find the best-matching article
for each name before requesting view counts.

Requirements:
    pip install pandas requests tqdm

Usage:
    python rank_top250_by_pageviews.py \
        --infile combined_mbti_unique.csv \
        --outfile top_250_popular.csv \
        --threads 8 \
        --days 30
"""

from __future__ import annotations
import argparse
import datetime as dt
import multiprocessing as mp
import re
import time
from typing import Tuple, Optional

import pandas as pd
import requests
from tqdm import tqdm

# API endpoints
VIEW_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{title}/daily/{start}/{end}"
)
SEARCH_API = "https://en.wikipedia.org/w/api.php"
DATE_FMT = "%Y%m%d"


def find_best_title(query: str) -> Optional[str]:
    """Use Wikipedia search API to find the best matching article title."""
    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        resp = requests.get(SEARCH_API, params=params, timeout=4).json()
        hits = resp.get("query", {}).get("search", [])
        if hits:
            return hits[0]["title"].replace(" ", "_")
    except Exception:
        pass
    return None


def fetch_views(title: str, days: int = 30) -> int:
    """Return total Wikipedia views for a page over given days."""
    end_date = dt.date.today() - dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=days - 1)
    url = VIEW_API.format(
        title=title,
        start=start_date.strftime(DATE_FMT),
        end=end_date.strftime(DATE_FMT),
    )
    try:
        r = requests.get(url, timeout=4)
        r.raise_for_status()
        return sum(item["views"] for item in r.json().get("items", []))
    except Exception:
        return -1


def worker(arg: Tuple[str, int]) -> Tuple[str, int]:
    name, days = arg
    title = find_best_title(name)
    if not title:
        return name, -1
    views = fetch_views(title, days)
    return name, views


def main(infile: str, outfile: str, days: int, threads: int):
    df = pd.read_csv(infile)
    names = df["name"].dropna().unique().tolist()

    print(f"▶ Resolving and querying {len(names):,} names …")
    t0 = time.time()

    with mp.Pool(processes=threads) as pool:
        results = list(
            tqdm(
                pool.imap(worker, [(n, days) for n in names]),
                total=len(names),
                unit="name",
            )
        )

    pv_df = pd.DataFrame(results, columns=["name", "views_30d"])

    merged = df.merge(pv_df, on="name", how="left")
    merged = merged.sort_values("views_30d", ascending=False)
    top250 = merged.head(250).reset_index(drop=True)
    top250.to_csv(outfile, index=False, encoding="utf-8")

    print(f"✔ Saved top 250 → {outfile} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="combined_mbti_unique.csv")
    ap.add_argument("--outfile", default="top_250_popular.csv")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    main(args.infile, args.outfile, args.days, args.threads)


# #!/usr/bin/env python3
# """rank_top250_by_pageviews.py
# ───────────────────────────────────────────────────────────────────────────────
# Ranks the people in *combined_mbti_unique.csv* by recent Wikipedia popularity
# (30‑day total page‑views) and writes **top_250_popular.csv**.

# **Why this method?**  Wikipedia page‑views are a free, public proxy for how
# many people are actively looking up a given name.  Summing the last 30 days
# smooths out one‑day spikes but still reflects current interest.

# Usage (single‑threaded, ~2500 req/min safe):
#     python rank_top250_by_pageviews.py --infile combined_mbti_unique.csv \
#                                        --outfile top_250_popular.csv \
#                                        --threads 8 \
#                                        --days 30

# Requirements:
#     pip install pandas requests tqdm
# """

# # ───────────────────────────── standard libs ────────────────────────────────
# from __future__ import annotations
# import argparse
# import datetime as dt
# import functools
# import multiprocessing as mp
# import re
# import time
# from pathlib import Path
# from typing import Tuple, Optional

# # ───────────────────────────── third‑party ──────────────────────────────────
# import pandas as pd
# import requests
# from tqdm import tqdm

# # ───────────────────────────── constants ────────────────────────────────────
# API_ENDPOINT = (
#     "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
#     "en.wikipedia.org/all-access/user/{title}/daily/{start}/{end}"
# )
# DATE_FMT = "%Y%m%d"  # YYYYMMDD for API

# # ───────────────────────────── helpers ──────────────────────────────────────

# def make_title(name: str) -> str:
#     """Convert person name to Wikipedia URL title."""
#     return re.sub(" +", "_", name.strip())


# def fetch_30d_views(title: str, days: int = 30) -> int:
#     """Return sum of last *days* daily page‑views or −1 if missing."""
#     end_date = dt.date.today() - dt.timedelta(days=1)  # yesterday (API lag)
#     start_date = end_date - dt.timedelta(days=days - 1)
#     url = API_ENDPOINT.format(
#         title=title,
#         start=start_date.strftime(DATE_FMT),
#         end=end_date.strftime(DATE_FMT),
#     )
#     try:
#         data = requests.get(url, timeout=4).json()
#         return sum(item["views"] for item in data.get("items", []))
#     except Exception:
#         return -1  # indicates failure / non‑existent article


# def worker(arg: Tuple[str, int]) -> Tuple[str, int]:
#     title, days = arg
#     views = fetch_30d_views(title, days)
#     return title, views


# # ───────────────────────────── main routine ────────────────────────────────

# def main(infile: str, outfile: str, days: int, threads: int):
#     df = pd.read_csv(infile)

#     names = (
#         df["name"].dropna().unique().tolist()
#     )  # dedup upfront (≈ 51 k)

#     print(f"▶ Querying page‑views for {len(names):,} names …")
#     t0 = time.time()

#     # Multiprocessing pool for parallel fetches
#     with mp.Pool(processes=threads) as pool:
#         results = list(
#             tqdm(
#                 pool.imap(worker, [(make_title(n), days) for n in names]),
#                 total=len(names),
#                 unit="name",
#             )
#         )

#     # Build DataFrame of results
#     pv_df = pd.DataFrame(results, columns=["title", "views_30d"])
#     pv_df["name"] = pv_df["title"].str.replace("_", " ")

#     # Merge back to original df, keep view counts
#     merged = (
#         df.merge(pv_df[["name", "views_30d"]], on="name", how="left")
#           .sort_values("views_30d", ascending=False)
#     )

#     top250 = merged.head(250).reset_index(drop=True)
#     top250.to_csv(outfile, index=False, encoding="utf-8")

#     elapsed = time.time() - t0
#     print(f"✔ Saved top 250 → {outfile}  (elapsed {elapsed/60:.1f} min)")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--infile",  default="combined_mbti_unique.csv")
#     ap.add_argument("--outfile", default="top_250_popular.csv")
#     ap.add_argument("--days",    type=int, default=30, help="look‑back window")
#     ap.add_argument("--threads", type=int, default=8,  help="parallel workers")
#     args = ap.parse_args()

#     main(args.infile, args.outfile, args.days, args.threads)
