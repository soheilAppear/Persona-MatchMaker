#!/usr/bin/env python3
"""
duplicate_persona_removal.py
Remove duplicate (name, category) rows from a characters JSON file.

Usage:
  python duplicate_persona_removal.py \
      --input  characters.json \
      --output characters_dedup.json
"""
import json
import argparse
from pathlib import Path

def deduplicate(path_in: str, path_out: str) -> None:
    data = json.loads(Path(path_in).read_text(encoding="utf-8"))
    seen = set()
    unique = []
    for item in data:
        key = (item["name"].strip(), item["category"].strip())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    Path(path_out).write_text(
        json.dumps(unique, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Removed {len(data) - len(unique)} duplicates.")
    print(f"Saved  {len(unique)} unique personas → {path_out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="input characters JSON")
    ap.add_argument("--output", required=True, help="output cleaned JSON")
    args = ap.parse_args()
    deduplicate(args.input, args.output)
