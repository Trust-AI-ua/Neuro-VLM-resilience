#!/usr/bin/env python3
"""Stream the CC3M dataset and write an ``all.tsv`` file."""

import csv
import os
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1] / "runs" / "cc3m"
OUT_TSV = ROOT / "shared" / "all.tsv"
OUT_TSV.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "google-research-datasets/conceptual_captions",
    split="train",
    streaming=True,
)

with OUT_TSV.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["image_url", "caption"])
    for i, ex in enumerate(ds):
        url = ex.get("image_url")
        cap = ex.get("caption")
        if not url or not cap:
            continue
        writer.writerow([url.strip(), cap.strip()])
        if (i + 1) % 100000 == 0:
            print(f"[cc3m] wrote {i+1:,} rows")

print("done.")
