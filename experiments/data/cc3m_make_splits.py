#!/usr/bin/env python3
"""Download CC3M samples to build member and non-member splits."""

import csv
import http.client
import imghdr
import os
import random
import socket
import ssl
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "runs" / "cc3m"
ALL_TSV = ROOT / "shared" / "all.tsv"
IMG_DIR = ROOT / "shared" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# outputs
MEMBERS_PATHS = ROOT / "shared" / "members400_paths.txt"
NONMEMBERS_PATHS = ROOT / "shared" / "nonmembers400_paths.txt"
MEMBERS_TSV = ROOT / "shared" / "members400.tsv"
NONMEMBERS_TSV = ROOT / "shared" / "nonmembers400.tsv"

# --- config ---
TARGET_PER_SET = 400
MAX_SCAN = 20000         # scan up to this many rows in all.tsv
TIMEOUT = 8              # seconds per request
RETRIES = 2
MIN_BYTES = 2048
SKIP_SUBSTR = [
    "gettyimages", "tumblr", "wordpress", "blogspot",
    "placeholder", "stock-photo"
]

# SSL + timeouts
ssl._create_default_https_context = ssl._create_unverified_context
socket.setdefaulttimeout(TIMEOUT)

# --- custom User-Agent to avoid 403s / disconnections ---
opener = urllib.request.build_opener()
opener.addheaders = [
    (
        "User-Agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
]
urllib.request.install_opener(opener)


def valid_url(u: str) -> bool:
    u = (u or "").lower()
    return u.startswith(("http://", "https://")) and all(s not in u for s in SKIP_SUBSTR)


def try_download(url: str, path: Path) -> bool:
    """Download with minimal validation and exponential backoff."""

    for attempt in range(RETRIES):
        try:
            urllib.request.urlretrieve(url, path)

            # size check
            if not path.exists() or path.stat().st_size < MIN_BYTES:
                time.sleep(0.25 * (2 ** attempt))
                continue

            # structural checks
            if imghdr.what(path) is None:
                time.sleep(0.25 * (2 ** attempt))
                continue

            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                time.sleep(0.25 * (2 ** attempt))
                continue

            return True

        except (
            URLError,
            HTTPError,
            socket.timeout,
            ssl.SSLError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ):
            # short jitter + backoff
            time.sleep(0.25 * (2 ** attempt) + random.random() * 0.1)

    # all retries failed; clean up partial
    try:
        path.unlink()
    except Exception:
        pass
    return False


# load rows
rows = []
with ALL_TSV.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for i, row in enumerate(reader):
        if i >= MAX_SCAN:
            break
        url, cap = row.get("image_url", ""), row.get("caption", "")
        if not url or not cap:
            continue
        if not valid_url(url):
            continue
        rows.append((url.strip(), cap.strip()))

random.seed(42)
random.shuffle(rows)

members, nonmembers = [], []
mid, nid = 0, 0
for i, (url, cap) in enumerate(rows):
    if len(members) >= TARGET_PER_SET and len(nonmembers) >= TARGET_PER_SET:
        break

    picking_members = len(members) < TARGET_PER_SET
    prefix = "member" if picking_members else "nonmember"
    idx = mid if picking_members else (TARGET_PER_SET + nid)
    img_name = f"{prefix}_{idx:04d}.jpg"
    img_path = IMG_DIR / img_name

    ok = try_download(url, img_path)
    if not ok:
        if (i % 25) == 0:
            print(f"[scan {i}] skipped (bad download/placeholder)")
        continue

    rec = (img_path, url, cap)
    if picking_members:
        members.append(rec)
        mid += 1
    else:
        nonmembers.append(rec)
        nid += 1

    if (len(members) + len(nonmembers)) % 10 == 0:
        print(f"[progress] members={len(members)} nonmembers={len(nonmembers)} (scanned {i})")

print(f"done: {len(members)} members, {len(nonmembers)} nonmembers")

# write path lists (for caption gen)
with MEMBERS_PATHS.open("w", encoding="utf-8") as handle:
    for path, _, _ in members:
        handle.write(str(path) + "\n")
with NONMEMBERS_PATHS.open("w", encoding="utf-8") as handle:
    for path, _, _ in nonmembers:
        handle.write(str(path) + "\n")

# write aligned TSVs (path \t caption)
with MEMBERS_TSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    for path, _, cap in members:
        writer.writerow([path, cap])
with NONMEMBERS_TSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    for path, _, cap in nonmembers:
        writer.writerow([path, cap])

print("files written:")
print(" ", MEMBERS_PATHS)
print(" ", NONMEMBERS_PATHS)
print(" ", MEMBERS_TSV)
print(" ", NONMEMBERS_TSV)