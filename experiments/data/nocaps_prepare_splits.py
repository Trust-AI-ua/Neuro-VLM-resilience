import argparse
import json
import os
import random
from pathlib import Path

def load_nocaps(captions_json, images_dir):
    """Return NoCaps items with available captions and absolute paths."""
    with open(captions_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    images = data.get("images", [])
    ann    = data.get("annotations", [])
    # Build index -> filename
    idx2file = {i: im.get("file_name","").strip() for i, im in enumerate(images)}
    # Collect first caption per image (can switch to any policy)
    caps = {}
    for a in ann:
        iid = a.get("image_id")
        if iid is None: 
            continue
        if iid not in caps:
            caps[iid] = a.get("caption","").strip()
    # Build items with absolute paths
    items = []
    images_dir = Path(images_dir)
    for idx, fname in idx2file.items():
        if not fname: 
            continue
        p = images_dir / fname
        if p.is_file():
            ref = caps.get(idx, "")
            items.append({"id": idx, "file_name": fname, "path": str(p), "ref": ref})
    return items

def write_list(p, lines):
    """Write a newline-delimited file of strings."""
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for ln in lines:
            f.write(str(ln).strip() + "\n")

def write_tsv(p, rows):
    """Write a TSV with (path, reference caption) tuples."""
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for (path, ref) in rows:
            f.write(f"{path}\t{ref}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",     required=True)
    ap.add_argument("--captions_json",  required=True)
    ap.add_argument("--out_dir",        required=True)
    ap.add_argument("--train_n",        type=int, default=360)
    ap.add_argument("--val_n",          type=int, default=40)
    ap.add_argument("--members_n",      type=int, default=400)
    ap.add_argument("--nonmembers_n",   type=int, default=400)
    ap.add_argument("--seed",           type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    items = load_nocaps(args.captions_json, args.images_dir)
    # keep only those with a non-empty caption
    items = [it for it in items if it["ref"]]
    total = len(items)
    if total < (args.members_n + args.nonmembers_n):
        raise ValueError(f"Not enough captioned images ({total}) for {args.members_n}+{args.nonmembers_n}.")

    # Sample member/non-member disjoint sets
    idxs = list(range(total))
    random.shuffle(idxs)
    mem_idx = idxs[:args.members_n]
    non_idx = idxs[args.members_n:args.members_n + args.nonmembers_n]

    members    = [items[i] for i in mem_idx]
    nonmembers = [items[i] for i in non_idx]

    # Train/val from the members set
    if len(members) < args.train_n + args.val_n:
        raise ValueError("Members not enough for requested train/val sizes.")
    random.shuffle(members)
    train = members[:args.train_n]
    val   = members[args.train_n:args.train_n + args.val_n]

    out_root = Path(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    # Write TSVs
    write_tsv(str(out_root / "train.tsv"), [(r["path"], r["ref"]) for r in train])
    write_tsv(str(out_root / "val.tsv"),   [(r["path"], r["ref"]) for r in val])

    # Write path/id lists
    write_list(str(out_root / "members_paths.txt"),    [r["path"] for r in members])
    write_list(str(out_root / "nonmembers_paths.txt"), [r["path"] for r in nonmembers])
    write_list(str(out_root / "members_ids.txt"),      [Path(r["file_name"]).stem for r in members])
    write_list(str(out_root / "nonmembers_ids.txt"),   [Path(r["file_name"]).stem for r in nonmembers])

    # Convenience TSVs (path \t ref) for direct CLIP scoring if desired
    write_tsv(str(out_root / "members.tsv"),    [(r["path"], r["ref"]) for r in members])
    write_tsv(str(out_root / "nonmembers.tsv"), [(r["path"], r["ref"]) for r in nonmembers])

    print(f"Members: {len(members)} | Non-members: {len(nonmembers)} (disjoint: True)")
    print("Wrote:")
    for rel in ["train.tsv","val.tsv",
                "members_paths.txt","nonmembers_paths.txt",
                "members_ids.txt","nonmembers_ids.txt",
                "members.tsv","nonmembers.tsv"]:
        print(f"  {out_root/rel}")

if __name__ == "__main__":
    main()
