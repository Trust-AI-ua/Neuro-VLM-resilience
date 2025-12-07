#!/usr/bin/env python3
"""Compute PaliGemma caption similarity for Flickr30k references."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

def load_all_refs(tsv_path):
    """Return reference captions keyed by full path and basename."""
    tsv_path = Path(tsv_path)
    refs_by_full, refs_by_base = defaultdict(list), defaultdict(list)
    with tsv_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                img, cap = line.split("\t", 1)
            except ValueError:
                continue
            img = img.strip()
            cap = cap.strip()
            if not img or not cap:
                continue
            base = Path(img).name
            if cap not in refs_by_full[img]:
                refs_by_full[img].append(cap)
            if cap not in refs_by_base[base]:
                refs_by_base[base].append(cap)
    return refs_by_full, refs_by_base

def best_rouge2_f1(pred, refs, rouge):
    """Return the best ROUGE-2 F1 between prediction and reference list."""
    best = 0.0
    for r in refs:
        try:
            sc = rouge.get_scores(pred, r, avg=True)
            best = max(best, sc["rouge-2"]["f"])
        except Exception:
            pass
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_json", required=True)   # paligemma_member.json or paligemma_nonmember.json
    ap.add_argument("--f30k_all_tsv", required=True)    # .../runs/flickr30k/shared/all.tsv
    ap.add_argument("--set", required=True, choices=["member","nonmember"])
    ap.add_argument("--tau", required=True, type=int)   # 0 or 5
    ap.add_argument("--out_json", required=True)        # .../sim/paligemma_{set}.sim.authors.json
    default_results = Path(__file__).resolve().parent.parent / "results"
    ap.add_argument("--summary_csv", default=str(default_results / "similarity_summary.csv"))
    args = ap.parse_args()

    # libs
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception:
        raise SystemExit("[ERROR] Install sentence-transformers")
    try:
        from rouge import Rouge
    except Exception:
        raise SystemExit("[ERROR] Install rouge")

    # load generated captions
    caps_path = Path(args.captions_json)
    items = json.loads(caps_path.read_text(encoding="utf-8"))
    print(f"[Flickr30k · Paligemma2 · τ={args.tau}] set={args.set} | generated={len(items)}")

    # load references (Flickr30k)
    refs_full, refs_base = load_all_refs(args.f30k_all_tsv)
    if not refs_full and not refs_base:
        raise SystemExit(f"[ERROR] No Flickr30k refs loaded from {args.f30k_all_tsv}")
    print(f"Refs index -> FULL={len(refs_full)} BASE={len(refs_base)}")

    # scorers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    rouge = Rouge()

    # helpers
    def best_mpnet(pred, refs):
        if not refs: return None
        emb_pred = st.encode([pred], convert_to_tensor=True, normalize_embeddings=True)
        emb_refs = st.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
        sims = (emb_pred @ emb_refs.T).squeeze(0)
        return float(sims.max().item())

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs)/len(xs) if xs else 0.0

    # compute per-item best-over-refs + write authors format
    out, mp_all, r2_all, missing = [], [], [], 0
    stems = []
    for it in items:
        path = (it.get("image_path") or "").strip()
        y = (it.get("caption") or "").strip()
        base = Path(path).name
        stems.append(base)

        # refs by full-path first; else by basename
        refs = refs_full.get(path) or refs_base.get(base, [])
        if not refs:
            missing += 1
            mp, r2 = None, None
        else:
            mp = best_mpnet(y, refs)
            r2 = best_rouge2_f1(y, refs, rouge)

        if mp is not None: mp_all.append(mp)
        if r2 is not None: r2_all.append(r2)

        out.append({
            "image_id": base,              # matches Flickr30k naming
            "ref": "<max-over-Flickr30k-refs>",
            "similarity_0.1": {
                "embedding_mpn": mp,
                "rouge2_f": r2
            },
            "similarity": {                # keep the flat copy for future use
                "MPNet_max": mp,
                "ROUGE2_F1_max": r2
            }
        })

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)

    mp_m = mean(mp_all); r2_m = mean(r2_all)
    print(f"[Flickr30k · Paligemma2 · τ={args.tau}] {args.set}: MPNet_max={mp_m:.4f}, ROUGE2_F1_max={r2_m:.4f} (n={len(items)}, missing={missing})")
    print(f"Wrote per-item similarity -> {out_path}")

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset","model","tau","set","metric","mean","n"])
        w.writerow(["Flickr30k","PALIGEMMA2",args.tau,args.set,"MPNet_max",f"{mp_m:.6f}",len(items)])
        w.writerow(["Flickr30k","PALIGEMMA2",args.tau,args.set,"ROUGE2_F1_max",f"{r2_m:.6f}",len(items)])
    print("Appended means to:", summary_path)

if __name__ == "__main__":
    main()