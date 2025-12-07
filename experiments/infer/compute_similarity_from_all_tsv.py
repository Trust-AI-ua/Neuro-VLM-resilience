#!/usr/bin/env python3
"""Score caption similarity against an ``ALL.tsv`` reference file."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

def load_refs_all(path):
    """Return reference captions keyed by image basename."""
    path = Path(path)
    refs_by_base = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            img, cap = parts[0].strip(), parts[1].strip()
            if not cap:
                continue
            base = Path(img).name
            if cap not in refs_by_base[base]:
                refs_by_base[base].append(cap)
    return refs_by_base

def mean(xs):
    """Compute the mean ignoring ``None`` entries."""
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else 0.0

def get_pred_and_base(item):
    """Return (pred_text, image_base) or (None, None) if missing."""
    # ViT-GPT2 schema
    if "image_path" in item and "caption" in item:
        pred = (item.get("caption") or "").strip()
        base = Path(item.get("image_path") or "").name
        return pred, base
    # BLIP schema
    if "image_id" in item and ("pred" in item or "caption" in item):
        iid = (str(item.get("image_id")) or "").strip()
        pred = (item.get("pred") or item.get("caption") or "").strip()
        base = iid if iid.lower().endswith(".jpg") else f"{iid}.jpg"
        return pred, base
    # Fallback: try generic keys
    for k_text in ("pred", "caption"):
        if k_text in item:
            txt = (item.get(k_text) or "").strip()
            # try any path-like key to derive base
            for k_path in ("image_path", "path", "img", "image"):
                if k_path in item:
                    base = Path(item.get(k_path) or "").name
                    return txt, base
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps_json", required=True)
    ap.add_argument("--all_tsv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--split", choices=["member","nonmember"], required=True)
    default_results = Path(__file__).resolve().parent.parent / "results"
    ap.add_argument("--summary_csv", default=str(default_results / "similarity_summary.csv"))
    args = ap.parse_args()

    # Lazy imports to keep errors informative
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception:
        print("[ERROR] Install sentence-transformers", file=sys.stderr); sys.exit(2)
    try:
        from rouge import Rouge
    except Exception:
        print("[ERROR] Install rouge", file=sys.stderr); sys.exit(2)

    with open(args.caps_json, "r", encoding="utf-8") as handle:
        caps = json.load(handle)
    refs = load_refs_all(args.all_tsv)
    print(f"Loaded captions: {len(caps)} | Refs BASE: {len(refs)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    rouge = Rouge()

    def best_rouge2_f1(pred, ref_list):
        best = 0.0
        for r in ref_list:
            try:
                sc = rouge.get_scores(pred, r, avg=True)
                best = max(best, sc["rouge-2"]["f"])
            except Exception:
                pass
        return best

    def best_mpnet(pred, ref_list):
        if not ref_list: return None
        emb_pred = st.encode([pred], convert_to_tensor=True, normalize_embeddings=True)
        emb_refs = st.encode(ref_list, convert_to_tensor=True, normalize_embeddings=True)
        sims = (emb_pred @ emb_refs.T).squeeze(0)
        return float(sims.max().item())

    out, mp_all, r2_all, matched = [], [], [], 0
    for it in caps:
        pred, base = get_pred_and_base(it)
        if pred is None or not base:
            out.append({"pred": pred, "image_base": base, "mpnet_max": None, "rouge2_f1_max": None})
            continue
        ref_list = refs.get(base, [])
        if pred and ref_list:
            mp = best_mpnet(pred, ref_list)
            r2 = best_rouge2_f1(pred, ref_list)
            matched += 1
        else:
            mp, r2 = None, None
        rec = {"image_base": base, "pred": pred, "mpnet_max": mp, "rouge2_f1_max": r2}
        # keep original identifiers if present (nice for debugging)
        for k in ("image_id","image_path"):
            if k in it: rec[k] = it[k]
        out.append(rec)
        if mp is not None: mp_all.append(mp)
        if r2 is not None: r2_all.append(r2)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(f"Wrote per-item similarity -> {out_path}")
    print(f"Matched items: {matched} | missing_ref_matches: {len(caps)-matched}")

    mp_m, r2_m = mean(mp_all), mean(r2_all)
    print(f"Means: mpnet={mp_m:.4f} | rouge2_f1={r2_m:.4f}")

    # Append to summary (MPNet_max & ROUGE2_F1_max)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["dataset","model","tau","set","metric","mean","n"])
        tau_str = str(int(args.tau)) if float(args.tau).is_integer() else str(args.tau)
        w.writerow([args.dataset, args.model, tau_str, args.split, "MPNet_max", f"{mp_m:.6f}", matched])
        w.writerow([args.dataset, args.model, tau_str, args.split, "ROUGE2_F1_max", f"{r2_m:.6f}", matched])
    print("Appended means to:", summary_path)

if __name__ == "__main__":
    main()