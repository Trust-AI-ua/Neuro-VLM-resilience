#!/usr/bin/env python3
"""Compute PaliGemma caption similarity for the NoCaps reference set."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rouge import Rouge
from sentence_transformers import SentenceTransformer

def load_all_tsv(tsv_path):
    """Return reference captions keyed by full path and basename."""
    tsv_path = Path(tsv_path)
    refs_by_full, refs_by_base = defaultdict(list), defaultdict(list)
    with tsv_path.open("r", encoding="utf-8") as f:
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
            if cap not in refs_by_full[img]:
                refs_by_full[img].append(cap)
            if cap not in refs_by_base[base]:
                refs_by_base[base].append(cap)
    return refs_by_full, refs_by_base

def best_rouge2_f1(pred, refs, rouge):
    """Return the best ROUGE-2 F1 score across references."""
    best = 0.0
    for r in refs:
        try:
            sc = rouge.get_scores(pred, r, avg=True)
            best = max(best, sc["rouge-2"]["f"])
        except Exception:
            pass
    return best

def best_mpnet(pred, refs, st):
    """Return the maximum MPNet cosine similarity between prediction and references."""
    if not refs: return None
    emb_pred = st.encode([pred], convert_to_tensor=True, normalize_embeddings=True)
    emb_refs = st.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    sims = (emb_pred @ emb_refs.T).squeeze(0)
    return float(sims.max().item())

def append_summary_rows(summary_csv, dataset, model, tau, setname, mp_mean, r2_mean, n_mp, n_r2):
    """Append summary statistics to the shared CSV, creating it if needed."""
    summary_path = Path(summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if need_header:
            writer.writerow(["dataset","model","tau","set","metric","mean","n"])
        tau_str = str(int(float(tau))) if float(tau).is_integer() else str(tau)
        writer.writerow([dataset, model, tau_str, setname, "MPNet_max", f"{mp_mean:.6f}", n_mp])
        writer.writerow([dataset, model, tau_str, setname, "ROUGE2_F1_max", f"{r2_mean:.6f}", n_r2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_json", required=True)  # paligemma_member.json / paligemma_nonmember.json
    ap.add_argument("--all_tsv", required=True)        # .../runs/nocaps/shared/all.tsv
    ap.add_argument("--set", required=True, choices=["member","nonmember"])
    ap.add_argument("--tau", required=True, type=int)  # 0 or 5
    ap.add_argument("--out_json", required=True)       # .../sim/paligemma_*.sim.authors.json
    default_results = Path(__file__).resolve().parent.parent / "results"
    ap.add_argument("--summary_csv", default=str(default_results / "similarity_summary.csv"))
    args = ap.parse_args()

    caps_path = Path(args.captions_json)
    items = json.loads(caps_path.read_text(encoding="utf-8"))
    print(f"[NoCaps · Paligemma2 · τ={args.tau}] set={args.set} | generated={len(items)}")

    refs_full, refs_base = load_all_tsv(args.all_tsv)
    print(f"Refs index -> FULL={len(refs_full)} BASE={len(refs_base)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    rouge = Rouge()

    out, mp_all, r2_all = [], [], []
    for it in items:
        path = (it.get("image_path") or "").strip()
        pred = (it.get("caption") or "").strip()
        base = Path(path).name
        refs = refs_full.get(path) or refs_base.get(base, [])
        if not refs:
            out.append({"image_id": base, "ref": None,
                        "similarity_0.1": {"embedding_mpn": None, "rouge2_f": None}})
            continue

        mp = best_mpnet(pred, refs, st)
        r2 = best_rouge2_f1(pred, refs, rouge)
        if mp is not None: mp_all.append(mp)
        if r2 is not None: r2_all.append(r2)

        out.append({
            "image_id": base,
            "ref": "<max-over-NoCaps-refs>",
            "similarity_0.1": {"embedding_mpn": mp, "rouge2_f": r2},
            "similarity": {"MPNet_max": mp, "ROUGE2_F1_max": r2}
        })

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    def mean_safe(vals):
        """Compute a float mean, guarding against empty lists."""
        return float(np.mean(vals)) if vals else 0.0

    mp_mean = mean_safe(mp_all)
    r2_mean = mean_safe(r2_all)
    print(
        f"[NoCaps · Paligemma2 · τ={args.tau}] {args.set}: "
        f"MPNet_max={mp_mean:.4f}, ROUGE2_F1_max={r2_mean:.4f} "
        f"(n={len(mp_all)}, missing={len(items)-len(mp_all)})"
    )

    append_summary_rows(
        args.summary_csv,
        "NoCaps",
        "Paligemma2",
        args.tau,
        args.set,
        mp_mean,
        r2_mean,
        len(mp_all),
        len(r2_all),
    )

    print(f"Wrote per-item similarity -> {out_path}")
    print("Appended means to:", Path(args.summary_csv))

if __name__ == "__main__":
    main()