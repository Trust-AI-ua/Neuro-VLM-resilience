#!/usr/bin/env python
"""Wrapper for running the reference non-member attack across granularities."""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORS_ATTACK = REPO_ROOT / "experiments/reference_non_member_inference.py"

def norm_metric(m):
    """Normalize metric aliases to the attack script's expected identifiers."""
    m = (m or "").strip().lower().replace(" ", "")
    if m in ("mpnet","mpnet_max","embedding_mpn"): return "embedding_mpn"
    if m in ("rouge-2","rouge2","rouge2_f","rouge2_f1"): return "rouge2_f"
    if m in ("clip","embedding_cosine","cosine"): return "embedding_cosine"
    # fallback: return as-is (lowercased)
    return m

def infer_dataset(path):
    """Infer dataset label from a similarity file path."""
    # expect .../experiments/runs/<dataset>/...
    m = re.search(r"/experiments/runs/([^/]+)/", path)
    return m.group(1).upper() if m else "UNKNOWN"

def infer_model(path):
    """Infer model label from a similarity file path."""
    # expect .../experiments/runs/<dataset>/<model>/...
    m = re.search(r"/experiments/runs/[^/]+/([^/]+)/", path)
    if not m: return "UNKNOWN"
    key = m.group(1).lower()
    if key == "blip": return "BLIP"
    if key == "clip": return "CLIP"
    if key == "paligemma2": return "PALIGEMMA2"
    if key in ("vitgpt2","vit-gpt2","vit_gpt2"): return "ViT-GPT2"
    return key.upper()

def write_unified(csv_path, rows):
    """Append rows with schema (dataset, model, tau, metric, g, accuracy)."""
    csv_path = Path(csv_path)
    write_header = True
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                first = f.readline().strip().split(",")
                write_header = first != ["dataset","model","tau","metric","g","accuracy"]
        except Exception:
            # if unreadable, we will just rewrite header then append
            write_header = True

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset","model","tau","metric","g","accuracy"])
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member_sim", required=True, help="members sim JSON (authors format)")
    ap.add_argument("--nonmember_sim", required=True, help="non-members sim JSON (authors format)")
    ap.add_argument("--granularity", required=True, help="single value or comma list, e.g., 50,100,200")
    ap.add_argument("--metric", required=True, help="e.g., embedding_mpn | rouge2_f | MPNet | ROUGE-2 | CLIP")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--append_csv", default="", help="optional CSV path for unified schema (dataset,model,tau,metric,g,accuracy)")
    ap.add_argument("--tau", default="", help="tau tag for CSV (string accepted; will be normalized if numeric)")
    ap.add_argument("--dataset", default=None, help="explicit dataset label (optional; else inferred)")
    ap.add_argument("--model", default=None, help="explicit model label (optional; else inferred)")
    args = ap.parse_args()

    if not AUTHORS_ATTACK.is_file():
        print(f"[!] Authors' attack script not found: {AUTHORS_ATTACK}")
        sys.exit(1)

    # normalize / infer labels
    metric = norm_metric(args.metric)
    try:
        tau_str = str(int(float(args.tau))) if str(args.tau).replace(".","",1).isdigit() else str(args.tau)
    except Exception:
        tau_str = str(args.tau)

    dataset = args.dataset or infer_dataset(args.member_sim)
    model   = args.model   or infer_model(args.member_sim)

    grans = [g.strip() for g in str(args.granularity).split(",") if g.strip()]
    rows_to_append = []

    # Ensure similarity_0.1 exists ---
    def ensure_similarity_key(path):
        """Ensure legacy similarity files expose the ``similarity_0.1`` block."""
        sim_path = Path(path)
        with sim_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        changed = False
        for it in data:
            if "similarity_0.1" not in it:
                if "mpnet_max" in it or "rouge2_f1_max" in it:
                    it["similarity_0.1"] = {
                        "embedding_mpn": it.get("mpnet_max"),
                        "rouge2_f": it.get("rouge2_f1_max"),
                    }
                    changed = True
        if changed:
            with sim_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            print(f"[PATCH] Added similarity_0.1 key to {sim_path}")

    ensure_similarity_key(args.member_sim)
    ensure_similarity_key(args.nonmember_sim)

    for g in grans:
        cmd = [
            sys.executable, str(AUTHORS_ATTACK),
            "--member_similarity_file", args.member_sim,
            "--non_member_similarity_file", args.nonmember_sim,
            "--granularity", str(int(g)),
            "--temperature", str(args.temperature),
            "--similarity_metric", metric,
        ]
        print(">>", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        out = res.stdout.strip()
        print(out)
        if res.stderr.strip():
            print(res.stderr.strip(), file=sys.stderr)

        # Parse "Accuracy: 0.9444"
        acc = None
        for line in out.splitlines():
            if line.strip().lower().startswith("accuracy:"):
                try:
                    acc = float(line.split(":")[1].strip())
                except Exception:
                    pass
        if acc is not None and args.append_csv:
            rows_to_append.append((dataset, model, tau_str, metric, int(g), f"{acc:.4f}"))

    if rows_to_append and args.append_csv:
        write_unified(args.append_csv, rows_to_append)

if __name__ == "__main__":
    main()
