#!/usr/bin/env python3
"""Compute PaliGemma caption similarity against COCO references (MPNet + ROUGE-2)."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------- simple ROUGE-2 F1 (bigram) ----------
def _tok(s):
    """Lowercase and tokenize on whitespace while stripping punctuation."""
    return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in s).split() if t]

def _bigrams(tokens):
    """Return the set of bigrams for a token list."""
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()

def rouge2_f1(hyp, ref):
    """Compute ROUGE-2 F1 between two strings."""
    H = _bigrams(_tok(hyp)); R = _bigrams(_tok(ref))
    if not H and not R: return 1.0
    if not H or not R:  return 0.0
    overlap = len(H & R)
    pr = overlap / max(len(H), 1)
    rc = overlap / max(len(R), 1)
    return 0.0 if (pr+rc) == 0 else 2*pr*rc/(pr+rc)

# ---------- MPNet encoder ----------
def embed_mpnet(sentences):
    """Embed sentences with MPNet, normalizing vectors for cosine similarity."""
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=dev)
    return model.encode(
        sentences, batch_size=64, convert_to_numpy=True,
        show_progress_bar=False, normalize_embeddings=True
    )

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"


def append_summary_row(dataset, model, tau, setname, metric, mean, n):
    """Append a summary statistic row to the shared CSV, creating it if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "similarity_summary.csv"
    header = "dataset,model,tau,set,metric,mean,n\n"
    line = f"{dataset},{model},{int(float(tau))},{setname},{metric},{mean:.6f},{n}\n"
    if not out.exists():
        out.write_text(header + line)
    else:
        with out.open("a", encoding="utf-8") as f:
            f.write(line)

def load_coco_refs(ann_dir):
    """Load COCO reference captions grouped by image stem."""
    ann_dir = Path(ann_dir)
    refs = defaultdict(list)  # stem -> list[captions]
    for fname in ["captions_train2017.json", "captions_val2017.json"]:
        p = ann_dir / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        id2file = {img["id"]: img["file_name"] for img in data.get("images", [])}
        for ann in data.get("annotations", []):
            fn = id2file.get(ann["image_id"])
            if not fn: 
                continue
            stem = Path(fn).stem
            cap = (ann.get("caption") or "").strip()
            if cap:
                refs[stem].append(cap)
    return refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_json", required=True)  # paligemma_member.json or paligemma_nonmember.json
    ap.add_argument("--coco_ann_dir", required=True)   # .../data/coco/annotations
    ap.add_argument("--set", required=True, choices=["member","nonmember"])
    ap.add_argument("--tau", required=True, type=int)  # 0 or 5
    ap.add_argument("--out_json", required=True)       # .../sim/XXXX.sim.authors.json
    ap.add_argument("--temperature", type=float, default=0.1,
                help="Temperature key for authors schema, e.g., 0.1 -> similarity_0.1")
    args = ap.parse_args()
    temp_key = f"similarity_{args.temperature:.1f}"

    # load generated captions
    items = json.loads(Path(args.captions_json).read_text())
    # get image stem from image_path or image_id
    def stem_for(it):
        imgid = str(it.get("image_id","")).strip()
        if imgid:
            return Path(imgid).stem
        p = str(it.get("image_path","")).strip()
        return Path(p).stem

    refs = load_coco_refs(args.coco_ann_dir)
    if not refs:
        print(f"[ERROR] No COCO refs loaded from {args.coco_ann_dir}")
        return

    # Restrict to items that have reference captions.
    pairs = []
    missing = 0
    for it in items:
        s = stem_for(it)
        hyp = (it.get("caption") or "").strip()
        if not hyp or s not in refs:
            missing += 1
            continue
        pairs.append((s, hyp))

    if not pairs:
        print(f"[WARN] No matching items; missing={missing}")
        append_summary_row("COCO","Paligemma2",args.tau,args.set,"MPNet_max",0.0,0)
        append_summary_row("COCO","Paligemma2",args.tau,args.set,"ROUGE2_F1_max",0.0,0)
        return

    stems, hyps = zip(*pairs)

    # MPNet max over refs
    hyp_emb = embed_mpnet(list(hyps))
    ref_texts = []
    offsets = [0]
    for s in stems:
        R = refs[s]
        ref_texts.extend(R)
        offsets.append(offsets[-1] + len(R))
    ref_emb = embed_mpnet(ref_texts)

    mp_max = []
    for i in range(len(stems)):
        lo, hi = offsets[i], offsets[i+1]
        sims = ref_emb[lo:hi] @ hyp_emb[i]
        mp_max.append(float(np.max(sims)) if sims.size else 0.0)

    # ROUGE-2 F1 max over refs
    r2_max = []
    for i, s in enumerate(stems):
        hyp = hyps[i]
        r2_max.append(max(rouge2_f1(hyp, r) for r in refs[s]) if refs[s] else 0.0)

    mp_mean = float(np.mean(mp_max))
    r2_mean = float(np.mean(r2_max))
    n = len(stems)

    print(f"[COCO · Paligemma2 · τ={args.tau}] {args.set}: MPNet_max={mp_mean:.4f}, ROUGE2_F1_max={r2_mean:.4f} (n={n}, missing={missing})")

    # Authors-style per-item JSON (attack expects similarity_{T} -> {embedding_mpn, rouge2_f}).
    out = []
    for i, s in enumerate(stems):
        mp_val = None if mp_max[i] is None else float(mp_max[i])
        r2_val = None if r2_max[i] is None else float(r2_max[i])

        out.append({
            "image_id": s + ".jpg",
            "ref": "<max-over-COCO-refs>",  # optional; can remove if you prefer
            temp_key: {
                "embedding_mpn": mp_val,
                "rouge2_f": r2_val
            },
            # (Optional) keep legacy flat block for backwards-compatibility
            "similarity": {
                "MPNet_max": mp_val,
                "ROUGE2_F1_max": r2_val
            }
        })
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote per-item similarity -> {args.out_json}")

    append_summary_row("COCO","Paligemma2",args.tau,args.set,"MPNet_max",mp_mean,n)
    append_summary_row("COCO","Paligemma2",args.tau,args.set,"ROUGE2_F1_max",r2_mean,n)

if __name__ == "__main__":
    main()