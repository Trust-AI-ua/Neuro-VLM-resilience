"""Reference-based non-member inference attack implementation."""

import argparse
import json
import random

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

def load_data(member_similarity_file, non_member_similarity_file, temperature, metric):
    """Load member and non-member similarity scores for the chosen metric."""
    # Temperature is a float; the JSON key is "similarity_0.1" etc., so string-format it.
    temp_key = f"similarity_{temperature}"
    with open(member_similarity_file, 'r') as f:
        member_all = json.load(f)
    with open(non_member_similarity_file, 'r') as f:
        nonmember_all = json.load(f)

    member = [item[temp_key][metric] for item in member_all]
    nonmember = [item[temp_key][metric] for item in nonmember_all]
    # defensive: drop Nones if any
    member = [x for x in member if x is not None]
    nonmember = [x for x in nonmember if x is not None]
    return member, nonmember

def reference_non_member_inference(member_data, non_member_data, granularity, rng):
    """Compute ROC-AUC using reproducible reference/target splits."""
    # --- fixed, reproducible split
    idx = list(range(len(non_member_data)))
    rng.shuffle(idx)
    half = len(idx) // 2
    ref_idx = idx[:half]
    tgt_idx = idx[half:]
    reference_non_member = [non_member_data[i] for i in ref_idx]
    target_non_member    = [non_member_data[i] for i in tgt_idx]
    target_member        = member_data

    # --- granularity guard
    g = int(granularity)
    max_g = min(len(reference_non_member), len(target_non_member), len(target_member))
    if g > max_g:
        g = max_g  # clamp; can also raise if preferred
    if g <= 1:
        # not meaningful; return chance AUC
        return 0.5

    p_scores = []
    labels   = []

    TRIALS = 1000  # keep same default; now fully reproducible via rng
    for _ in range(TRIALS):
        # draw without replacement within a trial for stability
        s_tm  = rng.sample(target_member, g)
        s_rnm = rng.sample(reference_non_member, g)
        s_tnm = rng.sample(target_non_member, g)

        m_tm  = float(np.mean(s_tm))
        m_rnm = float(np.mean(s_rnm))
        m_tnm = float(np.mean(s_tnm))

        v_tm  = float(np.var(s_tm,  ddof=1))
        v_rnm = float(np.var(s_rnm, ddof=1))
        v_tnm = float(np.var(s_tnm, ddof=1))

        # two-sample z with pooled denom
        z_member = (m_tm  - m_rnm) / np.sqrt(v_tm  / g + v_rnm / g + 1e-12)
        z_nonmem = (m_tnm - m_rnm) / np.sqrt(v_tnm / g + v_rnm / g + 1e-12)

        # tail p-values: smaller p => "more member-like"
        p_member = 1.0 - norm.cdf(z_member)
        p_nonmem = 1.0 - norm.cdf(z_nonmem)

        p_scores.append(p_member); labels.append(0)  # 0 = member
        p_scores.append(p_nonmem); labels.append(1)  # 1 = non-member

    auc = roc_auc_score(labels, p_scores)
    return auc

def main(args):
    member, nonmember = load_data(
        args.member_similarity_file, args.non_member_similarity_file,
        args.temperature, args.similarity_metric
    )

    # seeded RNG for reproducibility
    rng = random.Random(args.seed)

    # Run several outer repeats with different (but reproducible) splits.
    aucs = []
    OUTER = 10  # was 5; a little more averaging reduces variance
    for _ in range(OUTER):
        auc = reference_non_member_inference(member, nonmember, args.granularity, rng)
        aucs.append(auc)

    avg_auc = float(np.mean(aucs)) if aucs else 0.5
    print(f"Accuracy: {avg_auc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member_similarity_file', type=str, required=True)
    parser.add_argument('--non_member_similarity_file', type=str, required=True)
    parser.add_argument('--granularity', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--similarity_metric', type=str, default='rouge2_f')
    parser.add_argument('--seed', type=int, default=1234)  # NEW
    args = parser.parse_args()
    main(args)