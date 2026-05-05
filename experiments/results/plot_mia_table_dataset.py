#!/usr/bin/env python3
import os, re, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Paths
ATTACK_CSV = "experiments/results/attack_accuracy.csv"
SIM_CSV    = "experiments/results/similarity_summary.csv"
RUN_ROOT   = "experiments/runs"
OUTDIR     = "experiments/results/"

# Configuration
WANT_MODELS   = {"BLIP", "PaliGemma2", "ViT-GPT2"}
WANT_DATASETS = ["COCO", "NoCaps", "CC3M"]    # order
WANT_TAUS     = {0, 2, 3}
TAU_TO_THREAT = {0: "BASELINE", 2: "NEURO", 3: "NEURO++"}

# metric name -> JSON key
SIM_KEY = {
    "MPNet":    "mpnet_max",
    "ROUGE-2":  "rouge2_f1_max",
}

# Helpers
def norm_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def std_model(s):
    if not isinstance(s, str): 
        return s
    x = s.lower().strip()
    if "vit" in x and "gpt" in x: 
        return "ViT-GPT2"
    if "vit-gpt2" in x or "vit_gpt2" in x: 
        return "ViT-GPT2"
    if "blip" in x: 
        return "BLIP"
    if "pali" in x and "gemma" in x: 
        return "PaliGemma2"
    return s

def std_dataset(s):
    if not isinstance(s, str): 
        return s
    x = s.lower().strip()
    if "coco" in x: 
        return "COCO"
    if "nocaps" in x: 
        return "NoCaps"
    if "cc3m" in x or "conceptual" in x: 
        return "CC3M"
    return s

def coerce_tau(v):
    try:
        if isinstance(v, str):
            m = re.search(r"(-?\d+\.?\d*)", v)
            if not m: 
                return None
            return int(float(m.group(1)))
        return int(float(v))
    except Exception:
        return None

def norm_metric_sim(m):
    if not isinstance(m, str):
        return None
    x = m.strip().lower()
    if "mpnet" in x:
        return "MPNet"
    if "rouge2" in x:
        return "ROUGE-2"
    return None

def is_percent_like(series):
    s = pd.to_numeric(series.dropna(), errors="coerce")
    if s.empty: 
        return False
    return s.mean() > 1.5

def fmt_pm(mu, sd, dec=3, as_percent=False):
    """Format mean ± std as a string; both already on the correct scale."""
    if mu is None or pd.isna(mu):
        return ""
    mu = float(mu)
    if sd is None or pd.isna(sd):
        return f"{mu:.{dec}f}"
    sd = float(sd)
    if as_percent:
        return f"{mu:.{dec}f} ± {sd:.{dec}f}"
    return f"{mu:.{dec}f} ± {sd:.{dec}f}"

def load_sim_dict(path, metric_key):
    """
    Load a JSON similarity file and return a dict:
        image_base -> similarity_value_for_given_metric
    """
    if path is None or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for item in data:
        base = item.get("image_base")
        if not base:
            if "image_id" in item:
                base = str(item["image_id"])
            elif "image_path" in item:
                base = os.path.basename(item["image_path"])
            else:
                continue
        val = item.get(metric_key, None)
        if val is None:
            continue
        try:
            out[base] = float(val)
        except Exception:
            continue
    return out

# Similarity panel: member / non-member means from similarity_summary.csv
def build_similarity_panel(sim_csv, dataset_name):
    sim = pd.read_csv(sim_csv)
    sim = norm_cols(sim)
    need = ["dataset","model","tau","set","metric","mean"]
    for c in need:
        if c not in sim.columns:
            raise ValueError(f"similarity_summary.csv missing column: {c}")

    # normalize
    sim["dataset"] = sim["dataset"].apply(std_dataset)
    sim["model"]   = sim["model"].apply(std_model)
    sim["tau"]     = sim["tau"].apply(coerce_tau)
    sim["metric"]  = sim["metric"].apply(norm_metric_sim)

    # keeping only wanted dataset / models / taus / metrics
    sim = sim[
        (sim["dataset"] == dataset_name)
        & (sim["model"].isin(WANT_MODELS))
        & (sim["tau"].isin(WANT_TAUS))
    ]
    sim = sim[sim["metric"].isin(["MPNet", "ROUGE-2"])]

    if sim.empty:
        return pd.DataFrame(
            columns=[
                "dataset","model","tau","metric",
                "member_mean","nonmember_mean"
            ]
        )

    key = ["dataset", "model", "tau", "metric"]

    # pivot to get member & nonmember in the same row
    piv = sim.pivot_table(
        index=key,
        columns="set",
        values="mean",
        aggfunc="mean",
    )

    for col in ["member", "nonmember"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv = piv.reset_index()

    # aggregate to get member/non-member means per (dataset,model,tau,metric)
    ag = (
        piv
        .groupby(key)
        .agg(
            member_mean    = ("member",    lambda x: pd.to_numeric(x, errors="coerce").mean()),
            nonmember_mean = ("nonmember", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        )
        .reset_index()
    )

    return ag  # dataset, model, tau, metric, member_mean, nonmember_mean

# Attack panel: mean + std from attack_accuracy.csv
def build_attack_panel(attack_csv, dataset_name):
    atta = pd.read_csv(attack_csv)
    atta = norm_cols(atta)
    need = ["dataset","model","tau","accuracy"]
    for c in need:
        if c not in atta.columns:
            raise ValueError(f"attack_accuracy.csv missing column: {c}")

    atta["dataset"] = atta["dataset"].apply(std_dataset)
    atta["model"]   = atta["model"].apply(std_model)
    atta["tau"]     = atta["tau"].apply(coerce_tau)

    atta = atta[(atta["dataset"] == dataset_name) &
                (atta["model"].isin(WANT_MODELS)) &
                (atta["tau"].isin(WANT_TAUS))]

    if atta.empty:
        return pd.DataFrame(columns=["dataset","model","tau","acc_mean_pct_abs","acc_std_pct_abs"])

    acc = pd.to_numeric(atta["accuracy"], errors="coerce")
    if not is_percent_like(acc):
        acc = acc * 100.0
    atta["accuracy"] = acc.abs()

    ag = (
        atta
        .groupby(["dataset","model","tau"])["accuracy"]
        .agg(["mean","std"])
        .reset_index()
        .rename(columns={"mean":"acc_mean_pct_abs","std":"acc_std_pct_abs"})
    )
    return ag  # dataset,model,tau,acc_mean_pct_abs,acc_std_pct_abs

# Table assembly
def make_panel_for_dataset(dataset_name):
    # similarity from similarity_summary.csv
    sim_ag = build_similarity_panel(SIM_CSV, dataset_name)
    # attack from CSV (mean + std)
    atk_ag = build_attack_panel(ATTACK_CSV, dataset_name)

    # MPNet rows
    mpn = (
        sim_ag[sim_ag["metric"] == "MPNet"]
        [["dataset","model","tau","member_mean","nonmember_mean"]]
        .rename(columns={
            "member_mean":    "mpnet_member",
            "nonmember_mean": "mpnet_nonmember",
        })
    )

    # ROUGE-2 rows
    r2 = (
        sim_ag[sim_ag["metric"] == "ROUGE-2"]
        [["dataset","model","tau","member_mean","nonmember_mean"]]
        .rename(columns={
            "member_mean":    "rouge_member",
            "nonmember_mean": "rouge_nonmember",
        })
    )

    merged = pd.merge(mpn, r2, on=["dataset","model","tau"], how="outer")
    merged = pd.merge(merged, atk_ag, on=["dataset","model","tau"], how="outer")

    merged["Threat Model"] = merged["tau"].map(TAU_TO_THREAT)
    merged["Dataset"] = merged["dataset"].fillna(dataset_name)
    merged["Model"] = merged["model"]

    # format similarity means as plain decimals (no ±)
    def fmt_sim(x):
        if x is None or pd.isna(x):
            return ""
        return f"{float(x):.3f}"

    merged["MPNet (Member)"]      = merged["mpnet_member"].map(fmt_sim)
    merged["MPNet (Non-Member)"]  = merged["mpnet_nonmember"].map(fmt_sim)
    merged["ROUGE-2 (Member)"]    = merged["rouge_member"].map(fmt_sim)
    merged["ROUGE-2 (Non-Member)"]= merged["rouge_nonmember"].map(fmt_sim)

    # Attack-ACC as mean ± std in percent
    merged["Attack-ACC"] = [
        fmt_pm(mu, sd, dec=2, as_percent=True)
        for mu, sd in zip(merged.get("acc_mean_pct_abs"), merged.get("acc_std_pct_abs"))
    ]

    merged = merged[merged["Model"].isin(["BLIP","PaliGemma2","ViT-GPT2"])]
    merged = merged[merged["Threat Model"].isin(["BASELINE","NEURO","NEURO++"])]

    model_order  = {"BLIP":0, "PaliGemma2":1, "ViT-GPT2":2}
    threat_order = {"BASELINE":0, "NEURO":1, "NEURO++":2}
    merged = merged.sort_values(
        by=["Model","Threat Model"],
        key=lambda s: s.map(model_order if s.name=="Model" else threat_order)
    )

    out = merged[
        [
            "Dataset",
            "Model",
            "Threat Model",
            "MPNet (Member)",
            "MPNet (Non-Member)",
            "ROUGE-2 (Member)",
            "ROUGE-2 (Non-Member)",
            "Attack-ACC",
        ]
    ]

    os.makedirs(OUTDIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out.to_csv(os.path.join(OUTDIR, f"table_{dataset_name}_{ts}.csv"), index=False)

def main():
    for d in WANT_DATASETS:
        make_panel_for_dataset(d)

if __name__ == "__main__":
    main()