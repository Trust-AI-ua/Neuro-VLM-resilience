#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

ACC_CSV = "experiments/results/attack_accuracy.csv"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

# τ and g values we want in the ablation
TAUS_TO_PLOT = [0, 1, 2, 3, 5]
G_TO_PLOT    = [50, 100, 200]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def std_model(s):
    if not isinstance(s, str):
        return s
    x = s.strip().lower()
    if "vit" in x and "gpt" in x:
        return "ViT-GPT2"
    if "blip" in x:
        return "BLIP"
    if "pali" in x and "gemma" in x:
        return "PaliGemma2"
    return s

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"experiments/results/ablation_attack_acc_{ts}")
    ensure_dir(out_dir)

    df = pd.read_csv(ACC_CSV)
    df.columns = [c.lower() for c in df.columns]

    # numeric columns
    for c in ("g", "tau", "accuracy"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows missing the essentials
    df = df.dropna(subset=["dataset", "model", "metric", "g", "tau", "accuracy"]).copy()
    df["g"] = df["g"].astype(int)

    # only g in {50,100,200}
    df = df[df["g"].isin(G_TO_PLOT)].copy()

    # normalize dataset
    norm_ds = {"coco": "COCO", "nocaps": "NoCaps", "cc3m": "CC3M"}
    df["dataset"] = (
        df["dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(norm_ds)
        .fillna(df["dataset"])
    )

    # only COCO and NoCaps for ablation
    ablation_datasets = ["COCO", "NoCaps"]
    df = df[df["dataset"].isin(ablation_datasets)].copy()

    # τ in {0,1,2,3,5}
    df = df[df["tau"].isin(TAUS_TO_PLOT)].copy()

    # normalize models
    df["model"] = df["model"].apply(std_model)

    # only BLIP and ViT-GPT2
    allowed_models = {"BLIP", "ViT-GPT2"}
    df = df[df["model"].isin(allowed_models)]

    if df.empty:
        print("[warn] filtered attack_accuracy is empty; nothing to plot.")
        return

    ds_order = [d for d in ["COCO", "NoCaps"] if d in df["dataset"].unique()]
    models   = [m for m in ["BLIP", "ViT-GPT2"] if m in df["model"].unique()]

    # base colors: one per model
    base_colors = plt.cm.tab10.colors
    color_map = {m: base_colors[i % len(base_colors)] for i, m in enumerate(models)}

    # Extremely distinct and visible styles for each tau
    tau_style = {
        0: dict(
            linestyle="-",    # solid
            marker="o",       # circle
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=2,
            label="τ = 0"
        ),
        1: dict(
            linestyle="--",   # dashed
            marker="s",       # square
            markersize=9,
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="τ = 1"
        ),
        2: dict(
            linestyle=":",    # dotted
            marker="^",       # triangle
            markersize=11,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=1.8,
            label="τ = 2"
        ),
        3: dict(
            linestyle="solid",    #
            marker="x",       # x mark
            markersize=11,
            markerfacecolor="yellow",
            markeredgecolor="red",
            markeredgewidth=1.8,
            label="τ = 3"
        ),
        5: dict(
            linestyle="-.",   # dash-dot
            marker="D",       # diamond
            markersize=10,
            markerfacecolor="cyan",
            markeredgecolor="black",
            markeredgewidth=1.8,
            label="τ = 5"
        )
    }

    for ds in ds_order:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue

        present_g = sorted(set(sub["g"].unique()) & set(G_TO_PLOT))
        if not present_g:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 4.4))

        for m in models:
            dm = sub[sub["model"] == m]
            if dm.empty:
                continue
            base_c = color_map[m]

            for t in TAUS_TO_PLOT:
                if t not in tau_style:
                    continue

                dmt = dm[dm["tau"] == t].copy()
                if dmt.empty:
                    continue

                # aggregate over metrics at each g
                p = (
                    dmt.groupby("g")["accuracy"]
                    .mean()
                    .reindex(present_g)
                )

                if p.isna().all():
                    continue

                sty = tau_style[t]
                ax.plot(
                    p.index.values,
                    p.values,
                    linestyle=sty["linestyle"],
                    color=base_c,
                    marker=sty["marker"],
                    markersize=sty["markersize"],
                    markerfacecolor=sty["markerfacecolor"],
                    markeredgecolor=sty["markeredgecolor"],
                    markeredgewidth=sty["markeredgewidth"],
                    linewidth=2.2,
                    label=f"{m} — {sty['label']}",
                )

        # ax.set_title(f"{ds}") # kept for reference but not needed
        ax.set_xlabel("Granularity (g)")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(G_TO_PLOT)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # dedupe legend labels
        handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                h2.append(h)
                l2.append(l)

        fig.legend(
            h2,
            l2,
            loc="lower center",
            ncol=2,   # 8 entries → 2×4
            bbox_to_anchor=(0.5, -0.20),
            frameon=False,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        fig.savefig(out_dir / f"{ds}_ablation_attack_accuracy.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{ds}_ablation_attack_accuracy.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved ablation plots -> {out_dir}")

if __name__ == "__main__":
    main()