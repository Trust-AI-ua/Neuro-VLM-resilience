#!/usr/bin/env python3
"""Plot attack accuracy curves per dataset using ROC-AUC output."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACC_CSV = Path(__file__).resolve().parents[1] / "attack_accuracy.csv"

plt.rcParams.update({
    "font.size": 36,
    "axes.titlesize": 36,
    "axes.labelsize": 34,
    "legend.fontsize": 32,
    "figure.titlesize": 38,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "legend.title_fontsize": 22,
    "font.weight": "bold"
})

def ensure_dir(path: Path) -> None:
    """Create ``path`` if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)

def _darken(color, factor=0.6):
    r, g, b = color[:3]
    return (r * factor, g * factor, b * factor)

def std_model(s):
    if not isinstance(s, str): return s
    x = s.strip().lower()
    if "vit" in x and "gpt" in x: return "ViT-GPT2"
    if "blip" in x: return "BLIP"
    if "pali" in x and "gemma" in x: return "PaliGemma2"
    return s

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(__file__).resolve().parent / "plots" / f"attack_acc_{ts}"
    ensure_dir(out_dir)

    df = pd.read_csv(ACC_CSV)
    df.columns = [c.lower() for c in df.columns]

    for c in ("g","tau","accuracy"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dataset","model","metric","g","tau","accuracy"]).copy()
    df["g"] = df["g"].astype(int)

    norm_ds = {"coco":"COCO","nocaps":"NoCaps","cc3m":"CC3M"}
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower().map(norm_ds).fillna(df["dataset"])
    ds_order = [d for d in ["COCO","NoCaps","CC3M"] if d in df["dataset"].unique()]

    # τ in {0,2,3}
    df = df[df["tau"].isin([0,2,3])].copy()

    df["model"] = df["model"].apply(std_model)
    # keep only BLIP, ViT-GPT2, PaliGemma2
    allowed_models = {"BLIP", "ViT-GPT2", "PaliGemma2"}
    df = df[df["model"].isin(allowed_models)]

    ds_order = [d for d in ["COCO","NoCaps", "CC3M"] if d in df["dataset"].unique()]
    models = sorted(df["model"].unique())
    base_colors = plt.cm.tab10.colors
    color_map = {m: base_colors[i % len(base_colors)] for i,m in enumerate(models)}

    # line/label per tau; τ=3 gets darker color to be visually distinct
    tau_style = {
        0: dict(
            linestyle="-",    # solid
            marker="o",       # circle
            markersize=18,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=2,
            label="BASELINE"
        ),
        2: dict(
            linestyle=":",    # dotted
            marker="^",       # triangle
            markersize=18,
            markerfacecolor="yellow",
            markeredgecolor="black",
            markeredgewidth=1.8,
            label="NEURO"
        ),
        3: dict(
            linestyle="-.",   # dash-dot
            marker="D",       # diamond
            markersize=18,
            markerfacecolor="cyan",
            markeredgecolor="black",
            markeredgewidth=1.8,
            label="NEURO++"
        )
    }

    for ds in ds_order:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        g_order = sorted(sub["g"].unique())
        fig, ax = plt.subplots(figsize=(18, 12))  # Increased figure width for better spacing

        for m in models:
            dm = sub[sub["model"] == m]
            if dm.empty:
                continue
            base_c = color_map[m]
            for t, sty in tau_style.items():
                dmt = dm[dm["tau"] == t].copy()
                if dmt.empty:
                    continue
                p = dmt.groupby("g")["accuracy"].mean().reindex(g_order)
                if p.isna().all():
                    continue
                ax.plot(
                    p.index.values, p.values,
                    linestyle=sty["linestyle"],
                    color=base_c,
                    marker=sty["marker"],
                    markersize=sty["markersize"],
                    markerfacecolor=sty["markerfacecolor"],
                    markeredgecolor=sty["markeredgecolor"],
                    markeredgewidth=sty["markeredgewidth"],
                    linewidth=8,
                    label=f"{m} — {sty['label']}"
                )

        ax.set_xlabel("Granularity (g)")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.1, 0.2))  # Restored detailed y-axis divisions
        ax.set_xticks(np.arange(0, 201, 25))  # Expanded x-axis divisions
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # dedupe legend labels
        handles, labels = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        fig.legend(h2, l2, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.20), frameon=False)

        # Adjust the margin between the legend and the x-axis label
        fig.subplots_adjust(bottom=0.15)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}_attack_accuracy.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{ds}_attack_accuracy.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved -> {out_dir}")

if __name__ == "__main__":
    main()