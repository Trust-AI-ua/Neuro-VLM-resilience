#!/usr/bin/env python3
"""Visualize member vs non-member similarity means by dataset and metric."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

SIM_CSV = Path(__file__).resolve().parents[1] / "similarity_summary.csv"

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

TAUS = [0, 2, 3]
TAU_TO_NAME = {0: "BASELINE", 2: "NEURO", 3: "NEURO++"}

def _lighten(color, amount=0.5):
    """Lighten an RGB(A) color by interpolating toward white."""
    r, g, b = color[:3]
    return (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )

def ensure_dir(path: Path) -> None:
    """Create ``path`` if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)

def norm_dataset(x: str) -> str:
    if not isinstance(x, str):
        return x
    t = x.strip().lower()
    if "coco" in t:
        return "COCO"
    if "nocaps" in t:
        return "NoCaps"
    if "cc3m" in t or "conceptual" in t:
        return "CC3M"
    return x

def norm_model(x: str) -> str:
    if not isinstance(x, str):
        return x
    t = x.strip().lower()
    if "blip" in t:
        return "BLIP"
    if "vit" in t and "gpt" in t:
        return "ViT-GPT2"
    if "pali" in t and "gemma" in t:
        return "PaliGemma2"
    return x

def norm_metric(x: str) -> str:
    if not isinstance(x, str):
        return x
    t = x.strip().lower()
    if "mpnet" in t:
        return "MPNet"
    if "rouge2" in t or "rouge-2" in t:
        return "ROUGE-2"
    return x

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(__file__).resolve().parent / "plots" / f"sim_means_by_ds_{ts}"
    ensure_dir(out_dir)

    s = pd.read_csv(SIM_CSV)
    need = {"dataset", "model", "tau", "set", "metric", "mean", "n"}
    missing = sorted(need - set(s.columns))
    if missing:
        raise ValueError(f"Missing columns in {SIM_CSV}: {missing}")

    # normalize columns and values
    s.columns = [c.strip().lower() for c in s.columns]
    s["dataset"] = s["dataset"].apply(norm_dataset)
    s["model"]   = s["model"].apply(norm_model)
    s["tau"]     = pd.to_numeric(s["tau"], errors="coerce")
    s["set"]     = s["set"].astype(str).str.strip().str.lower()
    s["metric"]  = s["metric"].apply(norm_metric)
    s["mean"]    = pd.to_numeric(s["mean"], errors="coerce")

    # filter to relevant rows
    allowed_models   = {"BLIP", "PaliGemma2", "ViT-GPT2"}
    allowed_datasets = {"COCO", "NoCaps", "CC3M"}
    allowed_metrics  = {"MPNet", "ROUGE-2"}

    s = s[s["tau"].isin(TAUS)]
    s = s[s["model"].isin(allowed_models)]
    s = s[s["dataset"].isin(allowed_datasets)]
    s = s[s["metric"].isin(allowed_metrics)]

    if s.empty:
        print("[warn] similarity_summary selection is empty; nothing to plot.")
        return

    # pivot to get member/nonmember means per (dataset, model, tau, metric)
    key = ["dataset", "model", "tau", "metric"]
    piv = s.pivot_table(index=key, columns="set", values="mean", aggfunc="mean")

    # ensure both columns exist
    for col in ["member", "nonmember"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv = piv.reset_index()

    # keep only rows with both member and nonmember defined
    df = piv.dropna(subset=["member", "nonmember"]).copy()
    if df.empty:
        print("[warn] No valid member/nonmember pairs; nothing to plot.")
        return

    # plotting order
    ds_order = [d for d in ["COCO", "NoCaps", "CC3M"] if d in df["dataset"].unique()]
    models   = [m for m in ["BLIP", "PaliGemma2", "ViT-GPT2"] if m in df["model"].unique()]
    colors   = plt.cm.tab10.colors
    model_color = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    # tau styles: only hatch/alpha differ, not color
    tau_style = {
        0: dict(alpha=0.95, hatch=None),
        2: dict(alpha=0.70, hatch='//'),
        3: dict(alpha=0.70, hatch='..'),
    }

    # Legends
    # Threat model (τ) legend — focus on hatch, not color
    tau_handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=(tau_style[tau]["hatch"] or ""),
            linewidth=0.8,
            label=TAU_TO_NAME[tau]
        )
        for tau in TAUS
    ]
    # Member vs non-member set legend — darker vs lighter blocks
    membership_handles = [
        Patch(facecolor="black", edgecolor="black", alpha=0.55, label="Member"),
        Patch(facecolor="0.7", edgecolor="0.35", alpha=0.85, label="Non-member"),
    ]

    # one figure per (dataset, metric)
    for ds in ds_order:
        g_ds = df[df["dataset"] == ds]
        if g_ds.empty:
            continue

        for metric in ["MPNet", "ROUGE-2"]:
            g_metric = g_ds[g_ds["metric"] == metric]
            if g_metric.empty:
                continue

            fig, ax = plt.subplots(figsize=(7.0, 3.8))
            title_metric = "MPNet" if metric == "MPNet" else "ROUGE-2"

            x_idx = np.arange(len(models))
            width = 0.22

            all_min_vals = []
            all_max_vals = []

            for i, m in enumerate(models):
                base_c = model_color[m]
                x0 = x_idx[i]

                for j, tau in enumerate(TAUS):
                    sub = g_metric[(g_metric["model"] == m) & (g_metric["tau"] == tau)]
                    if sub.empty:
                        continue

                    member_val = float(sub["member"].mean())
                    nonmember_val = float(sub["nonmember"].mean())
                    if np.isnan(member_val) or np.isnan(nonmember_val):
                        continue

                    # track full range for y-axis
                    all_min_vals.append(min(member_val, nonmember_val))
                    all_max_vals.append(max(member_val, nonmember_val))

                    # three taus: shift −w, 0, +w
                    shift = (j - 1) * width
                    slot_center = x0 + shift
                    bar_width = width * 0.9

                    sty = tau_style[tau]
                    nonmember_color = _lighten(base_c, 0.65)

                    # bottom segment: non-member mean
                    ax.bar(
                        slot_center,
                        nonmember_val,
                        bar_width,
                        color=nonmember_color,
                        alpha=max(0.25, sty["alpha"] * 0.65),
                        hatch=(sty["hatch"] or ""),
                        edgecolor=base_c,
                        linewidth=0.7,
                        label=None,
                    )

                    # top segment: (member - nonmember), stacked
                    stacked_height = max(member_val - nonmember_val, 0.0)
                    ax.bar(
                        slot_center,
                        stacked_height,
                        bar_width,
                        bottom=nonmember_val,
                        color=base_c,
                        alpha=sty["alpha"],
                        hatch=(sty["hatch"] or ""),
                        edgecolor=base_c,
                        linewidth=0.7,
                        label=None,
                    )

                    # if member < nonmember, visually mark the drop region
                    if member_val < nonmember_val:
                        ax.bar(
                            slot_center,
                            nonmember_val - member_val,
                            bar_width * 0.6,
                            bottom=member_val,
                            color="white",
                            edgecolor=base_c,
                            linewidth=0.6,
                            alpha=0.0,  # invisible fill, we mainly keep boundary
                            label=None,
                        )

                    # boundary line for non-member level
                    if nonmember_val > 0:
                        ax.hlines(
                            nonmember_val,
                            slot_center - bar_width / 2,
                            slot_center + bar_width / 2,
                            colors="black",
                            linewidth=0.4,
                            alpha=0.6,
                        )

            ax.set_xticks(x_idx)
            ax.set_xticklabels(models)
            ax.set_xlabel("Model")
            ax.set_ylabel("Member vs Non-member")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            # robust y-limits with padding
            if all_min_vals and all_max_vals:
                vmin = min(all_min_vals)
                vmax = max(all_max_vals)
                if vmin == vmax:
                    # degenerate case: flat values
                    ymin = max(0.0, vmin - 0.05)
                    ymax = min(1.0, vmax + 0.05)
                else:
                    span = vmax - vmin
                    margin = 0.15 * span
                    ymin = max(0.0, vmin - margin)
                    ymax = min(1.0, vmax + margin)

                # avoid zero-height range
                if ymax <= ymin:
                    ymax = min(1.0, ymin + 0.1)
                ax.set_ylim(ymin, ymax)

                fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))

                tau_legend = ax.legend(
                    tau_handles,
                    [h.get_label() for h in tau_handles],
                    title="Threat Model",
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0)
                )
                ax.add_artist(tau_legend)

                ax.legend(
                    membership_handles,
                    [h.get_label() for h in membership_handles],
                    title="Set",
                    frameon=False,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 0.55)
                )
            metric_tag = "mpnet" if metric == "MPNet" else "rouge2"
            fig.savefig(out_dir / f"{ds}_sim_means_{metric_tag}.png", dpi=300, bbox_inches="tight")
            fig.savefig(out_dir / f"{ds}_sim_means_{metric_tag}.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved similarity mean plots -> {out_dir}")

if __name__ == "__main__":
    main()