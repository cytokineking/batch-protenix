#!/usr/bin/env python3
"""Compare paired on-target vs off-target summary metrics for the same binder set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "binder_name",
    "binder_seq",
    "status",
    "target_name",
    "iptm_mean",
    "ipsae_mean",
    "best_iptm",
    "best_ipsae",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-csv", required=True, help="pair_summary.csv for the on-target run")
    parser.add_argument("--off-csv", required=True, help="pair_summary.csv for the off-target or decoy run")
    parser.add_argument("--output-dir", required=True, help="Directory to write plots and tables")
    parser.add_argument("--on-label", default="On-target", help="Display label for the on-target run")
    parser.add_argument("--off-label", default="Off-target", help="Display label for the off-target run")
    return parser.parse_args()


def load_pair_summary(path: Path, suffix: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    status_counts = {str(k): int(v) for k, v in df["status"].value_counts(dropna=False).items()}
    df = df.loc[df["status"] == "success"].copy()
    if df.empty:
        raise ValueError(f"{path} has no successful rows")

    key_cols = ["binder_name", "binder_seq"]
    if df.duplicated(key_cols).any():
        dupes = df.loc[df.duplicated(key_cols, keep=False), key_cols].drop_duplicates()
        raise ValueError(f"{path} has duplicate binders after filtering success rows:\n{dupes}")

    keep = [
        "binder_name",
        "binder_seq",
        "target_name",
        "iptm_mean",
        "ipsae_mean",
        "best_iptm",
        "best_ipsae",
    ]
    renamed = {
        "target_name": f"target_name_{suffix}",
        "iptm_mean": f"iptm_mean_{suffix}",
        "ipsae_mean": f"ipsae_mean_{suffix}",
        "best_iptm": f"best_iptm_{suffix}",
        "best_ipsae": f"best_ipsae_{suffix}",
    }
    return df.loc[:, keep].rename(columns=renamed), status_counts


def summarize_metric(merged: pd.DataFrame, on_col: str, off_col: str) -> Dict[str, float]:
    on_vals = merged[on_col].to_numpy(dtype=float)
    off_vals = merged[off_col].to_numpy(dtype=float)
    delta = on_vals - off_vals
    corr = float(np.corrcoef(on_vals, off_vals)[0, 1]) if len(merged) > 1 else float("nan")
    return {
        "n": int(len(merged)),
        "on_mean": float(np.mean(on_vals)),
        "off_mean": float(np.mean(off_vals)),
        "on_median": float(np.median(on_vals)),
        "off_median": float(np.median(off_vals)),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "fraction_on_gt_off": float(np.mean(on_vals > off_vals)),
        "correlation": corr,
    }


def _setup_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)


def plot_scatter(
    merged: pd.DataFrame,
    on_col: str,
    off_col: str,
    metric_label: str,
    on_label: str,
    off_label: str,
    out_path: Path,
) -> None:
    on_vals = merged[on_col].to_numpy(dtype=float)
    off_vals = merged[off_col].to_numpy(dtype=float)
    corr = np.corrcoef(on_vals, off_vals)[0, 1] if len(merged) > 1 else np.nan
    lo = float(min(on_vals.min(), off_vals.min()))
    hi = float(max(on_vals.max(), off_vals.max()))
    pad = max((hi - lo) * 0.04, 0.02)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(on_vals, off_vals, s=26, alpha=0.55, edgecolor="none", color="#2369bd")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#666666", linewidth=1)
    _setup_axes(
        ax,
        title=f"{metric_label}: {on_label} vs {off_label}",
        xlabel=f"{on_label} {metric_label}",
        ylabel=f"{off_label} {metric_label}",
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(
        0.03,
        0.97,
        f"n = {len(merged)}\nr = {corr:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_violin_strip(
    merged: pd.DataFrame,
    on_col: str,
    off_col: str,
    metric_label: str,
    on_label: str,
    off_label: str,
    out_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    on_vals = merged[on_col].to_numpy(dtype=float)
    off_vals = merged[off_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    parts = ax.violinplot([on_vals, off_vals], positions=[1, 2], widths=0.72, showmeans=True, showextrema=False)
    colors = ["#2f7ed8", "#d97b22"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#222222")
        parts["cmeans"].set_linewidth(1.4)

    ax.scatter(1 + rng.normal(0, 0.035, len(on_vals)), on_vals, s=11, alpha=0.40, color=colors[0], edgecolor="none")
    ax.scatter(2 + rng.normal(0, 0.035, len(off_vals)), off_vals, s=11, alpha=0.40, color=colors[1], edgecolor="none")
    ax.set_xticks([1, 2], [on_label, off_label])
    _setup_axes(ax, title=f"{metric_label} distribution", xlabel="", ylabel=metric_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_histogram_overlay(
    merged: pd.DataFrame,
    on_col: str,
    off_col: str,
    metric_label: str,
    on_label: str,
    off_label: str,
    out_path: Path,
) -> None:
    on_vals = merged[on_col].to_numpy(dtype=float)
    off_vals = merged[off_col].to_numpy(dtype=float)
    bins = np.linspace(min(on_vals.min(), off_vals.min()), max(on_vals.max(), off_vals.max()), 35)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(on_vals, bins=bins, density=True, alpha=0.45, color="#2f7ed8", label=on_label)
    ax.hist(off_vals, bins=bins, density=True, alpha=0.45, color="#d97b22", label=off_label)
    _setup_axes(ax, title=f"{metric_label} histogram", xlabel=metric_label, ylabel="Density")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    on_csv = Path(args.on_csv)
    off_csv = Path(args.off_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    on_df, on_status = load_pair_summary(on_csv, "on")
    off_df, off_status = load_pair_summary(off_csv, "off")

    merged = on_df.merge(off_df, on=["binder_name", "binder_seq"], how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No binders overlapped between the two summary files")

    merged["iptm_mean_delta_on_minus_off"] = merged["iptm_mean_on"] - merged["iptm_mean_off"]
    merged["ipsae_mean_delta_on_minus_off"] = merged["ipsae_mean_on"] - merged["ipsae_mean_off"]
    merged["best_iptm_delta_on_minus_off"] = merged["best_iptm_on"] - merged["best_iptm_off"]
    merged["best_ipsae_delta_on_minus_off"] = merged["best_ipsae_on"] - merged["best_ipsae_off"]
    merged = merged.sort_values("iptm_mean_delta_on_minus_off", ascending=False).reset_index(drop=True)

    merged.to_csv(output_dir / "binder_target_decoy_comparison.csv", index=False)

    cols_for_rank = [
        "binder_name",
        "target_name_on",
        "target_name_off",
        "iptm_mean_on",
        "iptm_mean_off",
        "iptm_mean_delta_on_minus_off",
        "ipsae_mean_on",
        "ipsae_mean_off",
        "ipsae_mean_delta_on_minus_off",
    ]
    merged.loc[:, cols_for_rank].head(50).to_csv(output_dir / "top50_selective_by_iptm_delta.csv", index=False)
    merged.sort_values("ipsae_mean_delta_on_minus_off", ascending=False).loc[:, cols_for_rank].head(50).to_csv(
        output_dir / "top50_selective_by_ipsae_delta.csv",
        index=False,
    )

    summary = {
        "inputs": {
            "on_csv": str(on_csv.resolve()),
            "off_csv": str(off_csv.resolve()),
            "on_label": args.on_label,
            "off_label": args.off_label,
        },
        "status_counts": {
            "on": on_status,
            "off": off_status,
        },
        "merged_rows": int(len(merged)),
        "metrics": {
            "iptm_mean": summarize_metric(merged, "iptm_mean_on", "iptm_mean_off"),
            "ipsae_mean": summarize_metric(merged, "ipsae_mean_on", "ipsae_mean_off"),
            "best_iptm": summarize_metric(merged, "best_iptm_on", "best_iptm_off"),
            "best_ipsae": summarize_metric(merged, "best_ipsae_on", "best_ipsae_off"),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_scatter(
        merged,
        on_col="iptm_mean_on",
        off_col="iptm_mean_off",
        metric_label="Mean ipTM",
        on_label=args.on_label,
        off_label=args.off_label,
        out_path=output_dir / "mean_iptm_scatter.png",
    )
    plot_scatter(
        merged,
        on_col="ipsae_mean_on",
        off_col="ipsae_mean_off",
        metric_label="Mean ipSAE",
        on_label=args.on_label,
        off_label=args.off_label,
        out_path=output_dir / "mean_ipsae_scatter.png",
    )
    plot_violin_strip(
        merged,
        on_col="iptm_mean_on",
        off_col="iptm_mean_off",
        metric_label="Mean ipTM",
        on_label=args.on_label,
        off_label=args.off_label,
        out_path=output_dir / "mean_iptm_violin_strip.png",
    )
    plot_violin_strip(
        merged,
        on_col="ipsae_mean_on",
        off_col="ipsae_mean_off",
        metric_label="Mean ipSAE",
        on_label=args.on_label,
        off_label=args.off_label,
        out_path=output_dir / "mean_ipsae_violin_strip.png",
    )
    plot_histogram_overlay(
        merged,
        on_col="iptm_mean_on",
        off_col="iptm_mean_off",
        metric_label="Mean ipTM",
        on_label=args.on_label,
        off_label=args.off_label,
        out_path=output_dir / "mean_iptm_histogram.png",
    )
    plot_histogram_overlay(
        merged,
        on_col="ipsae_mean_on",
        off_col="ipsae_mean_off",
        metric_label="Mean ipSAE",
        on_label=args.on_label,
        off_label=args.off_label,
        out_path=output_dir / "mean_ipsae_histogram.png",
    )


if __name__ == "__main__":
    main()
