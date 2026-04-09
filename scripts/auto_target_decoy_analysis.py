#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME = Path(tempfile.gettempdir()) / "xdg-cache"
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pair_csv_utils import short_hash
from scripts.compare_target_decoy_metrics import run_comparison


REQUIRED_COLUMNS = {
    "binder_name",
    "binder_seq",
    "status",
    "partner_role",
    "partner_name",
    "target_name",
    "target_seq",
    "iptm_mean",
    "ipsae_mean",
    "best_iptm",
    "best_ipsae",
}


def _slugify(text: str, max_len: int = 80) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or ""))
    slug = "-".join(filter(None, slug.split("-")))
    if not slug:
        slug = "x"
    return slug[:max_len]


def _seq_hash(seq: str) -> str:
    return short_hash(str(seq or ""), n=10)


def _target_group_dir(base_dir: Path, target_name: str, target_seq: str) -> Path:
    return base_dir / f"{_slugify(target_name)}__{_seq_hash(target_seq)}"


def _comparison_dir(
    base_dir: Path,
    target_name: str,
    target_seq: str,
    decoy_slot: str,
    decoy_name: str,
    decoy_seq: str,
) -> Path:
    return (
        _target_group_dir(base_dir, target_name, target_seq)
        / f"vs__{_slugify(decoy_slot)}__{_slugify(decoy_name)}__{_seq_hash(decoy_seq)}"
    )


def _load_pair_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns for automatic decoy analysis: {missing}")
    if "partner_slot" not in df.columns:
        df["partner_slot"] = df["partner_role"].fillna("").astype(str)
    if "partner_seq" not in df.columns:
        df["partner_seq"] = np.where(
            df["partner_role"].astype(str) == "target",
            df["target_seq"],
            "",
        )
    if "comparison_group_id" not in df.columns:
        df["comparison_group_id"] = ""
    for col in ("partner_role", "partner_slot", "partner_name", "partner_seq", "target_name", "target_seq"):
        df[col] = df[col].fillna("").astype(str)
    return df


def _grouped_metric_summary(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def _plot_multi_violin_strip(
    groups: Sequence[Tuple[str, np.ndarray]],
    *,
    metric_label: str,
    out_path: Path,
) -> None:
    labels = [label for label, _ in groups]
    arrays = [values.astype(float) for _, values in groups]
    positions = np.arange(1, len(arrays) + 1)
    colors = ["#2f7ed8", "#d97b22", "#5b8e3e", "#a44fb7", "#cc4e5c", "#7f7f7f"]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(arrays)), 6))
    parts = ax.violinplot(arrays, positions=positions, widths=0.72, showmeans=True, showextrema=False)
    for idx, body in enumerate(parts["bodies"]):
        color = colors[idx % len(colors)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#222222")
        parts["cmeans"].set_linewidth(1.4)

    for idx, values in enumerate(arrays, start=1):
        color = colors[(idx - 1) % len(colors)]
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(idx + jitter, values, s=11, alpha=0.40, color=color, edgecolor="none")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_title(f"{metric_label} distribution")
    ax.set_ylabel(metric_label)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["status"].astype(str) == "success"].copy()


def _successful_binder_keys(df: pd.DataFrame) -> set[Tuple[str, str]]:
    successful = _successful_rows(df)
    return {
        (str(row.binder_name), str(row.binder_seq))
        for row in successful.loc[:, ["binder_name", "binder_seq"]].itertuples(index=False)
    }


def _build_group_label(slot: str, name: str, *, target: bool = False) -> str:
    if target:
        return f"On-target ({name})" if name else "On-target"
    if name:
        return f"{slot} ({name})"
    return slot


def _run_pairwise_analyses(df: pd.DataFrame, analyses_dir: Path) -> List[Dict[str, object]]:
    comparisons: List[Dict[str, object]] = []
    successful = _successful_rows(df)
    successful_decoys = successful.loc[successful["partner_role"] == "decoy"].copy()
    if successful_decoys.empty:
        return comparisons

    grouped = successful_decoys.groupby(
        ["target_name", "target_seq", "partner_slot", "partner_name", "partner_seq"],
        dropna=False,
    )
    for (target_name, target_seq, decoy_slot, decoy_name, decoy_seq), _group in grouped:
        on_rows = df.loc[
            (df["partner_role"] == "target")
            & (df["target_name"] == str(target_name))
            & (df["target_seq"] == str(target_seq))
        ].copy()
        off_rows = df.loc[
            (df["partner_role"] == "decoy")
            & (df["target_name"] == str(target_name))
            & (df["target_seq"] == str(target_seq))
            & (df["partner_slot"] == str(decoy_slot))
            & (df["partner_name"] == str(decoy_name))
            & (df["partner_seq"] == str(decoy_seq))
        ].copy()

        on_success = int((on_rows["status"].astype(str) == "success").sum())
        off_success = int((off_rows["status"].astype(str) == "success").sum())
        overlap_success = len(_successful_binder_keys(on_rows) & _successful_binder_keys(off_rows))
        output_dir = _comparison_dir(
            analyses_dir,
            str(target_name),
            str(target_seq),
            str(decoy_slot),
            str(decoy_name),
            str(decoy_seq),
        )
        record: Dict[str, object] = {
            "target_name": str(target_name),
            "target_seq_hash": _seq_hash(str(target_seq)),
            "decoy_slot": str(decoy_slot),
            "decoy_name": str(decoy_name),
            "decoy_seq_hash": _seq_hash(str(decoy_seq)),
            "output_dir": str(output_dir.resolve()),
            "successful_on_rows": on_success,
            "successful_decoy_rows": off_success,
            "overlapping_successful_binders": int(overlap_success),
            "status": "pending",
        }

        if on_success <= 0:
            record["status"] = "skipped"
            record["skip_reason"] = "no_successful_on_target_rows"
            comparisons.append(record)
            continue
        if off_success <= 0:
            record["status"] = "skipped"
            record["skip_reason"] = "no_successful_decoy_rows"
            comparisons.append(record)
            continue
        if overlap_success <= 0:
            record["status"] = "skipped"
            record["skip_reason"] = "no_overlapping_successful_binders"
            comparisons.append(record)
            continue

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="target_decoy_compare_") as tmp_dir:
                tmp_root = Path(tmp_dir)
                on_csv = tmp_root / "on_pair_summary.csv"
                off_csv = tmp_root / "off_pair_summary.csv"
                on_rows.to_csv(on_csv, index=False)
                off_adapted = off_rows.copy()
                off_adapted["target_name"] = off_adapted["partner_name"]
                off_adapted.to_csv(off_csv, index=False)
                summary = run_comparison(
                    on_csv=on_csv,
                    off_csv=off_csv,
                    output_dir=output_dir,
                    on_label=_build_group_label("target", str(target_name), target=True),
                    off_label=_build_group_label(str(decoy_slot), str(decoy_name)),
                )
            record["status"] = "success"
            record["merged_rows"] = int(summary.get("merged_rows", 0))
        except Exception as exc:  # noqa: BLE001
            output_dir.mkdir(parents=True, exist_ok=True)
            error_path = output_dir / "error.json"
            error_path.write_text(json.dumps({"error": str(exc)}, indent=2) + "\n", encoding="utf-8")
            record["status"] = "error"
            record["error"] = str(exc)
        comparisons.append(record)
    return comparisons


def _run_aggregate_analyses(df: pd.DataFrame, analyses_dir: Path) -> List[Dict[str, object]]:
    aggregates: List[Dict[str, object]] = []
    successful = _successful_rows(df)
    if successful.empty:
        return aggregates

    target_groups = successful.groupby(["target_name", "target_seq"], dropna=False)
    for (target_name, target_seq), target_group in target_groups:
        successful_targets = target_group.loc[target_group["partner_role"] == "target"].copy()
        successful_decoys = target_group.loc[target_group["partner_role"] == "decoy"].copy()
        decoy_groups = list(successful_decoys.groupby(["partner_slot", "partner_name", "partner_seq"], dropna=False))
        if len(decoy_groups) <= 1:
            continue

        aggregate_dir = _target_group_dir(analyses_dir, str(target_name), str(target_seq)) / "_aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)

        metric_groups_iptm: List[Tuple[str, np.ndarray]] = []
        metric_groups_ipsae: List[Tuple[str, np.ndarray]] = []
        summary_rows: List[Dict[str, object]] = []

        on_iptm = successful_targets["iptm_mean"].astype(float).to_numpy()
        on_ipsae = successful_targets["ipsae_mean"].astype(float).to_numpy()
        metric_groups_iptm.append((_build_group_label("target", str(target_name), target=True), on_iptm))
        metric_groups_ipsae.append((_build_group_label("target", str(target_name), target=True), on_ipsae))
        summary_rows.append(
            {
                "slot": "target",
                "display_name": str(target_name),
                "seq_hash": _seq_hash(str(target_seq)),
                "iptm_mean": _grouped_metric_summary(on_iptm),
                "ipsae_mean": _grouped_metric_summary(on_ipsae),
            }
        )

        pooled_iptm_parts: List[np.ndarray] = []
        pooled_ipsae_parts: List[np.ndarray] = []
        for (slot, decoy_name, decoy_seq), decoy_group in sorted(
            decoy_groups,
            key=lambda item: (str(item[0][0]), str(item[0][1]), str(item[0][2])),
        ):
            iptm_values = decoy_group["iptm_mean"].astype(float).to_numpy()
            ipsae_values = decoy_group["ipsae_mean"].astype(float).to_numpy()
            label = _build_group_label(str(slot), str(decoy_name))
            metric_groups_iptm.append((label, iptm_values))
            metric_groups_ipsae.append((label, ipsae_values))
            pooled_iptm_parts.append(iptm_values)
            pooled_ipsae_parts.append(ipsae_values)
            summary_rows.append(
                {
                    "slot": str(slot),
                    "display_name": str(decoy_name),
                    "seq_hash": _seq_hash(str(decoy_seq)),
                    "iptm_mean": _grouped_metric_summary(iptm_values),
                    "ipsae_mean": _grouped_metric_summary(ipsae_values),
                }
            )

        pooled_iptm = np.concatenate(pooled_iptm_parts) if pooled_iptm_parts else np.array([], dtype=float)
        pooled_ipsae = np.concatenate(pooled_ipsae_parts) if pooled_ipsae_parts else np.array([], dtype=float)
        metric_groups_iptm.append(("pooled decoy", pooled_iptm))
        metric_groups_ipsae.append(("pooled decoy", pooled_ipsae))
        summary_rows.append(
            {
                "slot": "pooled_decoy",
                "display_name": "pooled decoy",
                "seq_hash": "pooled",
                "iptm_mean": _grouped_metric_summary(pooled_iptm),
                "ipsae_mean": _grouped_metric_summary(pooled_ipsae),
            }
        )

        _plot_multi_violin_strip(
            metric_groups_iptm,
            metric_label="Mean ipTM",
            out_path=aggregate_dir / "mean_iptm_violin_strip.png",
        )
        _plot_multi_violin_strip(
            metric_groups_ipsae,
            metric_label="Mean ipSAE",
            out_path=aggregate_dir / "mean_ipsae_violin_strip.png",
        )

        summary = {
            "kind": "aggregate_multi_decoy_distribution",
            "descriptive_only": True,
            "target_name": str(target_name),
            "target_seq_hash": _seq_hash(str(target_seq)),
            "category_count": int(len(summary_rows)),
            "categories": summary_rows,
        }
        (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        aggregates.append(
            {
                "status": "success",
                "target_name": str(target_name),
                "target_seq_hash": _seq_hash(str(target_seq)),
                "output_dir": str(aggregate_dir.resolve()),
                "category_count": int(len(summary_rows)),
            }
        )
    return aggregates


def _write_analysis_summary(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_automatic_analysis(pair_summary: Path, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_summary_path = Path(pair_summary)
    analyses_dir = Path(output_dir)
    df = _load_pair_summary(pair_summary_path)

    summary: Dict[str, object] = {
        "analysis_status": "pending",
        "pair_summary": str(pair_summary_path.resolve()),
        "analyses_dir": str(analyses_dir.resolve()),
        "comparisons": [],
        "aggregates": [],
    }

    if df.loc[df["partner_role"] == "decoy"].empty:
        summary["analysis_status"] = "skipped_no_decoys"
        _write_analysis_summary(analyses_dir / "analysis_summary.json", summary)
        return summary

    if _successful_rows(df.loc[df["partner_role"] == "decoy"]).empty:
        summary["analysis_status"] = "skipped_no_successful_decoys"
        _write_analysis_summary(analyses_dir / "analysis_summary.json", summary)
        return summary

    comparisons = _run_pairwise_analyses(df, analyses_dir)
    aggregates = _run_aggregate_analyses(df, analyses_dir)

    comparison_errors = [item for item in comparisons if item.get("status") == "error"]
    if comparison_errors:
        analysis_status = "complete_with_errors"
    else:
        analysis_status = "complete_success"

    summary.update(
        {
            "analysis_status": analysis_status,
            "comparison_count": int(len(comparisons)),
            "aggregate_count": int(len(aggregates)),
            "comparisons": comparisons,
            "aggregates": aggregates,
        }
    )
    _write_analysis_summary(analyses_dir / "analysis_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic per-decoy and aggregate target/decoy analyses.")
    parser.add_argument("--pair-summary", required=True, help="pair_summary.csv path")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_automatic_analysis(Path(args.pair_summary), Path(args.output_dir))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
