#!/usr/bin/env python3
"""Archive normalized VHH template/MSA campaign metrics for later comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "runs" / "vhh_template_campaign_stats_archive_20260620"

CAMPAIGNS = [
    {
        "campaign": "gpa33_top20_dominant_vhh",
        "target": "GPA33",
        "root": ROOT / "runs" / "gpa33_dominant_vhh_template_msa_comparison_20260619_181815",
        "comparison_csv": ROOT
        / "runs"
        / "gpa33_dominant_vhh_template_msa_comparison_20260619_181815"
        / "current_four_arm_vhh_information_comparison_with_pose_rmsd.csv",
        "pose_rmsd_csv": ROOT
        / "runs"
        / "gpa33_dominant_vhh_template_msa_comparison_20260619_181815"
        / "current_pose_rmsd_vs_msa_only_reference.csv",
        "summary_json": ROOT
        / "runs"
        / "gpa33_dominant_vhh_template_msa_comparison_20260619_181815"
        / "current_pose_rmsd_vs_msa_only_reference_summary.json",
        "report_md": ROOT
        / "runs"
        / "gpa33_dominant_vhh_template_msa_comparison_20260619_181815"
        / "current_pose_rmsd_vs_msa_only_reference_report.md",
        "reference_prefix": "msa_only_no_template",
        "arm_prefixes": {
            "target_msa_only_no_vhh_info": "target_msa_only_no_vhh_info",
            "vhh_template_no_vhh_msa": "vhh_template_no_vhh_msa",
            "vhh_msa_only_no_template": "msa_only_no_template",
            "vhh_template_plus_vhh_msa": "vhh_template_plus_vhh_msa",
        },
        "rmsd_columns": {
            "target_msa_only_no_vhh_info": (
                "target_msa_only_no_vhh_info_vhh_ca_pose_rmsd_vs_msa_only_no_template",
                "target_msa_only_no_vhh_info_target_ca_alignment_rmsd_vs_msa_only_no_template",
            ),
            "vhh_template_no_vhh_msa": (
                "vhh_template_no_vhh_msa_vhh_ca_pose_rmsd_vs_msa_only_no_template",
                "vhh_template_no_vhh_msa_target_ca_alignment_rmsd_vs_msa_only_no_template",
            ),
            "vhh_msa_only_no_template": (
                "msa_only_no_template_vhh_ca_pose_rmsd_vs_msa_only_no_template",
                "msa_only_no_template_target_ca_alignment_rmsd_vs_msa_only_no_template",
            ),
            "vhh_template_plus_vhh_msa": (
                "vhh_template_plus_vhh_msa_vhh_ca_pose_rmsd_vs_msa_only_no_template",
                "vhh_template_plus_vhh_msa_target_ca_alignment_rmsd_vs_msa_only_no_template",
            ),
        },
    },
    {
        "campaign": "ceacam6_top20_dominant_vhh",
        "target": "CEACAM6",
        "root": ROOT / "runs" / "ceacam6_top20_dominant_vhh_template_msa_comparison_20260620_000000",
        "comparison_csv": ROOT
        / "runs"
        / "ceacam6_top20_dominant_vhh_template_msa_comparison_20260620_000000"
        / "ceacam6_four_arm_vhh_information_comparison_with_pose_rmsd.csv",
        "pose_rmsd_csv": ROOT
        / "runs"
        / "ceacam6_top20_dominant_vhh_template_msa_comparison_20260620_000000"
        / "ceacam6_pose_rmsd_vs_msa_only_reference.csv",
        "summary_json": ROOT
        / "runs"
        / "ceacam6_top20_dominant_vhh_template_msa_comparison_20260620_000000"
        / "ceacam6_pose_rmsd_vs_msa_only_reference_summary.json",
        "report_md": ROOT
        / "runs"
        / "ceacam6_top20_dominant_vhh_template_msa_comparison_20260620_000000"
        / "ceacam6_vhh_information_and_pose_rmsd_report.md",
        "reference_prefix": "vhh_msa_only_no_template",
        "arm_prefixes": {
            "target_msa_only_no_vhh_info": "target_msa_only_no_vhh_info",
            "vhh_template_no_vhh_msa": "vhh_template_no_vhh_msa",
            "vhh_msa_only_no_template": "vhh_msa_only_no_template",
            "vhh_template_plus_vhh_msa": "vhh_template_plus_vhh_msa",
        },
        "rmsd_columns": {
            "target_msa_only_no_vhh_info": (
                "target_msa_only_no_vhh_info_vhh_pose_ca_rmsd_to_msa_only",
                "target_msa_only_no_vhh_info_target_alignment_ca_rmsd",
            ),
            "vhh_template_no_vhh_msa": (
                "vhh_template_no_vhh_msa_vhh_pose_ca_rmsd_to_msa_only",
                "vhh_template_no_vhh_msa_target_alignment_ca_rmsd",
            ),
            "vhh_msa_only_no_template": (
                "vhh_msa_only_no_template_vhh_pose_ca_rmsd_to_msa_only",
                "vhh_msa_only_no_template_target_alignment_ca_rmsd",
            ),
            "vhh_template_plus_vhh_msa": (
                "vhh_template_plus_vhh_msa_vhh_pose_ca_rmsd_to_msa_only",
                "vhh_template_plus_vhh_msa_target_alignment_ca_rmsd",
            ),
        },
    },
]

ARMS = {
    "target_msa_only_no_vhh_info": {
        "label": "Target MSA only; no VHH MSA/template",
        "target_msa": True,
        "vhh_msa": False,
        "vhh_template": False,
        "is_reference": False,
    },
    "vhh_template_no_vhh_msa": {
        "label": "Target MSA + VHH template; no VHH MSA",
        "target_msa": True,
        "vhh_msa": False,
        "vhh_template": True,
        "is_reference": False,
    },
    "vhh_msa_only_no_template": {
        "label": "Target MSA + VHH MSA; no VHH template",
        "target_msa": True,
        "vhh_msa": True,
        "vhh_template": False,
        "is_reference": True,
    },
    "vhh_template_plus_vhh_msa": {
        "label": "Target MSA + VHH MSA + VHH template",
        "target_msa": True,
        "vhh_msa": True,
        "vhh_template": True,
        "is_reference": False,
    },
}


def to_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def maybe_get(row: pd.Series, col: str) -> Any:
    return row[col] if col in row.index else np.nan


def normalize_campaign(campaign: dict[str, Any]) -> pd.DataFrame:
    source = pd.read_csv(campaign["comparison_csv"])
    rows: list[dict[str, Any]] = []
    reference_prefix = campaign["reference_prefix"]

    for _, row in source.iterrows():
        binder_name = maybe_get(row, "binder_name")
        binder_sequence = maybe_get(row, "binder_sequence")
        if pd.isna(binder_sequence):
            binder_sequence = maybe_get(row, "binder_seq")

        target_name = maybe_get(row, "target_name")
        if pd.isna(target_name):
            target_name = campaign["target"]

        reference_best_ipsae = to_float(maybe_get(row, f"{reference_prefix}_best_ipsae"))
        reference_best_iptm = to_float(maybe_get(row, f"{reference_prefix}_best_iptm"))

        for arm_key, prefix in campaign["arm_prefixes"].items():
            pose_col, align_col = campaign["rmsd_columns"][arm_key]
            rows.append(
                {
                    "campaign": campaign["campaign"],
                    "target": campaign["target"],
                    "target_name": target_name,
                    "binder_name": binder_name,
                    "binder_sequence": binder_sequence,
                    "source_best_ipsae": maybe_get(row, "source_best_ipsae"),
                    "source_run": maybe_get(row, "source_run"),
                    "source_pair_id": maybe_get(row, "source_pair_id"),
                    "rank_by_prior_best_ipsae": maybe_get(row, "rank_by_prior_best_ipsae"),
                    "arm_key": arm_key,
                    "arm_label": ARMS[arm_key]["label"],
                    "target_msa": ARMS[arm_key]["target_msa"],
                    "vhh_msa": ARMS[arm_key]["vhh_msa"],
                    "vhh_template": ARMS[arm_key]["vhh_template"],
                    "is_msa_only_reference": ARMS[arm_key]["is_reference"],
                    "best_ipsae": to_float(maybe_get(row, f"{prefix}_best_ipsae")),
                    "best_iptm": to_float(maybe_get(row, f"{prefix}_best_iptm")),
                    "ipsae_mean": to_float(maybe_get(row, f"{prefix}_ipsae_mean")),
                    "iptm_mean": to_float(maybe_get(row, f"{prefix}_iptm_mean")),
                    "pose_rmsd_vs_vhh_msa_only_reference": to_float(maybe_get(row, pose_col)),
                    "target_alignment_rmsd_vs_vhh_msa_only_reference": to_float(maybe_get(row, align_col)),
                    "reference_best_ipsae": reference_best_ipsae,
                    "reference_best_iptm": reference_best_iptm,
                    "reference_arm_key": "vhh_msa_only_no_template",
                    "source_comparison_csv": str(campaign["comparison_csv"]),
                }
            )

    return pd.DataFrame(rows)


def metric_summary(values: pd.Series, prefix: str) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_std": np.nan,
        }
    return {
        f"{prefix}_n": int(clean.shape[0]),
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_max": float(clean.max()),
        f"{prefix}_std": float(clean.std(ddof=0)),
    }


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "all": long_df,
        "reference_best_ipsae_ge_0.8": long_df[long_df["reference_best_ipsae"] >= 0.8],
    }

    for subset_name, subset_df in subsets.items():
        group_cols = ["campaign", "target", "arm_key", "arm_label"]
        for keys, group in subset_df.groupby(group_cols, dropna=False, sort=False):
            campaign, target, arm_key, arm_label = keys
            row: dict[str, Any] = {
                "subset": subset_name,
                "campaign": campaign,
                "target": target,
                "arm_key": arm_key,
                "arm_label": arm_label,
                "target_msa": bool(group["target_msa"].iloc[0]),
                "vhh_msa": bool(group["vhh_msa"].iloc[0]),
                "vhh_template": bool(group["vhh_template"].iloc[0]),
                "is_msa_only_reference": bool(group["is_msa_only_reference"].iloc[0]),
                "binder_count": int(group["binder_name"].nunique()),
                "best_ipsae_ge_0.8_count": int((group["best_ipsae"] >= 0.8).sum()),
                "best_ipsae_lt_0.5_count": int((group["best_ipsae"] < 0.5).sum()),
                "best_iptm_ge_0.8_count": int((group["best_iptm"] >= 0.8).sum()),
                "pose_rmsd_le_2a_count": int(
                    (group["pose_rmsd_vs_vhh_msa_only_reference"] <= 2.0).sum()
                ),
                "pose_rmsd_le_5a_count": int(
                    (group["pose_rmsd_vs_vhh_msa_only_reference"] <= 5.0).sum()
                ),
                "pose_rmsd_le_10a_count": int(
                    (group["pose_rmsd_vs_vhh_msa_only_reference"] <= 10.0).sum()
                ),
            }
            row.update(metric_summary(group["best_ipsae"], "best_ipsae"))
            row.update(metric_summary(group["best_iptm"], "best_iptm"))
            row.update(
                metric_summary(
                    group["pose_rmsd_vs_vhh_msa_only_reference"],
                    "pose_rmsd_vs_vhh_msa_only_reference",
                )
            )
            row.update(
                metric_summary(
                    group["target_alignment_rmsd_vs_vhh_msa_only_reference"],
                    "target_alignment_rmsd_vs_vhh_msa_only_reference",
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(outputs: dict[str, Path], long_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    manifest = {
        "archive_name": ARCHIVE_DIR.name,
        "created_for": "Final VHH template/MSA comparison table prep before miniprotein campaigns",
        "metric_definition": (
            "Pose RMSD is computed by aligning each comparison complex onto the "
            "VHH-MSA-only reference using target C-alpha atoms, then calculating "
            "VHH binder C-alpha RMSD."
        ),
        "reference_arm": "vhh_msa_only_no_template",
        "notes": [
            "Target MSA is enabled in every VHH scenario.",
            "VHH-MSA-only without a VHH template is treated as the pose RMSD reference.",
            "CEACAM6 has two low-confidence MSA-only reference failures; use the reference_best_ipsae_ge_0.8 subset for pose RMSD summaries when comparing against ground-truth-like poses.",
        ],
        "arm_definitions": ARMS,
        "campaigns": [
            {
                "campaign": c["campaign"],
                "target": c["target"],
                "root": str(c["root"]),
                "comparison_csv": str(c["comparison_csv"]),
                "pose_rmsd_csv": str(c["pose_rmsd_csv"]),
                "summary_json": str(c["summary_json"]),
                "report_md": str(c["report_md"]),
            }
            for c in CAMPAIGNS
        ],
        "outputs": {key: str(value) for key, value in outputs.items()},
        "row_counts": {
            "per_binder_long_rows": int(long_df.shape[0]),
            "per_binder_unique_campaign_binders": int(
                long_df[["campaign", "binder_name"]].drop_duplicates().shape[0]
            ),
            "arm_summary_rows": int(summary_df.shape[0]),
        },
    }
    outputs["manifest_json"].write_text(json.dumps(manifest, indent=2) + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def write_readme(outputs: dict[str, Path], summary_df: pd.DataFrame) -> None:
    focused = summary_df[summary_df["subset"] == "all"].copy()
    focused = focused[
        [
            "campaign",
            "target",
            "arm_key",
            "best_ipsae_median",
            "best_ipsae_mean",
            "best_ipsae_ge_0.8_count",
            "best_iptm_median",
            "pose_rmsd_vs_vhh_msa_only_reference_median",
            "pose_rmsd_le_5a_count",
        ]
    ]
    table = markdown_table(focused)

    high_conf = summary_df[summary_df["subset"] == "reference_best_ipsae_ge_0.8"].copy()
    high_conf = high_conf[
        [
            "campaign",
            "target",
            "arm_key",
            "binder_count",
            "pose_rmsd_vs_vhh_msa_only_reference_median",
            "pose_rmsd_le_5a_count",
        ]
    ]
    high_conf_table = markdown_table(high_conf)

    readme = f"""# VHH Template/MSA Campaign Stats Archive

This directory stores normalized statistics for the GPA33 and CEACAM6 dominant-framework VHH campaigns so a later final table can be generated without relying on chat context.

## Files

- `vhh_campaign_per_binder_long.csv`: one row per campaign, binder, and scenario.
- `vhh_campaign_arm_summary.csv`: campaign-level scenario summaries for all rows and for the `reference_best_ipsae_ge_0.8` subset.
- `vhh_campaign_source_manifest.json`: source paths, arm definitions, and metric definitions.

## Scenario Summary

{table}

## High-Confidence Reference Pose RMSD Subset

Pose RMSD uses VHH-MSA-only/no-template as the reference after target C-alpha alignment. This subset filters to binders where that reference arm had `best_ipSAE >= 0.8`.

{high_conf_table}

## Regeneration

Run:

```bash
{Path(__file__).name}
```

from `/Users/aaronring/batch_protenix/scripts`, or:

```bash
python /Users/aaronring/batch_protenix/scripts/{Path(__file__).name}
```
"""
    outputs["readme_md"].write_text(readme)


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    long_df = pd.concat([normalize_campaign(c) for c in CAMPAIGNS], ignore_index=True)
    sort_cols = ["campaign", "binder_name", "arm_key"]
    long_df = long_df.sort_values(sort_cols).reset_index(drop=True)

    summary_df = build_summary(long_df)
    summary_df = summary_df.sort_values(["subset", "campaign", "arm_key"]).reset_index(drop=True)

    outputs = {
        "per_binder_long_csv": ARCHIVE_DIR / "vhh_campaign_per_binder_long.csv",
        "arm_summary_csv": ARCHIVE_DIR / "vhh_campaign_arm_summary.csv",
        "manifest_json": ARCHIVE_DIR / "vhh_campaign_source_manifest.json",
        "readme_md": ARCHIVE_DIR / "README.md",
    }

    long_df.to_csv(outputs["per_binder_long_csv"], index=False)
    summary_df.to_csv(outputs["arm_summary_csv"], index=False)
    write_manifest(outputs, long_df, summary_df)
    write_readme(outputs, summary_df)

    print(f"Wrote VHH campaign stats archive to {ARCHIVE_DIR}")
    for output in outputs.values():
        print(output)


if __name__ == "__main__":
    main()
