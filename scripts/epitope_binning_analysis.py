#!/usr/bin/env python3
"""Epitope binning and geometric analysis for predicted binder-target complexes."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gemmi
import numpy as np
import pandas as pd

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME = Path(tempfile.gettempdir()) / "xdg-cache"
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PAIR_SUMMARY_REQUIRED_COLUMNS = {
    "pair_id",
    "binder_name",
    "binder_seq",
    "target_name",
    "status",
    "best_ipsae",
    "ipsae_mean",
    "best_iptm",
    "best_pdockq2",
}

OFFTARGET_REQUIRED_COLUMNS = {
    "binder_name",
    "best_ipsae_on",
    "best_ipsae_off",
    "best_ipsae_delta_on_minus_off",
    "ipsae_mean_on",
    "ipsae_mean_off",
}

DEFAULT_JACCARD_DISTANCE_THRESHOLD = 0.55
DEFAULT_APPROACH_ANGLE_THRESHOLD_DEG = 35.0
DEFAULT_INTERFACE_CENTROID_THRESHOLD = 6.0


@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    seq_num: int
    insertion_code: str
    res_name: str

    @property
    def label(self) -> str:
        icode = self.insertion_code if self.insertion_code else ""
        return f"{self.chain_id}:{self.seq_num}{icode}:{self.res_name}"


@dataclass
class ComplexRecord:
    pair_id: str
    binder_name: str
    binder_seq: str
    target_name: str
    best_ipsae: float
    ipsae_mean: float
    best_iptm: float
    best_pdockq2: float
    passes_ipsae_threshold: bool
    cif_path: Path
    target_residue_keys: Tuple[ResidueKey, ...]
    target_ca_positions: np.ndarray
    target_contact_vector: np.ndarray
    target_residue_min_distances: np.ndarray
    contacted_residue_indices: np.ndarray
    contacted_residue_labels: List[str]
    n_contacted_residues: int
    target_interface_centroid: np.ndarray
    binder_interface_centroid: np.ndarray
    target_alignment_rmsd_before: float = float("nan")
    target_alignment_rmsd_after: float = float("nan")
    aligned_target_interface_centroid: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, dtype=float))
    aligned_binder_interface_centroid: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, dtype=float))
    approach_vector: np.ndarray = field(default_factory=lambda: np.full(3, np.nan, dtype=float))
    primary_cluster_index: int = -1
    primary_bin_id: str = ""
    sub_bin_index: int = -1
    sub_bin_id: str = ""
    final_bin_id: str = ""
    is_representative: bool = False
    transform: Optional[gemmi.Transform] = None


@dataclass
class OffTargetContext:
    frame: pd.DataFrame
    key_cols: List[str]
    direct_attach: bool
    offtarget_label_override: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-dir", required=True, help="Grouped by_target/<target> directory containing mmCIF files")
    parser.add_argument("--pair-summary", required=True, help="pair_summary.csv path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--comparison-csv", help="Optional off-target / decoy comparison CSV")
    parser.add_argument("--offtarget-label", help="Optional display label override for off-target annotations")
    parser.add_argument("--binder-chain", default="A", help="Binder chain ID in the grouped mmCIF files")
    parser.add_argument("--target-chain", default="B", help="Target chain ID in the grouped mmCIF files")
    parser.add_argument("--contact-cutoff", type=float, default=4.5, help="Heavy-atom contact cutoff in Angstrom")
    parser.add_argument(
        "--ipsae-threshold",
        type=float,
        default=0.6,
        help=(
            "Threshold used for filtering in default mode and for pass/fail annotation in "
            "--include-all-successful mode"
        ),
    )
    parser.add_argument(
        "--include-all-successful",
        action="store_true",
        help="Analyze all successful complexes instead of filtering by best_ipsae",
    )
    parser.add_argument("--target-name", help="Optional target name validation helper")
    parser.add_argument("--linkage", default="average", choices=["average"], help="Primary clustering linkage mode")
    parser.add_argument(
        "--jaccard-distance-threshold",
        type=float,
        default=DEFAULT_JACCARD_DISTANCE_THRESHOLD,
        help="Distance threshold for primary Jaccard clustering",
    )
    parser.add_argument(
        "--approach-angle-threshold-deg",
        type=float,
        default=DEFAULT_APPROACH_ANGLE_THRESHOLD_DEG,
        help="Maximum pairwise approach-vector angle within a sub-bin",
    )
    parser.add_argument(
        "--interface-centroid-threshold",
        type=float,
        default=DEFAULT_INTERFACE_CENTROID_THRESHOLD,
        help="Maximum aligned target-interface centroid separation within a sub-bin",
    )
    parser.add_argument(
        "--representative-mode",
        default="highest_ipsae",
        choices=["highest_ipsae", "medoid"],
        help="Representative selection mode for final bins",
    )
    parser.add_argument(
        "--write-aligned-cifs",
        dest="write_aligned_cifs",
        action="store_true",
        default=True,
        help="Write target-aligned representative mmCIFs under the output directory (default: enabled)",
    )
    parser.add_argument(
        "--no-write-aligned-cifs",
        dest="write_aligned_cifs",
        action="store_false",
        help="Disable writing target-aligned representative mmCIFs",
    )
    parser.add_argument(
        "--min-bin-size",
        type=int,
        default=1,
        help="Reporting-only minimum final-bin size for bin summaries and JSON exports",
    )
    return parser.parse_args()


def load_pair_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(PAIR_SUMMARY_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required pair_summary columns: {missing}")
    return df


def pair_id_to_grouped_cif_name(pair_id: str) -> str:
    if "_" not in pair_id:
        raise ValueError(f"Could not derive grouped mmCIF name from pair_id without row prefix: {pair_id}")
    return f"{pair_id.split('_', 1)[1]}.cif"


def residue_key_from_residue(chain_id: str, residue: gemmi.Residue) -> ResidueKey:
    insertion_code = str(residue.seqid.icode).strip()
    return ResidueKey(
        chain_id=chain_id,
        seq_num=int(residue.seqid.num),
        insertion_code=insertion_code,
        res_name=str(residue.name),
    )


def atom_position_array(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)


def position_from_array(coords: Sequence[float]) -> gemmi.Position:
    return gemmi.Position(float(coords[0]), float(coords[1]), float(coords[2]))


def transform_array(transform: gemmi.Transform, coords: np.ndarray) -> np.ndarray:
    pos = transform.apply(position_from_array(coords))
    return np.array([pos.x, pos.y, pos.z], dtype=float)


def is_heavy_atom(atom: gemmi.Atom) -> bool:
    return int(atom.element.atomic_number) > 1


def collect_chain_residues(chain: gemmi.Chain) -> List[gemmi.Residue]:
    residues = list(chain.get_polymer())
    if not residues:
        residues = list(chain)
    return residues


def find_residue_ca_position(residue: gemmi.Residue) -> np.ndarray:
    atom = residue.find_atom("CA", "\x00")
    if not atom:
        raise ValueError(f"Residue {residue.name} {residue.seqid} is missing a CA atom")
    return atom_position_array(atom)


def parse_complex_record(
    row: pd.Series,
    cif_dir: Path,
    binder_chain_id: str,
    target_chain_id: str,
    contact_cutoff: float,
    reference_target_keys: Optional[Tuple[ResidueKey, ...]] = None,
) -> ComplexRecord:
    cif_name = pair_id_to_grouped_cif_name(str(row["pair_id"]))
    cif_path = cif_dir / cif_name
    if not cif_path.exists():
        raise FileNotFoundError(f"Expected grouped mmCIF not found for pair_id {row['pair_id']}: {cif_path}")

    structure = gemmi.read_structure(str(cif_path))
    if len(structure) == 0:
        raise ValueError(f"{cif_path} contains no models")
    model = structure[0]
    try:
        binder_chain = model[binder_chain_id]
    except Exception as exc:
        raise ValueError(f"{cif_path} is missing binder chain {binder_chain_id}") from exc
    try:
        target_chain = model[target_chain_id]
    except Exception as exc:
        raise ValueError(f"{cif_path} is missing target chain {target_chain_id}") from exc

    target_residues = collect_chain_residues(target_chain)
    if not target_residues:
        raise ValueError(f"{cif_path} has no polymer residues for target chain {target_chain_id}")

    target_residue_keys: List[ResidueKey] = []
    target_ca_positions: List[np.ndarray] = []
    target_residue_heavy_positions: List[np.ndarray] = []
    target_key_to_index: Dict[ResidueKey, int] = {}
    for idx, residue in enumerate(target_residues):
        key = residue_key_from_residue(target_chain_id, residue)
        target_residue_keys.append(key)
        target_key_to_index[key] = idx
        target_ca_positions.append(find_residue_ca_position(residue))
        heavy_positions = [atom_position_array(atom) for atom in residue if is_heavy_atom(atom)]
        if not heavy_positions:
            raise ValueError(f"{cif_path} target residue {key.label} has no heavy atoms")
        target_residue_heavy_positions.append(np.vstack(heavy_positions))

    target_residue_keys_tuple = tuple(target_residue_keys)
    if reference_target_keys is not None and target_residue_keys_tuple != reference_target_keys:
        raise ValueError(f"{cif_path} target residue indexing does not match the retained-set reference residue index")

    neighbor_search = gemmi.NeighborSearch(model, structure.cell, contact_cutoff).populate()
    target_residue_min_distances = np.full(len(target_residue_keys), np.inf, dtype=float)
    binder_interface_positions: List[np.ndarray] = []

    binder_residues = collect_chain_residues(binder_chain)
    if not binder_residues:
        raise ValueError(f"{cif_path} has no polymer residues for binder chain {binder_chain_id}")

    for residue in binder_residues:
        for atom in residue:
            if not is_heavy_atom(atom):
                continue
            atom_contacts_target = False
            for mark in neighbor_search.find_neighbors(atom, min_dist=0.1, max_dist=contact_cutoff):
                cra = mark.to_cra(model)
                if cra.chain.name != target_chain_id:
                    continue
                if not is_heavy_atom(cra.atom):
                    continue
                key = residue_key_from_residue(target_chain_id, cra.residue)
                target_idx = target_key_to_index.get(key)
                if target_idx is None:
                    continue
                dist = float(atom.pos.dist(mark.pos))
                if dist < target_residue_min_distances[target_idx]:
                    target_residue_min_distances[target_idx] = dist
                atom_contacts_target = True
            if atom_contacts_target:
                binder_interface_positions.append(atom_position_array(atom))

    contacted_residue_indices = np.flatnonzero(np.isfinite(target_residue_min_distances))
    if len(contacted_residue_indices) == 0:
        raise ValueError(f"{cif_path} produced zero target contacts at {contact_cutoff:.2f} A")
    if not binder_interface_positions:
        raise ValueError(f"{cif_path} produced zero binder interface atoms at {contact_cutoff:.2f} A")

    target_contact_vector = np.zeros(len(target_residue_keys), dtype=bool)
    target_contact_vector[contacted_residue_indices] = True
    contacted_residue_labels = [target_residue_keys[idx].label for idx in contacted_residue_indices]
    target_interface_coords = np.vstack([target_residue_heavy_positions[idx] for idx in contacted_residue_indices])
    binder_interface_coords = np.vstack(binder_interface_positions)

    return ComplexRecord(
        pair_id=str(row["pair_id"]),
        binder_name=str(row["binder_name"]),
        binder_seq=str(row["binder_seq"]),
        target_name=str(row["target_name"]),
        best_ipsae=float(row["best_ipsae"]),
        ipsae_mean=float(row["ipsae_mean"]),
        best_iptm=float(row["best_iptm"]),
        best_pdockq2=float(row["best_pdockq2"]),
        passes_ipsae_threshold=bool(row["passes_ipsae_threshold"]),
        cif_path=cif_path,
        target_residue_keys=target_residue_keys_tuple,
        target_ca_positions=np.vstack(target_ca_positions),
        target_contact_vector=target_contact_vector,
        target_residue_min_distances=target_residue_min_distances,
        contacted_residue_indices=contacted_residue_indices,
        contacted_residue_labels=contacted_residue_labels,
        n_contacted_residues=int(len(contacted_residue_indices)),
        target_interface_centroid=target_interface_coords.mean(axis=0),
        binder_interface_centroid=binder_interface_coords.mean(axis=0),
    )


def compute_pre_alignment_rmsd(moving: np.ndarray, fixed: np.ndarray) -> float:
    diff = moving - fixed
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or not np.isfinite(norm):
        return np.full_like(vec, np.nan, dtype=float)
    return vec / norm


def jaccard_distance_matrix(contact_matrix: np.ndarray) -> np.ndarray:
    binary = contact_matrix.astype(np.int16, copy=False)
    intersections = binary @ binary.T
    row_sums = binary.sum(axis=1, keepdims=True)
    unions = row_sums + row_sums.T - intersections
    distances = np.zeros_like(intersections, dtype=float)
    valid = unions > 0
    distances[valid] = 1.0 - (intersections[valid] / unions[valid])
    np.fill_diagonal(distances, 0.0)
    return distances


def average_linkage_threshold_clusters(distance_matrix: np.ndarray, threshold: float) -> np.ndarray:
    n_items = distance_matrix.shape[0]
    if n_items == 0:
        return np.array([], dtype=int)
    if n_items == 1:
        return np.array([0], dtype=int)

    work = np.array(distance_matrix, dtype=float, copy=True)
    np.fill_diagonal(work, np.inf)
    active = np.ones(n_items, dtype=bool)
    sizes = np.ones(n_items, dtype=float)
    members: Dict[int, List[int]] = {idx: [idx] for idx in range(n_items)}

    while True:
        flat_index = int(np.argmin(work))
        min_distance = float(work.flat[flat_index])
        if not np.isfinite(min_distance) or min_distance > threshold:
            break
        left, right = divmod(flat_index, n_items)
        if left == right or not active[left] or not active[right]:
            work[left, right] = np.inf
            work[right, left] = np.inf
            continue
        if right < left:
            left, right = right, left
        active_mask = active.copy()
        active_mask[left] = False
        active_mask[right] = False
        if active_mask.any():
            merged = (sizes[left] * work[left, active_mask] + sizes[right] * work[right, active_mask]) / (
                sizes[left] + sizes[right]
            )
            work[left, active_mask] = merged
            work[active_mask, left] = merged
        work[left, left] = np.inf
        work[right, :] = np.inf
        work[:, right] = np.inf
        active[right] = False
        sizes[left] += sizes[right]
        members[left].extend(members.pop(right))

    labels = np.full(n_items, -1, dtype=int)
    active_roots = [idx for idx in range(n_items) if active[idx]]
    active_roots.sort(key=lambda idx: (-len(members[idx]), idx))
    for cluster_index, root in enumerate(active_roots):
        for item_idx in members[root]:
            labels[item_idx] = cluster_index
    return labels


def complete_linkage_threshold_clusters(distance_matrix: np.ndarray, threshold: float) -> np.ndarray:
    n_items = distance_matrix.shape[0]
    if n_items == 0:
        return np.array([], dtype=int)
    if n_items == 1:
        return np.array([0], dtype=int)

    work = np.array(distance_matrix, dtype=float, copy=True)
    np.fill_diagonal(work, np.inf)
    active = np.ones(n_items, dtype=bool)
    members: Dict[int, List[int]] = {idx: [idx] for idx in range(n_items)}

    while True:
        flat_index = int(np.argmin(work))
        min_distance = float(work.flat[flat_index])
        if not np.isfinite(min_distance) or min_distance > threshold:
            break
        left, right = divmod(flat_index, n_items)
        if left == right or not active[left] or not active[right]:
            work[left, right] = np.inf
            work[right, left] = np.inf
            continue
        if right < left:
            left, right = right, left

        active_mask = active.copy()
        active_mask[left] = False
        active_mask[right] = False
        if active_mask.any():
            merged = np.maximum(work[left, active_mask], work[right, active_mask])
            work[left, active_mask] = merged
            work[active_mask, left] = merged
        work[left, left] = np.inf
        work[right, :] = np.inf
        work[:, right] = np.inf
        active[right] = False
        members[left].extend(members.pop(right))

    labels = np.full(n_items, -1, dtype=int)
    active_roots = [idx for idx in range(n_items) if active[idx]]
    active_roots.sort(key=lambda idx: (-len(members[idx]), idx))
    for cluster_index, root in enumerate(active_roots):
        for item_idx in members[root]:
            labels[item_idx] = cluster_index
    return labels


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 180.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0 or not np.isfinite(denom):
        return 180.0
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(math.acos(cosine)))


def secondary_geometry_distance_matrix(
    vectors: np.ndarray,
    centroids: np.ndarray,
    angle_threshold_deg: float,
    centroid_threshold: float,
) -> np.ndarray:
    n_items = len(vectors)
    matrix = np.zeros((n_items, n_items), dtype=float)
    safe_angle = max(float(angle_threshold_deg), 1e-6)
    safe_centroid = max(float(centroid_threshold), 1e-6)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            angle = angle_between_vectors(vectors[i], vectors[j])
            centroid_dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            matrix[i, j] = matrix[j, i] = max(angle / safe_angle, centroid_dist / safe_centroid)
    return matrix


def pairwise_values(values: Sequence[np.ndarray], fn) -> List[float]:
    pairs: List[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            pairs.append(float(fn(values[i], values[j])))
    return pairs


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def build_final_bin_distance_matrix(
    records: Sequence[ComplexRecord],
    angle_threshold_deg: float,
    centroid_threshold: float,
) -> np.ndarray:
    n_items = len(records)
    if n_items <= 1:
        return np.zeros((n_items, n_items), dtype=float)
    contact_matrix = np.vstack([record.target_contact_vector.astype(int) for record in records])
    contact_distance = jaccard_distance_matrix(contact_matrix)
    geometry_distance = secondary_geometry_distance_matrix(
        np.vstack([record.approach_vector for record in records]),
        np.vstack([record.aligned_target_interface_centroid for record in records]),
        angle_threshold_deg,
        centroid_threshold,
    )
    return 0.5 * (contact_distance + geometry_distance)


def choose_representative(
    records: List[ComplexRecord],
    mode: str,
    local_distance_matrix: Optional[np.ndarray] = None,
) -> ComplexRecord:
    if mode == "highest_ipsae":
        return max(records, key=lambda record: (record.best_ipsae, record.ipsae_mean, record.best_iptm))
    if len(records) == 1:
        return records[0]
    if local_distance_matrix is None:
        raise ValueError("local_distance_matrix is required for medoid representative selection")
    medoid_local = int(np.argmin(np.mean(local_distance_matrix, axis=1)))
    return records[medoid_local]


def maybe_load_offtarget_context(
    comparison_csv: Optional[Path],
    retained_df: pd.DataFrame,
    target_name: str,
    offtarget_label_override: Optional[str],
) -> Optional[OffTargetContext]:
    if comparison_csv is None:
        return None
    df = pd.read_csv(comparison_csv)
    missing = sorted(OFFTARGET_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{comparison_csv} is missing required comparison columns: {missing}")

    if "target_name_on" in df.columns:
        target_mask = df["target_name_on"].astype(str) == str(target_name)
        if not target_mask.any():
            raise ValueError(
                f"{comparison_csv} contains no rows with target_name_on matching retained target {target_name!r}"
            )
        df = df.loc[target_mask].copy()

    if "target_name_off" not in df.columns:
        df["target_name_off"] = offtarget_label_override or "offtarget"
    df["target_name_off"] = df["target_name_off"].fillna(offtarget_label_override or "offtarget").astype(str)

    key_cols = ["binder_name"]
    if "binder_seq" in df.columns and "binder_seq" in retained_df.columns:
        key_cols.append("binder_seq")

    merge_keys = retained_df.loc[:, key_cols].drop_duplicates()
    df = merge_keys.merge(df, on=key_cols, how="inner")
    if df.empty:
        return OffTargetContext(
            frame=df,
            key_cols=key_cols,
            direct_attach=True,
            offtarget_label_override=offtarget_label_override,
        )

    duplicate_key_cols = key_cols + ["target_name_off"]
    if df.duplicated(duplicate_key_cols).any():
        dupes = df.loc[df.duplicated(duplicate_key_cols, keep=False), duplicate_key_cols].drop_duplicates()
        raise ValueError(
            f"{comparison_csv} contains duplicate rows for the same binder/off-target key after filtering:\n{dupes}"
        )

    counts_per_binder = df.groupby(key_cols, dropna=False).size()
    direct_attach = bool(counts_per_binder.max() <= 1)
    return OffTargetContext(
        frame=df.reset_index(drop=True),
        key_cols=key_cols,
        direct_attach=direct_attach,
        offtarget_label_override=offtarget_label_override,
    )


def apply_direct_offtarget_merge(base_df: pd.DataFrame, context: Optional[OffTargetContext]) -> pd.DataFrame:
    if context is None or context.frame.empty or not context.direct_attach:
        return base_df.copy()
    merge_df = context.frame.copy()
    merge_df["offtarget_label"] = (
        context.offtarget_label_override if context.offtarget_label_override else merge_df["target_name_off"]
    )
    return base_df.merge(merge_df, on=context.key_cols, how="left", validate="one_to_one")


def make_core_dataframe(records: Sequence[ComplexRecord], ipsae_threshold: float) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "pair_id": record.pair_id,
                "binder_name": record.binder_name,
                "binder_seq": record.binder_seq,
                "target_name": record.target_name,
                "best_ipsae": record.best_ipsae,
                "ipsae_mean": record.ipsae_mean,
                "best_iptm": record.best_iptm,
                "best_pdockq2": record.best_pdockq2,
                "ipsae_threshold_used": float(ipsae_threshold),
                "passes_ipsae_threshold": bool(record.passes_ipsae_threshold),
                "cif_path": str(record.cif_path),
            }
        )
    return pd.DataFrame(rows).sort_values(["best_ipsae", "binder_name"], ascending=[False, True]).reset_index(drop=True)


def write_target_residue_occupancy(
    records: Sequence[ComplexRecord],
    target_residue_keys: Sequence[ResidueKey],
    out_path: Path,
) -> pd.DataFrame:
    contact_matrix = np.vstack([record.target_contact_vector.astype(int) for record in records])
    min_dist_matrix = np.vstack([record.target_residue_min_distances for record in records])
    rows = []
    n_records = len(records)
    for idx, key in enumerate(target_residue_keys):
        residue_dists = min_dist_matrix[:, idx]
        contacting = np.isfinite(residue_dists)
        rows.append(
            {
                "target_chain": key.chain_id,
                "auth_seq_id": key.seq_num,
                "insertion_code": key.insertion_code,
                "res_name": key.res_name,
                "residue_label": key.label,
                "contact_count": int(contacting.sum()),
                "contact_frequency": float(contacting.mean()) if n_records else float("nan"),
                "mean_min_distance": float(np.mean(residue_dists[contacting])) if contacting.any() else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def assign_primary_bins(records: List[ComplexRecord], threshold: float) -> np.ndarray:
    contact_matrix = np.vstack([record.target_contact_vector for record in records])
    distance_matrix = jaccard_distance_matrix(contact_matrix)
    labels = average_linkage_threshold_clusters(distance_matrix, threshold)
    for idx, record in enumerate(records):
        record.primary_cluster_index = int(labels[idx])
    primary_sizes = Counter(labels)
    ordered_clusters = [cluster for cluster, _ in sorted(primary_sizes.items(), key=lambda item: (-item[1], item[0]))]
    cluster_to_bin = {cluster: f"P{order + 1:03d}" for order, cluster in enumerate(ordered_clusters)}
    for record in records:
        record.primary_bin_id = cluster_to_bin[record.primary_cluster_index]
    return distance_matrix


def align_records(records: List[ComplexRecord]) -> ComplexRecord:
    reference = max(records, key=lambda record: (record.best_ipsae, record.ipsae_mean, record.best_iptm))
    fixed_positions = [position_from_array(coords) for coords in reference.target_ca_positions]
    for record in records:
        moving_positions = [position_from_array(coords) for coords in record.target_ca_positions]
        result = gemmi.superpose_positions(fixed_positions, moving_positions)
        record.transform = result.transform
        record.target_alignment_rmsd_before = compute_pre_alignment_rmsd(record.target_ca_positions, reference.target_ca_positions)
        record.target_alignment_rmsd_after = float(result.rmsd)
        record.aligned_target_interface_centroid = transform_array(result.transform, record.target_interface_centroid)
        record.aligned_binder_interface_centroid = transform_array(result.transform, record.binder_interface_centroid)
        record.approach_vector = normalize_vector(
            record.aligned_binder_interface_centroid - record.aligned_target_interface_centroid
        )
    return reference


def assign_secondary_bins(
    records: List[ComplexRecord],
    angle_threshold_deg: float,
    centroid_threshold: float,
) -> None:
    grouped: Dict[str, List[ComplexRecord]] = defaultdict(list)
    for record in records:
        grouped[record.primary_bin_id].append(record)

    for primary_bin_id, group in grouped.items():
        group.sort(key=lambda record: (-record.best_ipsae, record.binder_name, record.pair_id))
        vectors = np.vstack([record.approach_vector for record in group])
        centroids = np.vstack([record.aligned_target_interface_centroid for record in group])
        geometry_distance = secondary_geometry_distance_matrix(vectors, centroids, angle_threshold_deg, centroid_threshold)
        sub_labels = complete_linkage_threshold_clusters(geometry_distance, 1.0)
        sub_sizes = Counter(sub_labels)
        ordered_subs = [label for label, _ in sorted(sub_sizes.items(), key=lambda item: (-item[1], item[0]))]
        sub_to_id = {label: f"S{idx + 1:02d}" for idx, label in enumerate(ordered_subs)}
        for record, sub_label in zip(group, sub_labels):
            record.sub_bin_index = int(sub_label)
            record.sub_bin_id = sub_to_id[int(sub_label)]
            record.final_bin_id = f"{primary_bin_id}_{record.sub_bin_id}"


def build_bin_members(records: Sequence[ComplexRecord]) -> Dict[str, List[ComplexRecord]]:
    grouped: Dict[str, List[ComplexRecord]] = defaultdict(list)
    for idx, record in enumerate(records):
        setattr(record, "_member_index", idx)
        grouped[record.final_bin_id].append(record)
    for members in grouped.values():
        members.sort(key=lambda record: (-record.best_ipsae, record.binder_name, record.pair_id))
    return grouped


def build_per_complex_geometry(records: Sequence[ComplexRecord], ipsae_threshold: float) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "pair_id": record.pair_id,
                "binder_name": record.binder_name,
                "binder_seq": record.binder_seq,
                "target_name": record.target_name,
                "best_ipsae": record.best_ipsae,
                "ipsae_mean": record.ipsae_mean,
                "best_iptm": record.best_iptm,
                "best_pdockq2": record.best_pdockq2,
                "ipsae_threshold_used": float(ipsae_threshold),
                "passes_ipsae_threshold": bool(record.passes_ipsae_threshold),
                "contact_count": record.n_contacted_residues,
                "contacted_target_residues": ";".join(record.contacted_residue_labels),
                "target_interface_centroid_x": float(record.aligned_target_interface_centroid[0]),
                "target_interface_centroid_y": float(record.aligned_target_interface_centroid[1]),
                "target_interface_centroid_z": float(record.aligned_target_interface_centroid[2]),
                "binder_interface_centroid_x": float(record.aligned_binder_interface_centroid[0]),
                "binder_interface_centroid_y": float(record.aligned_binder_interface_centroid[1]),
                "binder_interface_centroid_z": float(record.aligned_binder_interface_centroid[2]),
                "approach_vector_x": float(record.approach_vector[0]),
                "approach_vector_y": float(record.approach_vector[1]),
                "approach_vector_z": float(record.approach_vector[2]),
                "target_alignment_rmsd_before": record.target_alignment_rmsd_before,
                "target_alignment_rmsd_after": record.target_alignment_rmsd_after,
                "primary_bin_id": record.primary_bin_id,
                "sub_bin_id": record.sub_bin_id,
                "final_bin_id": record.final_bin_id,
                "is_representative": bool(record.is_representative),
            }
        )
    return pd.DataFrame(rows).sort_values(["final_bin_id", "best_ipsae"], ascending=[True, False]).reset_index(drop=True)


def build_epitope_bins(records: Sequence[ComplexRecord], ipsae_threshold: float) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "pair_id": record.pair_id,
                "binder_name": record.binder_name,
                "binder_seq": record.binder_seq,
                "target_name": record.target_name,
                "best_ipsae": record.best_ipsae,
                "best_iptm": record.best_iptm,
                "best_pdockq2": record.best_pdockq2,
                "ipsae_threshold_used": float(ipsae_threshold),
                "passes_ipsae_threshold": bool(record.passes_ipsae_threshold),
                "primary_bin_id": record.primary_bin_id,
                "sub_bin_id": record.sub_bin_id,
                "final_bin_id": record.final_bin_id,
                "contact_count": record.n_contacted_residues,
                "is_representative": bool(record.is_representative),
            }
        )
    return pd.DataFrame(rows).sort_values(["final_bin_id", "best_ipsae"], ascending=[True, False]).reset_index(drop=True)


def build_bin_outputs(
    bin_members: Dict[str, List[ComplexRecord]],
    target_residue_keys: Sequence[ResidueKey],
    representative_mode: str,
    min_bin_size: int,
    angle_threshold_deg: float,
    centroid_threshold: float,
    ipsae_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    summary_rows: List[dict] = []
    consensus_json: Dict[str, dict] = {}
    for final_bin_id, members in sorted(bin_members.items(), key=lambda item: (-len(item[1]), item[0])):
        local_distance_matrix = build_final_bin_distance_matrix(members, angle_threshold_deg, centroid_threshold)
        representative = choose_representative(members, representative_mode, local_distance_matrix=local_distance_matrix)
        representative.is_representative = True
        contact_matrix = np.vstack([member.target_contact_vector.astype(int) for member in members])
        contact_frequency = contact_matrix.mean(axis=0)
        intersection_mask = contact_frequency == 1.0
        median_mask = contact_frequency >= 0.5
        union_mask = contact_frequency > 0.0
        intersection_labels = [target_residue_keys[idx].label for idx in np.flatnonzero(intersection_mask)]
        median_labels = [target_residue_keys[idx].label for idx in np.flatnonzero(median_mask)]
        union_labels = [target_residue_keys[idx].label for idx in np.flatnonzero(union_mask)]
        angle_stats = summarize_numeric(pairwise_values([member.approach_vector for member in members], angle_between_vectors))
        centroid_stats = summarize_numeric(
            pairwise_values(
                [member.aligned_target_interface_centroid for member in members],
                lambda left, right: np.linalg.norm(left - right),
            )
        )
        rmsd_stats = summarize_numeric([member.target_alignment_rmsd_after for member in members])
        best_ipsae_values = np.asarray([member.best_ipsae for member in members], dtype=float)
        best_iptm_values = np.asarray([member.best_iptm for member in members], dtype=float)
        best_pdockq2_values = np.asarray([member.best_pdockq2 for member in members], dtype=float)
        ipsae_pass_count = int(sum(member.passes_ipsae_threshold for member in members))

        def metric_stats(values: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }

        ipsae_stats = metric_stats(best_ipsae_values)
        iptm_stats = metric_stats(best_iptm_values)
        pdockq2_stats = metric_stats(best_pdockq2_values)

        if len(members) >= max(1, min_bin_size):
            summary_rows.append(
                {
                    "final_bin_id": final_bin_id,
                    "primary_bin_id": members[0].primary_bin_id,
                    "sub_bin_id": members[0].sub_bin_id,
                    "bin_size": int(len(members)),
                    "bin_size_all": int(len(members)),
                    "bin_size_ipsae_pass": ipsae_pass_count,
                    "fraction_ipsae_pass": float(ipsae_pass_count / len(members)),
                    "representative_binder_name": representative.binder_name,
                    "representative_pair_id": representative.pair_id,
                    "representative_best_ipsae": representative.best_ipsae,
                    "representative_passes_ipsae_threshold": bool(representative.passes_ipsae_threshold),
                    "ipsae_threshold_used": float(ipsae_threshold),
                    "representative_mode": representative_mode,
                    "mean_best_ipsae": ipsae_stats["mean"],
                    "median_best_ipsae": ipsae_stats["median"],
                    "min_best_ipsae": ipsae_stats["min"],
                    "max_best_ipsae": ipsae_stats["max"],
                    "std_best_ipsae": ipsae_stats["std"],
                    "mean_best_iptm": iptm_stats["mean"],
                    "median_best_iptm": iptm_stats["median"],
                    "min_best_iptm": iptm_stats["min"],
                    "max_best_iptm": iptm_stats["max"],
                    "std_best_iptm": iptm_stats["std"],
                    "mean_best_pdockq2": pdockq2_stats["mean"],
                    "median_best_pdockq2": pdockq2_stats["median"],
                    "min_best_pdockq2": pdockq2_stats["min"],
                    "max_best_pdockq2": pdockq2_stats["max"],
                    "std_best_pdockq2": pdockq2_stats["std"],
                    "intersection_epitope": ";".join(intersection_labels),
                    "median_epitope": ";".join(median_labels),
                    "union_epitope": ";".join(union_labels),
                    "intersection_epitope_size": int(len(intersection_labels)),
                    "median_epitope_size": int(len(median_labels)),
                    "union_epitope_size": int(len(union_labels)),
                    "mean_target_alignment_rmsd_after": rmsd_stats["mean"],
                    "median_target_alignment_rmsd_after": rmsd_stats["median"],
                    "max_target_alignment_rmsd_after": rmsd_stats["max"],
                    "mean_within_bin_angle_deg": angle_stats["mean"],
                    "median_within_bin_angle_deg": angle_stats["median"],
                    "max_within_bin_angle_deg": angle_stats["max"],
                    "mean_within_bin_target_interface_centroid_dist": centroid_stats["mean"],
                    "median_within_bin_target_interface_centroid_dist": centroid_stats["median"],
                    "max_within_bin_target_interface_centroid_dist": centroid_stats["max"],
                }
            )
            consensus_json[final_bin_id] = {
                "primary_bin_id": members[0].primary_bin_id,
                "sub_bin_id": members[0].sub_bin_id,
                "bin_size": int(len(members)),
                "representative_binder_name": representative.binder_name,
                "representative_pair_id": representative.pair_id,
                "intersection_epitope": intersection_labels,
                "median_epitope": median_labels,
                "union_epitope": union_labels,
            }
    return pd.DataFrame(summary_rows), consensus_json


def build_offtarget_annotation(
    bin_members: Dict[str, List[ComplexRecord]],
    context: Optional[OffTargetContext],
) -> Optional[pd.DataFrame]:
    if context is None or context.frame.empty:
        return None

    membership_rows = []
    for final_bin_id, members in bin_members.items():
        for member in members:
            row = {
                "final_bin_id": final_bin_id,
                "primary_bin_id": member.primary_bin_id,
                "sub_bin_id": member.sub_bin_id,
                "binder_name": member.binder_name,
            }
            if "binder_seq" in context.key_cols:
                row["binder_seq"] = member.binder_seq
            membership_rows.append(row)
    membership_df = pd.DataFrame(membership_rows)
    merged = membership_df.merge(context.frame, on=context.key_cols, how="inner")
    if merged.empty:
        return None

    rows = []
    for (final_bin_id, target_name_off), group in merged.groupby(["final_bin_id", "target_name_off"], dropna=False):
        best_off = group["best_ipsae_off"].to_numpy(dtype=float)
        delta = group["best_ipsae_delta_on_minus_off"].to_numpy(dtype=float)
        top_rows = group.sort_values(["best_ipsae_off", "binder_name"], ascending=[False, True]).head(5)
        rows.append(
            {
                "final_bin_id": final_bin_id,
                "primary_bin_id": str(group["primary_bin_id"].iloc[0]),
                "sub_bin_id": str(group["sub_bin_id"].iloc[0]),
                "target_name_off": str(target_name_off),
                "offtarget_label": context.offtarget_label_override or str(target_name_off),
                "n_members_with_offtarget_data": int(len(group)),
                "mean_best_ipsae_off": float(np.mean(best_off)),
                "median_best_ipsae_off": float(np.median(best_off)),
                "mean_best_ipsae_delta_on_minus_off": float(np.mean(delta)),
                "median_best_ipsae_delta_on_minus_off": float(np.median(delta)),
                "fraction_best_ipsae_off_gt_0_2": float(np.mean(best_off > 0.2)),
                "fraction_best_ipsae_off_gt_0_5": float(np.mean(best_off > 0.5)),
                "top_offtarget_binders": ";".join(top_rows["binder_name"].astype(str)),
            }
        )
    return pd.DataFrame(rows).sort_values(["final_bin_id", "target_name_off"]).reset_index(drop=True)


def plot_target_residue_occupancy(occupancy_df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(occupancy_df))
    y = occupancy_df["contact_frequency"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, y, color="#2f7ed8", width=0.9)
    ax.set_xlabel("Target residue index")
    ax.set_ylabel("Contact frequency")
    ax.set_title("Target Residue Occupancy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.2)
    if len(x) <= 50:
        ax.set_xticks(x, occupancy_df["residue_label"].tolist(), rotation=90, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_contact_heatmap(records: Sequence[ComplexRecord], out_path: Path) -> None:
    if not records:
        return
    ordered = sorted(records, key=lambda record: (record.final_bin_id, -record.best_ipsae, record.binder_name))
    matrix = np.vstack([record.target_contact_vector.astype(int) for record in ordered])
    fig, ax = plt.subplots(figsize=(12, max(5.0, len(ordered) * 0.02)))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Target residue index")
    ax.set_ylabel("Complexes")
    ax.set_title("Epitope Contact Heatmap")
    current = None
    for row_idx, record in enumerate(ordered):
        if current is None:
            current = record.final_bin_id
        elif record.final_bin_id != current:
            ax.axhline(row_idx - 0.5, color="white", linewidth=1.2)
            current = record.final_bin_id
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_approach_geometry(records: Sequence[ComplexRecord], out_path: Path) -> None:
    if len(records) == 0:
        return
    centroids = np.vstack([record.aligned_target_interface_centroid for record in records])
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    if len(records) == 1:
        projection = np.array([[0.0, 0.0]])
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T
        projection = centered @ basis
    bins = [record.final_bin_id for record in records]
    unique_bins = sorted(dict.fromkeys(bins))
    color_map = {bin_id: plt.cm.tab20(idx % 20) for idx, bin_id in enumerate(unique_bins)}
    fig, ax = plt.subplots(figsize=(8, 6))
    for point, record in zip(projection, records):
        ax.scatter(point[0], point[1], s=28, color=color_map[record.final_bin_id], edgecolor="none")
    ax.set_title("Aligned Target Interface Centroid Scatter")
    ax.set_xlabel("Aligned target-interface centroid PC1")
    ax.set_ylabel("Aligned target-interface centroid PC2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_aligned_representative_cifs(bin_members: Dict[str, List[ComplexRecord]], outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    for final_bin_id, members in bin_members.items():
        representatives = [member for member in members if member.is_representative]
        for representative in representatives:
            structure = gemmi.read_structure(str(representative.cif_path))
            if len(structure) == 0 or representative.transform is None:
                continue
            clone = structure.clone()
            clone[0].transform_pos_and_adp(representative.transform)
            out_name = f"{final_bin_id}__{pair_id_to_grouped_cif_name(representative.pair_id)}"
            clone.make_mmcif_document().write_file(str(outdir / out_name))
            written += 1
    return written


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    cif_dir = Path(args.cif_dir)
    pair_summary_path = Path(args.pair_summary)
    outdir = Path(args.outdir)
    comparison_path = Path(args.comparison_csv) if args.comparison_csv else None

    if not cif_dir.is_dir():
        raise NotADirectoryError(f"--cif-dir is not a directory: {cif_dir}")
    outdir.mkdir(parents=True, exist_ok=True)

    pair_summary = load_pair_summary(pair_summary_path)
    if "partner_role" in pair_summary.columns:
        pair_summary = pair_summary.loc[pair_summary["partner_role"].astype(str) == "target"].copy()
    successful = pair_summary.loc[pair_summary["status"].astype(str) == "success"].copy()
    if successful.empty:
        raise ValueError("No successful complexes found in pair_summary.csv")
    successful["passes_ipsae_threshold"] = successful["best_ipsae"].astype(float) > float(args.ipsae_threshold)
    if args.include_all_successful:
        retained = successful.copy()
    else:
        retained = successful.loc[successful["passes_ipsae_threshold"]].copy()
    if retained.empty:
        if args.include_all_successful:
            raise ValueError("No successful complexes available for analysis")
        raise ValueError(f"No successful complexes passed best_ipsae > {args.ipsae_threshold:.3f}")

    retained_target_names = sorted(retained["target_name"].astype(str).unique())
    if len(retained_target_names) != 1:
        raise ValueError(f"Expected a single retained target_name, found: {retained_target_names}")
    retained_target_name = retained_target_names[0]
    if args.target_name and str(args.target_name) != retained_target_name:
        raise ValueError(
            f"--target-name {args.target_name!r} does not match retained target_name {retained_target_name!r}"
        )

    records: List[ComplexRecord] = []
    reference_target_keys: Optional[Tuple[ResidueKey, ...]] = None
    for row in retained.sort_values(["best_ipsae", "binder_name"], ascending=[False, True]).itertuples(index=False):
        row_series = pd.Series(row._asdict())
        record = parse_complex_record(
            row=row_series,
            cif_dir=cif_dir,
            binder_chain_id=args.binder_chain,
            target_chain_id=args.target_chain,
            contact_cutoff=args.contact_cutoff,
            reference_target_keys=reference_target_keys,
        )
        if reference_target_keys is None:
            reference_target_keys = record.target_residue_keys
        records.append(record)

    assert reference_target_keys is not None
    distance_matrix = assign_primary_bins(records, args.jaccard_distance_threshold)
    reference_record = align_records(records)
    assign_secondary_bins(records, args.approach_angle_threshold_deg, args.interface_centroid_threshold)
    bin_members = build_bin_members(records)

    bin_summary_df, consensus_json = build_bin_outputs(
        bin_members=bin_members,
        target_residue_keys=reference_target_keys,
        representative_mode=args.representative_mode,
        min_bin_size=max(1, int(args.min_bin_size)),
        angle_threshold_deg=args.approach_angle_threshold_deg,
        centroid_threshold=args.interface_centroid_threshold,
        ipsae_threshold=args.ipsae_threshold,
    )

    base_core_df = make_core_dataframe(records, args.ipsae_threshold)
    offtarget_context = maybe_load_offtarget_context(comparison_path, base_core_df, retained_target_name, args.offtarget_label)
    filtered_binders_df = apply_direct_offtarget_merge(base_core_df, offtarget_context)

    per_complex_geometry_df = build_per_complex_geometry(records, args.ipsae_threshold)
    if offtarget_context is not None and offtarget_context.direct_attach:
        attach_cols = offtarget_context.key_cols + [
            "target_name_off",
            "best_ipsae_on",
            "best_ipsae_off",
            "best_ipsae_delta_on_minus_off",
            "ipsae_mean_on",
            "ipsae_mean_off",
        ]
        if "offtarget_label" in filtered_binders_df.columns:
            attach_cols.append("offtarget_label")
        attach_df = filtered_binders_df.loc[:, attach_cols].drop_duplicates()
        per_complex_geometry_df = per_complex_geometry_df.merge(
            attach_df,
            on=offtarget_context.key_cols,
            how="left",
            validate="one_to_one",
        )

    epitope_bins_df = build_epitope_bins(records, args.ipsae_threshold)
    if offtarget_context is not None and offtarget_context.direct_attach:
        attach_cols = offtarget_context.key_cols + [
            "target_name_off",
            "best_ipsae_on",
            "best_ipsae_off",
            "best_ipsae_delta_on_minus_off",
            "ipsae_mean_on",
            "ipsae_mean_off",
        ]
        if "offtarget_label" in filtered_binders_df.columns:
            attach_cols.append("offtarget_label")
        attach_df = filtered_binders_df.loc[:, attach_cols].drop_duplicates()
        epitope_bins_df = epitope_bins_df.merge(attach_df, on=offtarget_context.key_cols, how="left", validate="one_to_one")

    target_occupancy_df = write_target_residue_occupancy(records, reference_target_keys, outdir / "target_residue_occupancy.csv")
    offtarget_annotation_df = build_offtarget_annotation(bin_members, offtarget_context)

    filtered_binders_df.to_csv(outdir / "analyzed_complexes.csv", index=False)
    filtered_binders_df.to_csv(outdir / "filtered_binders.csv", index=False)
    per_complex_geometry_df.to_csv(outdir / "per_complex_geometry.csv", index=False)
    epitope_bins_df.to_csv(outdir / "epitope_bins.csv", index=False)
    bin_summary_df.to_csv(outdir / "bin_summary.csv", index=False)
    if offtarget_annotation_df is not None:
        offtarget_annotation_df.to_csv(outdir / "bin_offtarget_annotation.csv", index=False)

    write_json(outdir / "bin_consensus_epitopes.json", consensus_json)

    plot_target_residue_occupancy(target_occupancy_df, outdir / "target_residue_occupancy.png")
    plot_contact_heatmap(records, outdir / "epitope_bin_heatmap.png")
    plot_approach_geometry(records, outdir / "approach_geometry_scatter.png")

    aligned_cif_count = 0
    if args.write_aligned_cifs:
        aligned_cif_count = write_aligned_representative_cifs(bin_members, outdir / "aligned_representatives")

    metadata = {
        "inputs": {
            "cif_dir": str(cif_dir.resolve()),
            "pair_summary": str(pair_summary_path.resolve()),
            "comparison_csv": str(comparison_path.resolve()) if comparison_path else None,
        },
        "args": vars(args),
        "counts": {
            "pair_summary_rows": int(len(pair_summary)),
            "successful_complexes": int(len(successful)),
            "retained_complexes": int(len(records)),
            "ipsae_pass_complexes": int(successful["passes_ipsae_threshold"].sum()),
            "target_residues": int(len(reference_target_keys)),
            "primary_bins": int(len({record.primary_bin_id for record in records})),
            "final_bins": int(len(bin_members)),
        },
        "analysis_mode": {
            "include_all_successful": bool(args.include_all_successful),
            "filtered_only": not bool(args.include_all_successful),
        },
        "thresholds": {
            "ipsae_threshold": float(args.ipsae_threshold),
            "contact_cutoff": float(args.contact_cutoff),
            "jaccard_distance_threshold": float(args.jaccard_distance_threshold),
            "approach_angle_threshold_deg": float(args.approach_angle_threshold_deg),
            "interface_centroid_threshold": float(args.interface_centroid_threshold),
        },
        "reference_model": {
            "pair_id": reference_record.pair_id,
            "binder_name": reference_record.binder_name,
            "best_ipsae": reference_record.best_ipsae,
            "cif_path": str(reference_record.cif_path),
        },
        "offtarget_annotation": {
            "present": bool(offtarget_context is not None),
            "direct_attach": bool(offtarget_context.direct_attach) if offtarget_context else False,
            "key_cols": offtarget_context.key_cols if offtarget_context else [],
            "rows": int(len(offtarget_context.frame)) if offtarget_context else 0,
            "unique_target_name_off": sorted(offtarget_context.frame["target_name_off"].astype(str).unique().tolist())
            if offtarget_context is not None and not offtarget_context.frame.empty
            else [],
        },
        "outputs": {
            "wrote_analyzed_complexes_csv": True,
            "wrote_filtered_binders_csv_compat": True,
            "wrote_bin_offtarget_annotation": bool(offtarget_annotation_df is not None and not offtarget_annotation_df.empty),
            "wrote_target_residue_occupancy_png": True,
            "wrote_epitope_bin_heatmap_png": True,
            "wrote_epitope_bin_dendrogram_png": False,
            "wrote_approach_geometry_scatter_png": True,
            "aligned_representative_cifs_written": int(aligned_cif_count),
        },
    }
    write_json(outdir / "analysis_metadata.json", metadata)

    print(
        f"[epitope-analysis] retained={len(records)} target_residues={len(reference_target_keys)} "
        f"primary_bins={metadata['counts']['primary_bins']} final_bins={metadata['counts']['final_bins']}"
    )
    print(f"[epitope-analysis] outputs written to {outdir}")


if __name__ == "__main__":
    main()
