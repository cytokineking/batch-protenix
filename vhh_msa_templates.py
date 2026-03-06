#!/usr/bin/env python3
"""
Helpers for VHH binder template analysis and query-swap MSA derivation.

This module intentionally keeps antibody-numbering imports lazy so local orchestration
can call remote helpers without requiring the numbering backend to be installed in the
local Python environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def normalize_sequence(seq: str) -> str:
    return re.sub(r"\s+", "", seq or "").upper()


def sequence_is_protein_like(seq: str) -> bool:
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYX]+", normalize_sequence(seq)))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 10) -> str:
    return sha256_hex(text)[:n]


def normalize_framework_mode(framework_mode: str = "lengths_only") -> str:
    mode_n = str(framework_mode or "lengths_only").strip().lower()
    if mode_n == "exact_frameworks":
        mode_n = "exact"
    if mode_n not in {"exact", "lengths_only"}:
        raise ValueError("framework_mode must be one of: exact,lengths_only")
    return mode_n


def framework_hash(fr1: str, fr2: str, fr3: str, fr4: str) -> str:
    return sha256_hex("|".join([fr1, fr2, fr3, fr4]))


def _parse_fasta_string(content: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    header = ""
    for line in (content or "").replace("\x00", "").strip().splitlines():
        line_n = line.strip()
        if not line_n:
            continue
        if line_n.startswith(">"):
            header = line_n[1:].strip()
            parsed[header] = ""
        else:
            parsed[header] = parsed.get(header, "") + line_n
    return parsed


def rewrite_query_in_non_pairing_a3m(non_pairing: str, query_seq: str) -> str:
    qseq = normalize_sequence(query_seq)
    parsed = _parse_fasta_string(non_pairing)
    records: List[Tuple[str, str]] = []
    for hdr, seq in parsed.items():
        h = (hdr or "").strip()
        s = (seq or "").strip()
        if not h or not s:
            continue
        if h.lower().startswith("query"):
            continue
        records.append((h, s))
    records.sort(key=lambda x: x[0])
    lines: List[str] = [">query", qseq]
    for hdr, seq in records:
        lines.append(f">{hdr}")
        lines.append(seq.rstrip())
    return "\n".join(lines).strip() + "\n"


def pairing_from_strategy(non_pairing: str, query_seq: str, pairing_strategy: str) -> str:
    strategy = (pairing_strategy or "greedy").strip().lower()
    if strategy == "copy_non_pairing":
        return non_pairing
    return f">query\n{normalize_sequence(query_seq)}\n"


class VhhNumberingError(RuntimeError):
    pass


@dataclass(frozen=True)
class VhhSegmentation:
    sequence: str
    numbering_scheme: str
    chain_class: str
    chain_type: str
    fr1: str
    cdr1: str
    fr2: str
    cdr2: str
    fr3: str
    cdr3: str
    fr4: str
    cdr1_register: str
    cdr2_register: str
    cdr3_register: str
    fr1_length: int
    fr2_length: int
    fr3_length: int
    fr4_length: int
    cdr1_length: int
    cdr2_length: int
    cdr3_length: int
    total_binder_length: int
    framework_hash: str


def _position_label(pos: Any) -> str:
    if hasattr(pos, "format"):
        try:
            return str(pos.format(chain_type=True, region=False))
        except TypeError:
            try:
                return str(pos.format())
            except Exception:  # noqa: BLE001
                pass
    return str(pos)


def _region_sequence_and_register(region_map: Dict[Any, str]) -> Tuple[str, str]:
    ordered = sorted(region_map.items(), key=lambda item: item[0])
    seq = "".join(str(aa) for _, aa in ordered)
    register = ",".join(_position_label(pos) for pos, _ in ordered)
    return seq, register


def _extract_region_map(chain: Any, region_name: str) -> Dict[Any, str]:
    regions = getattr(chain, "regions", None)
    if not isinstance(regions, dict):
        raise VhhNumberingError("Numbering backend did not expose chain.regions")
    region = regions.get(region_name)
    if not isinstance(region, dict) or not region:
        raise VhhNumberingError(f"Missing or empty region '{region_name}' in numbering output")
    return region


def number_vhh_sequence(sequence: str, numbering_scheme: str = "imgt") -> VhhSegmentation:
    seq = normalize_sequence(sequence)
    if not seq:
        raise VhhNumberingError("Empty sequence")
    if (numbering_scheme or "imgt").strip().lower() != "imgt":
        raise VhhNumberingError("Only IMGT numbering is supported in MVP")
    try:
        from abnumber import Chain  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise VhhNumberingError(
            "AbNumber is required for VHH numbering. Ensure the runtime installs abnumber, anarcii, and hmmer."
        ) from exc

    try:
        # AbNumber defaults to the legacy ANARCI backend, but this repo installs the
        # modern ANARCII stack. Opt in explicitly so local and Modal runtimes behave the same way.
        chain = Chain(
            seq,
            scheme="imgt",
            cdr_definition="imgt",
            allowed_species=None,
            use_anarcii=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise VhhNumberingError(f"IMGT numbering failed: {exc}") from exc

    chain_type = str(getattr(chain, "chain_type", "") or "").strip().upper()
    if chain_type != "H":
        raise VhhNumberingError(f"Sequence numbered as non-heavy chain: chain_type={chain_type or 'unknown'}")

    variable_seq = normalize_sequence(str(getattr(chain, "seq", "") or ""))
    tail_seq = normalize_sequence(str(getattr(chain, "tail", "") or ""))
    if tail_seq:
        raise VhhNumberingError("Sequence includes a non-empty constant/tail region")
    if variable_seq != seq:
        raise VhhNumberingError("Numbered variable sequence does not exactly match input sequence")

    fr1, _ = _region_sequence_and_register(_extract_region_map(chain, "FR1"))
    cdr1, cdr1_register = _region_sequence_and_register(_extract_region_map(chain, "CDR1"))
    fr2, _ = _region_sequence_and_register(_extract_region_map(chain, "FR2"))
    cdr2, cdr2_register = _region_sequence_and_register(_extract_region_map(chain, "CDR2"))
    fr3, _ = _region_sequence_and_register(_extract_region_map(chain, "FR3"))
    cdr3, cdr3_register = _region_sequence_and_register(_extract_region_map(chain, "CDR3"))
    fr4, _ = _region_sequence_and_register(_extract_region_map(chain, "FR4"))

    rebuilt = "".join([fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4])
    if rebuilt != seq:
        raise VhhNumberingError("FR/CDR segmentation did not reconstruct the input sequence exactly")

    return VhhSegmentation(
        sequence=seq,
        numbering_scheme="imgt",
        chain_class="vhh",
        chain_type=chain_type,
        fr1=fr1,
        cdr1=cdr1,
        fr2=fr2,
        cdr2=cdr2,
        fr3=fr3,
        cdr3=cdr3,
        fr4=fr4,
        cdr1_register=cdr1_register,
        cdr2_register=cdr2_register,
        cdr3_register=cdr3_register,
        fr1_length=len(fr1),
        fr2_length=len(fr2),
        fr3_length=len(fr3),
        fr4_length=len(fr4),
        cdr1_length=len(cdr1),
        cdr2_length=len(cdr2),
        cdr3_length=len(cdr3),
        total_binder_length=len(seq),
        framework_hash=framework_hash(fr1, fr2, fr3, fr4),
    )


def build_canonical_template_key(
    segmentation: VhhSegmentation,
    framework_mode: str = "exact",
) -> Dict[str, Any]:
    framework_mode_n = normalize_framework_mode(framework_mode)
    key = {
        "numbering_scheme": segmentation.numbering_scheme,
        "chain_class": segmentation.chain_class,
        "cdr1_register": segmentation.cdr1_register,
        "cdr2_register": segmentation.cdr2_register,
        "cdr3_register": segmentation.cdr3_register,
    }
    if framework_mode_n == "lengths_only":
        key.update(
            {
                "fr1_length": segmentation.fr1_length,
                "fr2_length": segmentation.fr2_length,
                "fr3_length": segmentation.fr3_length,
                "fr4_length": segmentation.fr4_length,
            }
        )
    else:
        key.update(
            {
                "fr1": segmentation.fr1,
                "fr2": segmentation.fr2,
                "fr3": segmentation.fr3,
                "fr4": segmentation.fr4,
            }
        )
    return key


def canonical_template_key_json(segmentation: VhhSegmentation, framework_mode: str = "exact") -> str:
    return json.dumps(
        build_canonical_template_key(segmentation, framework_mode=framework_mode),
        sort_keys=True,
        separators=(",", ":"),
    )


def analyze_unique_binders(
    binder_records: Sequence[Dict[str, Any]],
    numbering_scheme: str = "imgt",
) -> Dict[str, List[Dict[str, Any]]]:
    analyzed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for record in binder_records:
        binder_name = str(record.get("binder_name_first") or record.get("binder_name") or "unnamed")
        binder_sequence = normalize_sequence(str(record.get("binder_sequence") or record.get("binder_seq") or ""))
        row_index = int(record.get("first_row_index", record.get("row_index", -1)))

        if not binder_sequence:
            rejected.append(
                {
                    "binder_name": binder_name,
                    "binder_sequence": binder_sequence,
                    "reason": "empty binder sequence",
                    "stage": "normalize",
                    "first_row_index": row_index,
                }
            )
            continue
        if not sequence_is_protein_like(binder_sequence):
            rejected.append(
                {
                    "binder_name": binder_name,
                    "binder_sequence": binder_sequence,
                    "reason": "binder sequence is not protein-like",
                    "stage": "validate",
                    "first_row_index": row_index,
                }
            )
            continue

        try:
            seg = number_vhh_sequence(binder_sequence, numbering_scheme=numbering_scheme)
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {
                    "binder_name": binder_name,
                    "binder_sequence": binder_sequence,
                    "reason": str(exc),
                    "stage": "numbering",
                    "first_row_index": row_index,
                }
            )
            continue

        template_key_json = canonical_template_key_json(seg, framework_mode="exact")
        lengths_only_key_json = canonical_template_key_json(seg, framework_mode="lengths_only")
        analyzed.append(
            {
                "binder_name_first": binder_name,
                "binder_sequence": binder_sequence,
                "binder_sequence_sha256": sha256_hex(binder_sequence),
                "first_row_index": row_index,
                "numbering_scheme": seg.numbering_scheme,
                "chain_class": seg.chain_class,
                "chain_type": seg.chain_type,
                "fr1": seg.fr1,
                "cdr1": seg.cdr1,
                "fr2": seg.fr2,
                "cdr2": seg.cdr2,
                "fr3": seg.fr3,
                "cdr3": seg.cdr3,
                "fr4": seg.fr4,
                "fr1_length": seg.fr1_length,
                "fr2_length": seg.fr2_length,
                "fr3_length": seg.fr3_length,
                "fr4_length": seg.fr4_length,
                "cdr1_length": seg.cdr1_length,
                "cdr2_length": seg.cdr2_length,
                "cdr3_length": seg.cdr3_length,
                "cdr1_register": seg.cdr1_register,
                "cdr2_register": seg.cdr2_register,
                "cdr3_register": seg.cdr3_register,
                "total_binder_length": seg.total_binder_length,
                "framework_hash": seg.framework_hash,
                "canonical_template_key_json": template_key_json,
                "canonical_template_key_hash": short_hash(template_key_json, n=16),
                "lengths_only_template_key_json": lengths_only_key_json,
                "lengths_only_template_key_hash": short_hash(lengths_only_key_json, n=16),
            }
        )

    analyzed.sort(key=lambda row: (int(row["first_row_index"]), str(row["binder_sequence"])))
    rejected.sort(key=lambda row: (int(row.get("first_row_index", -1)), str(row.get("binder_sequence", ""))))
    return {"analyzed": analyzed, "rejected": rejected}


def analyze_vhh_sequence(sequence: str, numbering_scheme: str = "imgt") -> Dict[str, Any]:
    seg = number_vhh_sequence(sequence, numbering_scheme=numbering_scheme)
    template_key_json = canonical_template_key_json(seg, framework_mode="exact")
    lengths_only_key_json = canonical_template_key_json(seg, framework_mode="lengths_only")
    return {
        "binder_sequence": seg.sequence,
        "binder_sequence_sha256": sha256_hex(seg.sequence),
        "numbering_scheme": seg.numbering_scheme,
        "chain_class": seg.chain_class,
        "chain_type": seg.chain_type,
        "fr1": seg.fr1,
        "cdr1": seg.cdr1,
        "fr2": seg.fr2,
        "cdr2": seg.cdr2,
        "fr3": seg.fr3,
        "cdr3": seg.cdr3,
        "fr4": seg.fr4,
        "fr1_length": seg.fr1_length,
        "fr2_length": seg.fr2_length,
        "fr3_length": seg.fr3_length,
        "fr4_length": seg.fr4_length,
        "cdr1_length": seg.cdr1_length,
        "cdr2_length": seg.cdr2_length,
        "cdr3_length": seg.cdr3_length,
        "cdr1_register": seg.cdr1_register,
        "cdr2_register": seg.cdr2_register,
        "cdr3_register": seg.cdr3_register,
        "total_binder_length": seg.total_binder_length,
        "framework_hash": seg.framework_hash,
        "canonical_template_key_json": template_key_json,
        "canonical_template_key_hash": short_hash(template_key_json, n=16),
        "lengths_only_template_key_json": lengths_only_key_json,
        "lengths_only_template_key_hash": short_hash(lengths_only_key_json, n=16),
    }


def extract_unique_binders(pair_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in pair_rows:
        binder_sequence = normalize_sequence(str(row.get("binder_seq") or row.get("binder_sequence") or ""))
        if not binder_sequence or binder_sequence in seen:
            continue
        seen.add(binder_sequence)
        out.append(
            {
                "binder_name_first": str(row.get("binder_name") or row.get("binder_name_first") or "unnamed"),
                "binder_sequence": binder_sequence,
                "first_row_index": int(row.get("row_index", row.get("first_row_index", -1))),
            }
        )
    return out


def merge_binder_analyses(
    unique_binders: Sequence[Dict[str, Any]],
    remote_results: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    analyzed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for binder in unique_binders:
        binder_sequence = normalize_sequence(str(binder.get("binder_sequence") or ""))
        binder_name = str(binder.get("binder_name_first") or "unnamed")
        row_index = int(binder.get("first_row_index", -1))
        result = dict(remote_results.get(binder_sequence) or {})
        if not result.get("ok"):
            rejected.append(
                {
                    "binder_name": binder_name,
                    "binder_sequence": binder_sequence,
                    "reason": str(result.get("error") or "analysis failed"),
                    "stage": str(result.get("stage") or "numbering"),
                    "first_row_index": row_index,
                }
            )
            continue
        analysis = dict(result.get("analysis") or {})
        analysis["binder_name_first"] = binder_name
        analysis["binder_sequence"] = binder_sequence
        analysis["binder_sequence_sha256"] = analysis.get("binder_sequence_sha256") or sha256_hex(binder_sequence)
        analysis["first_row_index"] = row_index
        analyzed.append(analysis)
    analyzed.sort(key=lambda row: (int(row["first_row_index"]), str(row["binder_sequence"])))
    rejected.sort(key=lambda row: (int(row.get("first_row_index", -1)), str(row.get("binder_sequence", ""))))
    return {"analyzed": analyzed, "rejected": rejected}


def group_template_members(
    analyzed_records: Sequence[Dict[str, Any]],
    representative_policy: str = "first",
    framework_mode: str = "lengths_only",
) -> List[Dict[str, Any]]:
    policy = (representative_policy or "first").strip().lower()
    if policy != "first":
        raise ValueError(f"Unsupported representative policy '{representative_policy}'")

    framework_mode_n = normalize_framework_mode(framework_mode)
    key_json_field = "lengths_only_template_key_json" if framework_mode_n == "lengths_only" else "canonical_template_key_json"
    key_hash_field = "lengths_only_template_key_hash" if framework_mode_n == "lengths_only" else "canonical_template_key_hash"
    grouping_mode = framework_mode_n
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in analyzed_records:
        grouped.setdefault(str(row[key_json_field]), []).append(dict(row))

    groups: List[Dict[str, Any]] = []
    for key_json, members in grouped.items():
        members_sorted = sorted(members, key=lambda row: (int(row["first_row_index"]), str(row["binder_sequence"])))
        representative = members_sorted[0]
        template_key_hash = str(representative[key_hash_field])
        groups.append(
            {
                "canonical_template_key_json": key_json,
                "canonical_template_key_hash": template_key_hash,
                "template_key_json": key_json,
                "template_key_hash": template_key_hash,
                "template_grouping_mode": grouping_mode,
                "representative": representative,
                "members": members_sorted,
                "sort_key": (int(representative["first_row_index"]), template_key_hash),
            }
        )

    groups.sort(key=lambda group: group["sort_key"])
    for idx, group in enumerate(groups, start=1):
        group["template_id"] = f"vhh_tpl_{idx:04d}_{group['canonical_template_key_hash'][:8]}"
        del group["sort_key"]
    return groups


def assignments_rows_from_groups(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        template_id = str(group["template_id"])
        members = list(group["members"])
        representative = dict(group["representative"])
        group_size = len(members)
        rep_seq = str(representative["binder_sequence"])
        for member in members:
            row = dict(member)
            row["template_id"] = template_id
            row["is_representative"] = str(row["binder_sequence"]) == rep_seq
            row["group_size"] = group_size
            row["template_grouping_mode"] = str(group.get("template_grouping_mode", "exact"))
            rows.append(row)
    rows.sort(key=lambda row: (int(row["first_row_index"]), str(row["binder_sequence"])))
    return rows


def build_template_groups(
    analyzed_records: Sequence[Dict[str, Any]],
    representative_policy: str = "first",
    max_templates: int = 0,
    max_members_per_template: int = 0,
    framework_mode: str = "lengths_only",
) -> List[Dict[str, Any]]:
    framework_mode_n = normalize_framework_mode(framework_mode)
    groups = group_template_members(
        analyzed_records,
        representative_policy=representative_policy,
        framework_mode=framework_mode_n,
    )
    if max_templates and max_templates > 0:
        groups = groups[: int(max_templates)]

    for group in groups:
        representative = dict(group["representative"])
        members = list(group["members"])
        if max_members_per_template and max_members_per_template > 0:
            members_for_materialization = members[: int(max_members_per_template)]
        else:
            members_for_materialization = list(members)
        group["members_for_materialization"] = members_for_materialization
        group["group_size"] = len(members)
        group["representative_name"] = str(representative["binder_name_first"])
        group["representative_sequence"] = str(representative["binder_sequence"])
        group["numbering_scheme"] = str(representative["numbering_scheme"])
        group["chain_class"] = str(representative["chain_class"])
        group["template_grouping_mode"] = str(group.get("template_grouping_mode") or framework_mode_n)
        group["fr1"] = str(representative["fr1"])
        group["fr2"] = str(representative["fr2"])
        group["fr3"] = str(representative["fr3"])
        group["fr4"] = str(representative["fr4"])
        group["fr1_length"] = int(representative["fr1_length"])
        group["fr2_length"] = int(representative["fr2_length"])
        group["fr3_length"] = int(representative["fr3_length"])
        group["fr4_length"] = int(representative["fr4_length"])
        group["cdr1_length"] = int(representative["cdr1_length"])
        group["cdr2_length"] = int(representative["cdr2_length"])
        group["cdr3_length"] = int(representative["cdr3_length"])
        group["cdr1_register"] = str(representative["cdr1_register"])
        group["cdr2_register"] = str(representative["cdr2_register"])
        group["cdr3_register"] = str(representative["cdr3_register"])
        group["framework_hash"] = str(representative["framework_hash"])
    return groups


def build_assignment_rows(groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return assignments_rows_from_groups(groups)


def chunked(items: Sequence[Any], size: int) -> List[List[Any]]:
    chunk_size = max(1, int(size))
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def shard_round_robin(items: Sequence[Any], n_shards: int) -> List[List[Any]]:
    shards: List[List[Any]] = [[] for _ in range(max(1, int(n_shards)))]
    for idx, item in enumerate(items):
        shards[idx % len(shards)].append(item)
    return [shard for shard in shards if shard]


def csv_fieldnames(
    rows: Sequence[Dict[str, Any]],
    preferred: Optional[Sequence[str]] = None,
) -> List[str]:
    fields: List[str] = []
    seen: set[str] = set()
    for name in preferred or []:
        if name not in seen:
            fields.append(name)
            seen.add(name)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def template_validation_panel_summary(groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "template_count": len(groups),
        "member_count": sum(len(group.get("members", [])) for group in groups),
        "max_group_size": max((len(group.get("members", [])) for group in groups), default=0),
    }


def segmentation_to_dict(segmentation: VhhSegmentation) -> Dict[str, Any]:
    return asdict(segmentation)


def _looks_like_header(row: Sequence[str]) -> bool:
    lowered = [str(col or "").strip().lower() for col in row[:4]]
    header_tokens = {"binder_name", "binder", "binder_seq", "binder_sequence", "target_name", "target", "target_seq", "target_sequence"}
    return any(token in header_tokens for token in lowered)


def load_pair_rows(pair_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(pair_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        raw = list(reader)

    if not raw:
        return rows

    start = 1 if _looks_like_header(raw[0]) else 0
    for idx, row in enumerate(raw[start:], start=start):
        if not row:
            continue
        cols = list(row) + [""] * max(0, 4 - len(row))
        binder_name, binder_seq, target_name, target_seq = [str(col or "").strip() for col in cols[:4]]
        if not (binder_name and binder_seq and target_name and target_seq):
            continue
        rows.append(
            {
                "row_index": idx,
                "binder_name": binder_name,
                "binder_seq": normalize_sequence(binder_seq),
                "target_name": target_name,
                "target_seq": normalize_sequence(target_seq),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(rows, preferred=preferred)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def run_local_vhh_analysis(
    pair_csv: Path,
    output_dir: Path,
    numbering_scheme: str = "imgt",
    representative_policy: str = "first",
    framework_mode: str = "lengths_only",
    strict_numbering: bool = True,
    max_templates: int = 0,
    max_members_per_template: int = 0,
) -> Dict[str, Any]:
    framework_mode_n = normalize_framework_mode(framework_mode)
    pair_rows = load_pair_rows(pair_csv)
    if not pair_rows:
        raise ValueError(f"No valid rows found in {pair_csv}")

    unique_binders = extract_unique_binders(pair_rows)
    if not unique_binders:
        raise ValueError(f"No valid binder rows found in {pair_csv}")

    analysis = analyze_unique_binders(unique_binders, numbering_scheme=numbering_scheme)
    analyzed = list(analysis["analyzed"])
    rejected = list(analysis["rejected"])
    groups = build_template_groups(
        analyzed,
        representative_policy=representative_policy,
        max_templates=max_templates,
        max_members_per_template=max_members_per_template,
        framework_mode=framework_mode_n,
    )
    assignment_rows = build_assignment_rows(groups)
    template_summary_rows = [
        {
            "template_id": group["template_id"],
            "group_size": group["group_size"],
            "materialization_member_count": len(group.get("members_for_materialization", [])),
            "representative_name": group["representative_name"],
            "representative_sequence": group["representative_sequence"],
            "numbering_scheme": group["numbering_scheme"],
            "chain_class": group["chain_class"],
            "template_grouping_mode": group["template_grouping_mode"],
            "fr1": group["fr1"],
            "fr2": group["fr2"],
            "fr3": group["fr3"],
            "fr4": group["fr4"],
            "fr1_length": group["fr1_length"],
            "fr2_length": group["fr2_length"],
            "fr3_length": group["fr3_length"],
            "fr4_length": group["fr4_length"],
            "cdr1_length": group["cdr1_length"],
            "cdr2_length": group["cdr2_length"],
            "cdr3_length": group["cdr3_length"],
            "cdr1_register": group["cdr1_register"],
            "cdr2_register": group["cdr2_register"],
            "cdr3_register": group["cdr3_register"],
            "framework_hash": group["framework_hash"],
        }
        for group in groups
    ]
    manifest = {
        "pair_csv": str(pair_csv),
        "output_dir": str(output_dir),
        "numbering_scheme": numbering_scheme,
        "representative_policy": representative_policy,
        "framework_mode": framework_mode_n,
        "strict_numbering": bool(strict_numbering),
        "pair_row_count": len(pair_rows),
        "unique_binder_count": len(unique_binders),
        "accepted_binder_count": len(analyzed),
        "rejected_binder_count": len(rejected),
        "template_group_count": len(groups),
        "template_validation_summary": template_validation_panel_summary(groups),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "vhh_template_assignments.csv"
    summary_path = output_dir / "vhh_template_summary.csv"
    rejected_path = output_dir / "rejected_binders.csv"
    manifest_path = output_dir / "vhh_template_manifest.json"

    _write_csv(
        assignments_path,
        assignment_rows,
        preferred=[
            "binder_name_first",
            "binder_sequence",
            "first_row_index",
            "template_id",
            "is_representative",
            "group_size",
            "template_grouping_mode",
            "fr1_length",
            "fr2_length",
            "fr3_length",
            "fr4_length",
        ],
    )
    _write_csv(
        summary_path,
        template_summary_rows,
        preferred=[
            "template_id",
            "group_size",
            "materialization_member_count",
            "representative_name",
            "representative_sequence",
            "template_grouping_mode",
            "fr1_length",
            "fr2_length",
            "fr3_length",
            "fr4_length",
        ],
    )
    _write_csv(
        rejected_path,
        rejected,
        preferred=["binder_name", "binder_sequence", "reason", "stage", "first_row_index"],
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if strict_numbering and rejected:
        raise RuntimeError(
            f"Rejected {len(rejected)} binder sequence(s) during strict VHH numbering. See {rejected_path}."
        )

    return {
        "pair_rows": pair_rows,
        "unique_binders": unique_binders,
        "analyzed": analyzed,
        "rejected": rejected,
        "groups": groups,
        "assignments_path": assignments_path,
        "summary_path": summary_path,
        "rejected_path": rejected_path,
        "manifest_path": manifest_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local VHH template grouping without Modal.")
    parser.add_argument("--pair-csv", required=True, help="CSV with binder_name,binder_seq,target_name,target_seq columns")
    parser.add_argument("--output-dir", required=True, help="Directory for grouping artifacts")
    parser.add_argument("--numbering-scheme", default="imgt")
    parser.add_argument("--representative-policy", default="first")
    parser.add_argument("--framework-mode", choices=["exact", "lengths_only"], default="lengths_only")
    parser.add_argument("--strict-numbering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-templates", type=int, default=0)
    parser.add_argument("--max-members-per-template", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = run_local_vhh_analysis(
        pair_csv=Path(args.pair_csv),
        output_dir=Path(args.output_dir),
        numbering_scheme=str(args.numbering_scheme or "imgt").strip().lower(),
        representative_policy=str(args.representative_policy or "first").strip().lower(),
        framework_mode=str(args.framework_mode or "lengths_only").strip().lower(),
        strict_numbering=bool(args.strict_numbering),
        max_templates=int(args.max_templates),
        max_members_per_template=int(args.max_members_per_template),
    )
    print("VHH TEMPLATE ANALYSIS")
    print(f"Grouping mode    : {normalize_framework_mode(args.framework_mode)}")
    print(f"Pair rows        : {len(result['pair_rows'])}")
    print(f"Unique binders   : {len(result['unique_binders'])}")
    print(f"Accepted binders : {len(result['analyzed'])}")
    print(f"Rejected binders : {len(result['rejected'])}")
    print(f"Template groups  : {len(result['groups'])}")
    print(f"Assignments CSV  : {result['assignments_path']}")
    print(f"Summary CSV      : {result['summary_path']}")
    print(f"Rejected CSV     : {result['rejected_path']}")
    print(f"Manifest JSON    : {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
