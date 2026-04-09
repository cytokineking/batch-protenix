from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


CANONICAL_PAIR_HEADERS: Tuple[str, ...] = (
    "binder_name",
    "binder_sequence",
    "target_name",
    "target_sequence",
)
_CANONICAL_HEADER_SET = set(CANONICAL_PAIR_HEADERS)
_HEADER_ALIAS_TOKENS = {
    "binder",
    "binder_name",
    "binder_seq",
    "binder_sequence",
    "target",
    "target_name",
    "target_seq",
    "target_sequence",
}
_DECOY_HEADER_RE = re.compile(r"^decoy(?:(?P<index>[2-9][0-9]*))?_(?P<kind>name|sequence)$")


def sanitize_name(name: str) -> str:
    text = re.sub(r"[^\w\-.]", "_", str(name))
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def normalize_sequence(seq: str) -> str:
    return re.sub(r"\s+", "", seq or "").upper()


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:n]


def _comparison_group_id(
    row_index: int,
    binder_name: str,
    binder_seq: str,
    target_name: str,
    target_seq: str,
) -> str:
    key = "|".join(
        [
            str(int(row_index)),
            sanitize_name(binder_name),
            normalize_sequence(binder_seq),
            sanitize_name(target_name),
            normalize_sequence(target_seq),
        ]
    )
    return f"cg_r{int(row_index):05d}_{short_hash(key, n=12)}"


def _trim_table(raw: Sequence[Sequence[str]]) -> List[List[str]]:
    last_nonempty = -1
    for row in raw:
        for idx, cell in enumerate(row):
            if str(cell or "").strip():
                last_nonempty = max(last_nonempty, idx)
    width = last_nonempty + 1
    if width <= 0:
        return [[] for _ in raw]
    return [list(row[:width]) + [""] * max(0, width - len(row)) for row in raw]


def _header_detection_token(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip().lower())


def looks_like_header(row: Sequence[str]) -> bool:
    tokens = [_header_detection_token(cell) for cell in row if str(cell or "").strip()]
    if not tokens:
        return False
    for token in tokens:
        if token in _HEADER_ALIAS_TOKENS:
            return True
        if _DECOY_HEADER_RE.match(token):
            return True
    return False


def _line_number_from_row_index(row_index: int) -> int:
    return int(row_index) + 1


def _nonempty_row(row: Sequence[str]) -> bool:
    return any(str(cell or "").strip() for cell in row)


def _row_map(headers: Sequence[str], row: Sequence[str]) -> Dict[str, str]:
    padded = list(row[: len(headers)]) + [""] * max(0, len(headers) - len(row))
    return {str(header): str(value or "").strip() for header, value in zip(headers, padded)}


def _raise_header_error(message: str) -> None:
    guidance = (
        "Headered pair CSVs must use exactly:\n"
        "  binder_name,binder_sequence,target_name,target_sequence\n\n"
        "Wide target+decoy CSVs may additionally use:\n"
        "  decoy_name,decoy_sequence,decoy2_name,decoy2_sequence,..."
    )
    raise ValueError(f"{message}\n\n{guidance}")


def _build_source_row(row_index: int, row_map: Dict[str, str]) -> Dict[str, Any]:
    binder_name = sanitize_name(row_map["binder_name"])
    binder_seq = normalize_sequence(row_map["binder_sequence"])
    target_name = sanitize_name(row_map["target_name"])
    target_seq = normalize_sequence(row_map["target_sequence"])
    return {
        "row_index": int(row_index),
        "binder_name": binder_name,
        "binder_seq": binder_seq,
        "target_name": target_name,
        "target_seq": target_seq,
        "comparison_group_id": _comparison_group_id(
            int(row_index),
            binder_name,
            binder_seq,
            target_name,
            target_seq,
        ),
    }


def _pair_row_from_source(
    source_row: Dict[str, Any],
    *,
    partner_role: str,
    partner_slot: str,
    partner_name: str,
    partner_seq: str,
) -> Dict[str, Any]:
    return {
        "row_index": int(source_row["row_index"]),
        "comparison_group_id": str(source_row["comparison_group_id"]),
        "partner_role": str(partner_role),
        "partner_slot": str(partner_slot),
        "partner_name": sanitize_name(partner_name),
        "partner_seq": normalize_sequence(partner_seq),
        "binder_name": str(source_row["binder_name"]),
        "binder_seq": str(source_row["binder_seq"]),
        "target_name": str(source_row["target_name"]),
        "target_seq": str(source_row["target_seq"]),
    }


def _load_legacy_pair_rows(raw: Sequence[Sequence[str]]) -> Dict[str, Any]:
    pair_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    for row_index, row in enumerate(raw, start=0):
        if not _nonempty_row(row):
            continue
        cols = list(row) + [""] * max(0, 4 - len(row))
        binder_name, binder_seq, target_name, target_seq = [str(col or "").strip() for col in cols[:4]]
        if not (binder_name and binder_seq and target_name and target_seq):
            continue
        source_row = {
            "row_index": int(row_index),
            "binder_name": sanitize_name(binder_name),
            "binder_seq": normalize_sequence(binder_seq),
            "target_name": sanitize_name(target_name),
            "target_seq": normalize_sequence(target_seq),
            "comparison_group_id": _comparison_group_id(
                int(row_index),
                binder_name,
                binder_seq,
                target_name,
                target_seq,
            ),
        }
        source_rows.append(source_row)
        pair_rows.append(
            _pair_row_from_source(
                source_row,
                partner_role="target",
                partner_slot="target",
                partner_name=source_row["target_name"],
                partner_seq=source_row["target_seq"],
            )
        )
    return {
        "input_mode": "legacy_pair",
        "source_rows": source_rows,
        "pair_rows": pair_rows,
        "has_decoy_columns": False,
        "has_materialized_decoys": False,
    }


def _parse_header_mode(raw: Sequence[Sequence[str]]) -> Dict[str, Any]:
    trimmed = _trim_table(raw)
    headers_raw = [str(cell or "").strip() for cell in trimmed[0]]
    headers = [header.lower() for header in headers_raw]
    if not headers or not any(headers):
        return _load_legacy_pair_rows(raw)

    duplicates = sorted({header for header in headers if header and headers.count(header) > 1})
    if duplicates:
        _raise_header_error(f"Duplicate header names are not allowed: {duplicates}")
    if any(not header for header in headers):
        _raise_header_error("Blank header names are not allowed in headered CSV mode.")

    decoy_slots: Dict[str, Dict[str, str]] = {}
    unknown_headers: List[str] = []
    malformed_decoy_headers: List[str] = []
    for header_raw, header in zip(headers_raw, headers):
        if header in _CANONICAL_HEADER_SET:
            continue
        match = _DECOY_HEADER_RE.match(header)
        if match:
            slot = "decoy" if match.group("index") is None else f"decoy{match.group('index')}"
            kind = str(match.group("kind"))
            decoy_slots.setdefault(slot, {})[kind] = header
            continue
        unknown_headers.append(header_raw)
        if "decoy" in header or "binder" in header or "target" in header:
            malformed_decoy_headers.append(header_raw)

    has_decoy_columns = bool(decoy_slots)
    if has_decoy_columns:
        missing_required = sorted(_CANONICAL_HEADER_SET - set(headers))
        if missing_required:
            _raise_header_error(f"Wide CSV is missing required columns: {missing_required}")
        if malformed_decoy_headers:
            _raise_header_error(f"Malformed wide-mode header names: {sorted(malformed_decoy_headers)}")
        if unknown_headers:
            _raise_header_error(f"Unknown extra headers are not allowed in wide mode: {sorted(unknown_headers)}")
        for slot, kinds in decoy_slots.items():
            if sorted(kinds.keys()) != ["name", "sequence"]:
                _raise_header_error(
                    f"Decoy slot '{slot}' must provide both name and sequence headers; found {sorted(kinds.keys())}"
                )
        slot_numbers = sorted(
            int(slot.replace("decoy", "") or "1")
            for slot in decoy_slots.keys()
        )
        expected_numbers = list(range(1, max(slot_numbers, default=0) + 1))
        if slot_numbers != expected_numbers:
            _raise_header_error(
                "Decoy numbering must be contiguous: expected decoy,decoy2,decoy3,... without gaps."
            )
        input_mode = "wide"
    else:
        if set(headers) != _CANONICAL_HEADER_SET:
            if unknown_headers:
                _raise_header_error(
                    f"Unknown extra headers are not allowed in headered canonical pair mode: {sorted(unknown_headers)}"
                )
            _raise_header_error(
                "Headered canonical pair mode requires exactly the canonical 4 column names."
            )
        input_mode = "canonical_pair"

    source_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    duplicate_wide_keys: Dict[Tuple[str, str, str, str], int] = {}

    for row_index, row in enumerate(trimmed[1:], start=1):
        line_number = _line_number_from_row_index(row_index)
        if len(row) > len(headers) and any(str(cell or "").strip() for cell in row[len(headers) :]):
            raise ValueError(f"Line {line_number}: found extra data columns beyond the declared header width.")
        if not _nonempty_row(row):
            continue
        row_values = _row_map(headers, row)
        missing_required_cells = [
            header for header in CANONICAL_PAIR_HEADERS if not row_values.get(header, "").strip()
        ]
        if missing_required_cells:
            raise ValueError(
                f"Line {line_number}: missing required values for {missing_required_cells} in strict headered CSV mode."
            )

        source_row = _build_source_row(row_index, row_values)
        if input_mode == "wide":
            duplicate_key = (
                str(source_row["binder_name"]),
                str(source_row["binder_seq"]),
                str(source_row["target_name"]),
                str(source_row["target_seq"]),
            )
            first_row_index = duplicate_wide_keys.get(duplicate_key)
            if first_row_index is not None:
                raise ValueError(
                    "Wide CSV duplicate row detected for "
                    f"binder={duplicate_key[0]!r}, target={duplicate_key[2]!r}. "
                    f"First seen on line {_line_number_from_row_index(first_row_index)}; duplicate on line {line_number}."
                )
            duplicate_wide_keys[duplicate_key] = int(row_index)

        source_rows.append(source_row)
        pair_rows.append(
            _pair_row_from_source(
                source_row,
                partner_role="target",
                partner_slot="target",
                partner_name=source_row["target_name"],
                partner_seq=source_row["target_seq"],
            )
        )

        if input_mode != "wide":
            continue

        for slot_number in sorted(
            int(slot.replace("decoy", "") or "1")
            for slot in decoy_slots.keys()
        ):
            slot = "decoy" if slot_number == 1 else f"decoy{slot_number}"
            name_header = decoy_slots[slot]["name"]
            seq_header = decoy_slots[slot]["sequence"]
            decoy_name = str(row_values.get(name_header, "")).strip()
            decoy_seq = str(row_values.get(seq_header, "")).strip()
            if bool(decoy_name) != bool(decoy_seq):
                raise ValueError(
                    f"Line {line_number}: slot {slot!r} must provide both name and sequence or neither."
                )
            if not decoy_name and not decoy_seq:
                continue
            pair_rows.append(
                _pair_row_from_source(
                    source_row,
                    partner_role="decoy",
                    partner_slot=slot,
                    partner_name=decoy_name,
                    partner_seq=decoy_seq,
                )
            )

    return {
        "input_mode": input_mode,
        "source_rows": source_rows,
        "pair_rows": pair_rows,
        "has_decoy_columns": has_decoy_columns,
        "has_materialized_decoys": any(row["partner_role"] == "decoy" for row in pair_rows),
    }


def load_input_csv(path: Path) -> Dict[str, Any]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.reader(handle))
    if not raw_rows:
        return {
            "input_mode": "empty",
            "source_rows": [],
            "pair_rows": [],
            "has_decoy_columns": False,
            "has_materialized_decoys": False,
        }
    if looks_like_header(raw_rows[0]):
        return _parse_header_mode(raw_rows)
    return _load_legacy_pair_rows(raw_rows)


def load_pair_rows(path: Path) -> List[Dict[str, Any]]:
    return list(load_input_csv(path).get("pair_rows") or [])
