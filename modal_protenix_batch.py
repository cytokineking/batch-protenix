#!/usr/bin/env python3
"""
Modal deployment for Protenix v1 + ipSAE batch pipeline.

This pipeline is row-centric:
- Input CSV rows are treated as independent binder/target pairs.
- Protenix is run for each pair (and optionally antitarget/self variants).
- Best sample is selected by top ipTM.
- Optional ipSAE scoring is computed from Protenix outputs via an adapter.

MSA policy:
- Never uses the Protenix MSA server mode.
- Uses ColabFold/MMseqs server fetches when dynamic MSA generation is needed.
- Supports persistent MSA caching on a Modal volume.
"""

import csv
import datetime
import hashlib
import importlib
import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import modal


# =============================================================================
# APP + RESOURCES
# =============================================================================

app = modal.App("protenix-ipsae")

runtime_volume = modal.Volume.from_name("protenix-runtime-cache", create_if_missing=True)
msa_cache_volume = modal.Volume.from_name("protenix-msa-cache", create_if_missing=True)
results_dict = modal.Dict.from_name("protenix-results", create_if_missing=True)

GPU_TYPES = {
    "T4": "16GB - $0.59/h",
    "L4": "24GB - $0.80/h",
    "A10G": "24GB - $1.10/h",
    "L40S": "48GB - $1.95/h",
    "A100-40GB": "40GB - $2.10/h",
    "A100-80GB": "80GB - $2.50/h (DEFAULT)",
    "H100": "80GB - $3.95/h",
    "H200": "141GB - $4.54/h",
    "B200": "192GB - $6.25/h",
}
DEFAULT_GPU = "A100-80GB"

RUNTIME_ROOT = Path("/protenix_root")
CHECKPOINT_DIR = RUNTIME_ROOT / "checkpoint"
COMMON_DIR = RUNTIME_ROOT / "common"
MSA_CACHE_ROOT = Path("/msa_cache")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch>=2.4.0",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "scipy>=1.11",
        "pyyaml>=6.0",
        "requests>=2.31",
        "tqdm>=4.66",
        "biopython>=1.83",
        "biotite>=1.1.0",
        "gemmi>=0.6.3",
        "einops>=0.7",
        "ml-collections>=0.1.1",
        "dm-tree>=0.1.8",
        "hydra-core>=1.3",
        "pytorch-lightning>=2.0",
        "modelcif>=1.0",
        "fair-esm==2.0.0",
        "rdkit==2025.9.3",
        "scikit-learn>=1.5",
        "optree>=0.13",
        "typing_extensions",
        "protobuf",
    )
    .run_commands(
        "pip install cuequivariance-torch cuequivariance-ops-torch-cu12 || echo 'cuequivariance unavailable; continuing'",
    )
    .env(
        {
            "PYTHONPATH": "/root/Protenix:/root",
            "PROTENIX_ROOT_DIR": str(RUNTIME_ROOT),
            "LAYERNORM_TYPE": "torch",
        }
    )
    .add_local_dir("Protenix", "/root/Protenix", copy=True)
    .add_local_file("ipsae.py", "/root/ipsae.py", copy=True)
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def _sanitize_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def _slugify(name: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(name)).strip("-").lower()
    if not slug:
        slug = "x"
    return slug[:max_len]


def _short_hash(text: str, n: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _normalize_sequence(seq: str) -> str:
    return re.sub(r"\s+", "", seq or "").upper()


def _normalize_for_identity(seq: str) -> str:
    # Remove A3M lowercase insertions before uppercasing, then drop gaps/symbols.
    s = re.sub(r"\s+", "", seq or "")
    s = re.sub(r"[a-z]", "", s)
    s = re.sub(r"[\-\.]", "", s)
    s = s.upper()
    return re.sub(r"[^A-Z]", "", s)


def _identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def _sequence_is_protein_like(seq: str) -> bool:
    return bool(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYX]+", _normalize_sequence(seq)))


def _extract_query_from_a3m(content: str) -> str:
    lines = [ln.rstrip("\n") for ln in content.splitlines()]
    seq_lines: List[str] = []
    in_seq = False
    for line in lines:
        if line.startswith(">"):
            if in_seq and seq_lines:
                break
            in_seq = True
            continue
        if in_seq:
            if line.strip():
                seq_lines.append(line.strip())
    return "".join(seq_lines)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _cache_mode_flags(cache_mode: str, store_alias: bool = False) -> Tuple[bool, bool]:
    mode = (cache_mode or "readwrite").strip().lower()
    read = mode in {"read", "readwrite"}
    write = mode in {"write", "readwrite"}
    if store_alias:
        write = True
    return read, write


def _resolve_mmseqs_host(
    host_url: Optional[str],
    host_policy: str,
) -> str:
    policy = (host_policy or "strict").strip().lower()
    host = (host_url or "").strip()

    if not host:
        if policy == "allow-default":
            host = "https://api.colabfold.com"
        else:
            raise ValueError(
                "Host policy is strict: --mmseqs-host-url is required."
            )

    parsed = urlparse(host)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Invalid --mmseqs-host-url: {host}")

    disallowed = {"protenix-server.com", "www.protenix-server.com"}
    if parsed.netloc in disallowed:
        raise ValueError(
            f"Disallowed mmseqs host '{parsed.netloc}' for this pipeline policy."
        )

    if policy == "strict":
        allowlist = {"api.colabfold.com", "colabfold.mmseqs.com"}
        if parsed.netloc not in allowlist:
            raise ValueError(
                f"Host policy strict rejects '{parsed.netloc}'. Allowed: {sorted(allowlist)}"
            )

    return host.rstrip("/")


def _build_cache_key(
    sequence: str,
    role: str,
    host_url: str,
    msa_mode: str,
    pairing_strategy: str,
    db_tag: str,
) -> str:
    payload = "|".join(
        [
            _normalize_sequence(sequence),
            role,
            msa_mode,
            host_url,
            pairing_strategy,
            db_tag,
        ]
    )
    return _short_hash(payload, n=32)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# CSV + MSA MAP PARSING
# =============================================================================

def _looks_like_header(row: Sequence[str]) -> bool:
    s = " ".join(x.strip().lower() for x in row if x is not None)
    return (
        "binder" in s
        and ("target" in s or "antigen" in s)
        and "seq" in s
    )


def _load_pair_rows(pair_csv: Path) -> List[Dict[str, Any]]:
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
        binder_name, binder_seq, target_name, target_seq = [c.strip() for c in cols[:4]]
        if not (binder_name and binder_seq and target_name and target_seq):
            continue
        bseq = _normalize_sequence(binder_seq)
        tseq = _normalize_sequence(target_seq)
        rows.append(
            {
                "row_index": idx,
                "binder_name": _sanitize_name(binder_name),
                "binder_seq": bseq,
                "target_name": _sanitize_name(target_name),
                "target_seq": tseq,
            }
        )
    return rows


def _read_msa_pair(
    msa_dir: Optional[Path] = None,
    pairing_path: Optional[Path] = None,
    non_pairing_path: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    if msa_dir is not None:
        pairing_path = msa_dir / "pairing.a3m"
        non_pairing_path = msa_dir / "non_pairing.a3m"

    pairing = pairing_path.read_text() if pairing_path and pairing_path.exists() else None
    non_pairing = (
        non_pairing_path.read_text() if non_pairing_path and non_pairing_path.exists() else None
    )

    if not pairing and non_pairing:
        query = _extract_query_from_a3m(non_pairing)
        pairing = f">query\n{query}\n" if query else None

    return {"pairing": pairing, "non_pairing": non_pairing}


def _parse_target_msa_map_csv(path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Returns lookup keys:
    - seq:<target_sequence>
    - name:<target_name>
    - name_seq:<target_name>|<target_sequence>
    """
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_l = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            tname = row_l.get("target_name") or row_l.get("name") or row_l.get("target")
            tseq = _normalize_sequence(row_l.get("target_sequence") or row_l.get("sequence") or "")

            msa_dir_s = row_l.get("msa_dir") or row_l.get("msa_path")
            pairing_s = row_l.get("pairing_path")
            non_pairing_s = row_l.get("non_pairing_path") or row_l.get("unpaired_msa_path")

            msa_dir = Path(msa_dir_s) if msa_dir_s else None
            pairing_path = Path(pairing_s) if pairing_s else None
            non_pairing_path = Path(non_pairing_s) if non_pairing_s else None

            pair = _read_msa_pair(msa_dir=msa_dir, pairing_path=pairing_path, non_pairing_path=non_pairing_path)
            if not pair.get("non_pairing"):
                continue

            if tname and tseq:
                mapping[f"name_seq:{_sanitize_name(tname)}|{tseq}"] = pair
            if tseq:
                mapping[f"seq:{tseq}"] = pair
            if tname:
                mapping[f"name:{_sanitize_name(tname)}"] = pair
    return mapping


def _target_msa_lookup(
    mapping: Dict[str, Dict[str, Optional[str]]],
    target_name: str,
    target_seq: str,
) -> Optional[Dict[str, Optional[str]]]:
    keys = [
        f"name_seq:{_sanitize_name(target_name)}|{_normalize_sequence(target_seq)}",
        f"seq:{_normalize_sequence(target_seq)}",
        f"name:{_sanitize_name(target_name)}",
    ]
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _load_single_name_sequence_csv(path: Path, label: str) -> Tuple[str, str]:
    """
    Load a one-entry CSV with columns: name, sequence.
    Header row is optional. This helper intentionally supports exactly one entry.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"{label} CSV is empty: {path}")

    start = 0
    h0 = [c.strip().lower() for c in rows[0]]
    if len(h0) >= 2 and ("name" in h0[0]) and ("seq" in h0[1] or "sequence" in h0[1]):
        start = 1

    values: List[Tuple[str, str]] = []
    for row in rows[start:]:
        if not row:
            continue
        cols = list(row) + ["", ""]
        name = cols[0].strip()
        seq = _normalize_sequence(cols[1])
        if not (name and seq):
            continue
        values.append((_sanitize_name(name), seq))

    if not values:
        raise ValueError(f"No valid entries found in {label} CSV: {path}")
    if len(values) > 1:
        raise ValueError(
            f"{label} CSV currently supports exactly one entry; found {len(values)} in {path}"
        )
    return values[0]


# =============================================================================
# PROTENIX BOOTSTRAP
# =============================================================================

def _download_with_retry(url: str, out_path: Path, retries: int = 4) -> None:
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, str(out_path))
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(min(30, 2 ** i))
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def _checkpoint_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 100 * 1024 * 1024:
        return False
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ok = isinstance(ckpt, dict) and "model" in ckpt
        del ckpt
        return bool(ok)
    except Exception:  # noqa: BLE001
        return False


def _checkpoint_file_present(path: Path) -> bool:
    # Fast existence check used in per-task inference validation.
    return bool(path.exists() and path.stat().st_size >= 100 * 1024 * 1024)


def _ensure_protenix_runtime(model_name: str, populate_missing: bool = False) -> None:
    """
    Validate runtime files for inference, or populate them during explicit init.

    - populate_missing=False: fail-fast validation only (no downloads, no lock)
    - populate_missing=True: download missing/invalid runtime files, then commit volume
    """
    required_common = {
        "ccd_components_file": "components.cif",
        "ccd_components_rdkit_mol_file": "components.cif.rdkit_mol.pkl",
        "pdb_cluster_file": "clusters-by-entity-40.txt",
        "obsolete_release_data_csv": "obsolete_release_date.csv",
    }

    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    COMMON_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}.pt"
    missing: List[str] = []

    if populate_missing:
        from protenix.web_service.dependency_url import URL

        if model_name not in URL:
            raise ValueError(f"Unknown Protenix model: {model_name}")

        if not _checkpoint_is_valid(checkpoint_path):
            if checkpoint_path.exists():
                checkpoint_path.unlink(missing_ok=True)
            print(f"Downloading checkpoint for {model_name}...")
            _download_with_retry(URL[model_name], checkpoint_path)
            if not _checkpoint_is_valid(checkpoint_path):
                raise RuntimeError(f"Checkpoint validation failed for {checkpoint_path}")

        for key, fname in required_common.items():
            fpath = COMMON_DIR / fname
            if fpath.exists() and fpath.stat().st_size > 0:
                continue
            url = URL[key]
            print(f"Downloading runtime cache: {fname}")
            _download_with_retry(url, fpath)
            if not fpath.exists() or fpath.stat().st_size == 0:
                raise RuntimeError(f"Failed to prepare runtime cache file {fpath}")

        runtime_volume.commit()
        return

    if not _checkpoint_file_present(checkpoint_path):
        missing.append(str(checkpoint_path))

    for fname in required_common.values():
        fpath = COMMON_DIR / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing.append(str(fpath))

    if missing:
        msg = [
            "Runtime cache is missing required Protenix artifacts:",
            *[f"  - {p}" for p in missing],
            "Initialize once with:",
            f"  modal run modal_protenix_batch.py::init_protenix_runtime --model-name {model_name}",
        ]
        raise RuntimeError("\n".join(msg))


def _check_protenix_imports() -> Dict[str, str]:
    """Return import errors for required runtime modules."""
    required_modules = [
        "esm",
        "rdkit",
        "sklearn",
        "optree",
        "protenix.data.inference.infer_dataloader",
        "runner.inference",
    ]
    errors: Dict[str, str] = {}
    for mod in required_modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            errors[mod] = f"{type(exc).__name__}: {exc}"
    return errors


# =============================================================================
# MSA FETCH + CACHE
# =============================================================================

def _parse_fasta_string(fasta_string: str) -> Dict[str, str]:
    fasta_dict: Dict[str, str] = {}
    header = ""
    for line in fasta_string.strip().split("\n"):
        if line.startswith(">"):
            header = line[1:].strip()
            fasta_dict[header] = ""
        else:
            fasta_dict[header] += line.strip()
    return fasta_dict


def _fetch_msa_colabfold(
    sequence: str,
    host_url: str,
    user_agent: str,
    pairing_strategy: str = "greedy",
    max_submit_retries: int = 6,
    max_status_polls: int = 120,
) -> Dict[str, str]:
    """
    Fetch non-pairing MSA from ColabFold server, then synthesize pairing MSA content.
    Returns A3M content strings: {"non_pairing": ..., "pairing": ...}
    """
    import requests
    import tarfile

    strategy = (pairing_strategy or "greedy").strip().lower()
    if strategy not in {"greedy", "query_only", "copy_non_pairing"}:
        raise ValueError(
            f"Unsupported pairing_strategy '{pairing_strategy}'. "
            "Use one of: greedy, query_only, copy_non_pairing"
        )

    query_seq = _normalize_sequence(sequence)
    query_fasta = f">query_0\n{query_seq}"

    headers = {"User-Agent": user_agent}

    def submit_once() -> Dict[str, Any]:
        resp = requests.post(
            f"{host_url}/ticket/msa",
            data={"q": query_fasta, "mode": "env"},
            timeout=20,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()

    out = None
    for attempt in range(max_submit_retries):
        out = submit_once()
        status = str(out.get("status", "")).upper()
        if status in {"UNKNOWN", "RATELIMIT"}:
            sleep_s = min(60, 2 ** attempt + 3)
            time.sleep(sleep_s)
            continue
        break

    if not out:
        raise RuntimeError("MSA submit failed: empty response")

    status = str(out.get("status", "")).upper()
    if status in {"ERROR", "MAINTENANCE"}:
        raise RuntimeError(f"MSA submit failed with status={status}")

    ticket_id = out.get("id")
    if not ticket_id:
        raise RuntimeError(f"MSA submit did not return ticket id: {out}")

    current = out
    for _ in range(max_status_polls):
        status = str(current.get("status", "")).upper()
        if status == "COMPLETE":
            break
        if status in {"ERROR", "MAINTENANCE"}:
            raise RuntimeError(f"MSA job failed with status={status}")
        time.sleep(10)
        resp = requests.get(f"{host_url}/ticket/{ticket_id}", timeout=20, headers=headers)
        resp.raise_for_status()
        current = resp.json()
    else:
        raise TimeoutError(f"MSA polling timed out for ticket {ticket_id}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tar_path = td_path / "out.tar.gz"

        dl = requests.get(
            f"{host_url}/result/download/{ticket_id}", timeout=60, headers=headers
        )
        dl.raise_for_status()
        tar_path.write_bytes(dl.content)

        with tarfile.open(tar_path) as tgz:
            tgz.extractall(td_path)

        env_file = td_path / "bfd.mgnify30.metaeuk30.smag30.a3m"
        uniref_file = td_path / "uniref.a3m"

        lines: List[str] = [">query", query_seq]

        for fpath in [env_file, uniref_file]:
            if not fpath.exists():
                continue
            content = fpath.read_text(errors="ignore").replace("\x00", "")
            parsed = _parse_fasta_string(content)
            for k, v in parsed.items():
                if k.startswith("query_"):
                    continue
                lines.append(f">{k}")
                lines.append(v)

        non_pairing = "\n".join(lines).strip() + "\n"
        # For monomer queries there is no true pair-wise partner context. Keep default
        # behavior query-only; allow explicit copy_non_pairing when requested.
        if strategy == "copy_non_pairing":
            pairing = non_pairing
        else:
            pairing = f">query\n{query_seq}\n"
        return {"non_pairing": non_pairing, "pairing": pairing}


def _cache_entry_valid(cache_dir: Path) -> bool:
    return (cache_dir / "non_pairing.a3m").exists()


def _write_cache_entry(
    cache_dir: Path,
    pairing: Optional[str],
    non_pairing: str,
    metadata: Dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "non_pairing.a3m").write_text(non_pairing)
    if pairing:
        (cache_dir / "pairing.a3m").write_text(pairing)
    _save_json(cache_dir / "metadata.json", metadata)


def _read_cache_entry(cache_dir: Path) -> Dict[str, Optional[str]]:
    pair_path = cache_dir / "pairing.a3m"
    non_path = cache_dir / "non_pairing.a3m"
    return {
        "pairing": pair_path.read_text() if pair_path.exists() else None,
        "non_pairing": non_path.read_text() if non_path.exists() else None,
    }


def _get_or_fetch_cached_msa(
    sequence: str,
    role: str,
    host_url: str,
    msa_mode: str,
    pairing_strategy: str,
    db_tag: str,
    read_cache: bool,
    write_cache: bool,
    max_fetch_attempts: int,
) -> Dict[str, Any]:
    cache_key = _build_cache_key(
        sequence=sequence,
        role=role,
        host_url=host_url,
        msa_mode=msa_mode,
        pairing_strategy=pairing_strategy,
        db_tag=db_tag,
    )
    cache_dir = MSA_CACHE_ROOT / cache_key

    if read_cache and _cache_entry_valid(cache_dir):
        return {
            "status": "cached",
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
        }

    last_err: Optional[Exception] = None
    for attempt in range(max_fetch_attempts):
        try:
            fetched = _fetch_msa_colabfold(
                sequence=sequence,
                host_url=host_url,
                user_agent="batch-protenix/1.0",
                pairing_strategy=pairing_strategy,
            )
            if write_cache:
                metadata = {
                    "sequence_sha256": hashlib.sha256(_normalize_sequence(sequence).encode("utf-8")).hexdigest(),
                    "role": role,
                    "host_url": host_url,
                    "msa_mode": msa_mode,
                    "pairing_strategy": pairing_strategy,
                    "db_tag": db_tag,
                    "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
                }
                _write_cache_entry(
                    cache_dir=cache_dir,
                    pairing=fetched.get("pairing"),
                    non_pairing=fetched["non_pairing"],
                    metadata=metadata,
                )
                msa_cache_volume.commit()
                return {
                    "status": "fetched_and_cached",
                    "cache_key": cache_key,
                    "cache_dir": str(cache_dir),
                }

            # No-cache modes return inline content and do not write to shared volume.
            return {
                "status": "fetched_no_cache",
                "pairing": fetched.get("pairing"),
                "non_pairing": fetched["non_pairing"],
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            sleep_s = min(60, (2 ** attempt) + 2)
            time.sleep(sleep_s)

    raise RuntimeError(f"MSA fetch failed after retries: {last_err}")


@app.function(
    image=image,
    timeout=7200,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def precompute_msas(
    sequences: List[str],
    role: str,
    host_url: str,
    cache_mode: str = "readwrite",
    store_fetched_msas: bool = False,
    msa_mode: str = "colabfold",
    pairing_strategy: str = "greedy",
    db_tag: str = "colabfold_env",
    max_fetch_attempts: int = 4,
) -> Dict[str, Dict[str, Any]]:
    read_cache, write_cache = _cache_mode_flags(cache_mode, store_alias=store_fetched_msas)

    results: Dict[str, Dict[str, Any]] = {}
    unique_sequences = sorted({_normalize_sequence(s) for s in sequences if s})

    for seq in unique_sequences:
        try:
            info = _get_or_fetch_cached_msa(
                sequence=seq,
                role=role,
                host_url=host_url,
                msa_mode=msa_mode,
                pairing_strategy=pairing_strategy,
                db_tag=db_tag,
                read_cache=read_cache,
                write_cache=write_cache,
                max_fetch_attempts=max_fetch_attempts,
            )
            results[seq] = info
        except Exception as exc:  # noqa: BLE001
            results[seq] = {"status": "error", "error": str(exc)}

    return results


@app.function(
    image=image,
    timeout=1800,
    max_containers=1,
    volumes={str(RUNTIME_ROOT): runtime_volume},
)
def preflight_protenix_runtime(model_name: str) -> Dict[str, Any]:
    """
    One-time preflight:
    - verify required Python dependencies import cleanly
    - verify checkpoint + common runtime assets are present on shared volume
    """
    import_errors = _check_protenix_imports()
    if import_errors:
        raise RuntimeError(f"Dependency preflight failed: {import_errors}")

    _ensure_protenix_runtime(model_name, populate_missing=False)
    return {"status": "ok", "model_name": model_name}


def _load_msa_ref(msa_ref: Dict[str, Any]) -> Dict[str, Optional[str]]:
    source = msa_ref.get("source")
    if source == "none":
        return {"pairing": None, "non_pairing": None}
    if source == "inline":
        return {
            "pairing": msa_ref.get("pairing"),
            "non_pairing": msa_ref.get("non_pairing"),
        }
    if source == "cache":
        cache_key = msa_ref.get("cache_key")
        if not cache_key:
            return {"pairing": None, "non_pairing": None}
        cache_dir = MSA_CACHE_ROOT / cache_key
        if not _cache_entry_valid(cache_dir):
            return {"pairing": None, "non_pairing": None}
        return _read_cache_entry(cache_dir)
    return {"pairing": None, "non_pairing": None}


# =============================================================================
# ipSAE ADAPTER + PARSING
# =============================================================================

def _convert_chain_pair_iptm_to_matrix(chain_pair: Any) -> Optional[List[List[float]]]:
    if chain_pair is None:
        return None
    if isinstance(chain_pair, list):
        try:
            return [[float(x) for x in row] for row in chain_pair]
        except Exception:  # noqa: BLE001
            return None
    if isinstance(chain_pair, dict):
        # Expect keys like "0", "1"... each mapping to dict/list.
        idxs = sorted(int(k) for k in chain_pair.keys() if str(k).isdigit())
        n = (max(idxs) + 1) if idxs else 0
        mat = [[0.0 for _ in range(n)] for _ in range(n)]
        for i_str, row in chain_pair.items():
            if not str(i_str).isdigit():
                continue
            i = int(i_str)
            if isinstance(row, dict):
                for j_str, val in row.items():
                    if str(j_str).isdigit():
                        mat[i][int(j_str)] = float(val)
            elif isinstance(row, list):
                for j, val in enumerate(row):
                    mat[i][j] = float(val)
        return mat
    return None


def _write_ipsae_adapter_files(
    full_data_path: Path,
    summary_path: Path,
    out_dir: Path,
) -> Tuple[Path, Path]:
    full_data = _load_json(full_data_path)
    summary = _load_json(summary_path)

    pae = full_data.get("token_pair_pae")
    atom_plddt = full_data.get("atom_plddt")
    if pae is None or atom_plddt is None:
        raise ValueError("Missing token_pair_pae or atom_plddt in Protenix full_data JSON")

    # Protenix atom_plddt is typically 0-1; ipsae.py expects AF3-style 0-100 atom_plddts.
    atom_plddts = [float(x) for x in atom_plddt]
    if atom_plddts and max(atom_plddts) <= 1.5:
        atom_plddts = [x * 100.0 for x in atom_plddts]

    adapted_full = {
        "pae": pae,
        "atom_plddts": atom_plddts,
    }

    chain_pair_iptm = _convert_chain_pair_iptm_to_matrix(summary.get("chain_pair_iptm"))
    adapted_summary = {"chain_pair_iptm": chain_pair_iptm or [[0.0, 0.0], [0.0, 0.0]]}

    out_dir.mkdir(parents=True, exist_ok=True)
    adapted_full_path = out_dir / (full_data_path.stem + "_ipsae.json")
    _save_json(adapted_full_path, adapted_full)

    # ipsae.py expects summary filename by replacing full_data -> summary_confidences
    expected_summary_name = adapted_full_path.name.replace("full_data", "summary_confidences")
    adapted_summary_path = out_dir / expected_summary_name
    _save_json(adapted_summary_path, adapted_summary)

    return adapted_full_path, adapted_summary_path


def _is_binder_partner_pair(
    chain1: str,
    chain2: str,
    binder_chains: Sequence[str],
    partner_chains: Sequence[str],
) -> bool:
    bset = set(binder_chains)
    pset = set(partner_chains)
    return (chain1 in bset and chain2 in pset) or (chain2 in bset and chain1 in pset)


def _parse_ipsae_output_filtered(
    output: str,
    binder_chains: Sequence[str],
    partner_chains: Sequence[str],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    max_rows: List[Dict[str, Any]] = []
    asym_values: List[float] = []

    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith(("Chn1", "#")):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue

        chain1, chain2 = parts[0], parts[1]
        row_type = parts[4].lower()

        if not _is_binder_partner_pair(chain1, chain2, binder_chains, partner_chains):
            continue

        try:
            if row_type == "max":
                row = {
                    "ipSAE": float(parts[5]),
                    "ipSAE_d0chn": float(parts[6]),
                    "ipSAE_d0dom": float(parts[7]),
                    "ipTM_af": float(parts[8]),
                    "pDockQ": float(parts[10]),
                    "pDockQ2": float(parts[11]),
                    "LIS": float(parts[12]),
                }
                if len(parts) > 13:
                    row["n0res"] = int(parts[13])
                if len(parts) > 14:
                    row["n0chn"] = int(parts[14])
                max_rows.append(row)
            elif row_type == "asym":
                asym_values.append(float(parts[5]))
        except Exception:  # noqa: BLE001
            continue

    if max_rows:
        best = max(max_rows, key=lambda x: x.get("ipSAE", 0.0))
        metrics.update(best)
    if asym_values:
        metrics["ipSAE_min"] = min(asym_values)
        metrics["ipSAE_max"] = max(asym_values)
    return metrics


def _run_ipsae(
    adapted_full_path: Path,
    cif_path: Path,
    pae_cutoff: float,
    dist_cutoff: float,
    binder_chains: Sequence[str],
    partner_chains: Sequence[str],
) -> Dict[str, Any]:
    cmd = [
        "python",
        "/root/ipsae.py",
        str(adapted_full_path),
        str(cif_path),
        str(int(pae_cutoff)),
        str(int(dist_cutoff)),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            "error": f"ipsae failed: {proc.stderr[:500]}",
            "raw_stdout": proc.stdout[-2000:],
            "raw_stderr": proc.stderr[-2000:],
        }

    pae_str = f"{int(pae_cutoff):02d}" if pae_cutoff < 10 else str(int(pae_cutoff))
    dist_str = f"{int(dist_cutoff):02d}" if dist_cutoff < 10 else str(int(dist_cutoff))
    txt_path = Path(str(cif_path).replace(".cif", f"_{pae_str}_{dist_str}.txt"))
    if not txt_path.exists():
        matches = list(cif_path.parent.glob(f"*_{pae_str}_{dist_str}.txt"))
        if matches:
            txt_path = matches[0]
        else:
            return {"error": "ipsae output txt not found"}

    txt = txt_path.read_text()
    parsed = _parse_ipsae_output_filtered(
        output=txt,
        binder_chains=binder_chains,
        partner_chains=partner_chains,
    )
    parsed["raw_text"] = txt
    return parsed


# =============================================================================
# PROTENIX TASK EXECUTION
# =============================================================================

def _build_protenix_input_json(
    sample_name: str,
    binder_seq: str,
    partner_seq: str,
    binder_msa: Dict[str, Optional[str]],
    partner_msa: Dict[str, Optional[str]],
    input_dir: Path,
) -> Tuple[Path, Dict[str, List[str]]]:
    input_dir.mkdir(parents=True, exist_ok=True)

    def write_msa_files(prefix: str, msa: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {"pairedMsaPath": None, "unpairedMsaPath": None}
        if msa.get("pairing"):
            p = input_dir / f"{prefix}_pairing.a3m"
            p.write_text(msa["pairing"])
            out["pairedMsaPath"] = str(p)
        if msa.get("non_pairing"):
            u = input_dir / f"{prefix}_non_pairing.a3m"
            u.write_text(msa["non_pairing"])
            out["unpairedMsaPath"] = str(u)
        return out

    binder_paths = write_msa_files("binder", binder_msa)
    partner_paths = write_msa_files("partner", partner_msa)

    binder_chain = {
        "proteinChain": {
            "sequence": _normalize_sequence(binder_seq),
            "count": 1,
        }
    }
    if binder_paths["pairedMsaPath"]:
        binder_chain["proteinChain"]["pairedMsaPath"] = binder_paths["pairedMsaPath"]
    if binder_paths["unpairedMsaPath"]:
        binder_chain["proteinChain"]["unpairedMsaPath"] = binder_paths["unpairedMsaPath"]

    partner_chain = {
        "proteinChain": {
            "sequence": _normalize_sequence(partner_seq),
            "count": 1,
        }
    }
    if partner_paths["pairedMsaPath"]:
        partner_chain["proteinChain"]["pairedMsaPath"] = partner_paths["pairedMsaPath"]
    if partner_paths["unpairedMsaPath"]:
        partner_chain["proteinChain"]["unpairedMsaPath"] = partner_paths["unpairedMsaPath"]

    payload = [{"name": sample_name, "sequences": [binder_chain, partner_chain]}]
    json_path = input_dir / "input.json"
    _save_json(json_path, payload)

    # Deterministic role mapping for downstream ipSAE filtering.
    chain_role_map = {
        "binder": ["A"],
        "partner": ["B"],
    }
    return json_path, chain_role_map


def _build_protenix_inference_cmd(
    input_json: Path,
    out_dir: Path,
    model_name: str,
    seeds_csv: str,
    n_sample: int,
    n_step: int,
    n_cycle: int,
    use_msa: bool,
) -> List[str]:
    return [
        "python",
        "-m",
        "runner.inference",
        "--input_json_path",
        str(input_json),
        "--dump_dir",
        str(out_dir),
        "--seeds",
        seeds_csv,
        "--model_name",
        model_name,
        "--load_checkpoint_dir",
        str(CHECKPOINT_DIR),
        "--sample_diffusion.N_sample",
        str(n_sample),
        "--sample_diffusion.N_step",
        str(n_step),
        "--model.N_cycle",
        str(n_cycle),
        "--enable_tf32",
        "true",
        "--enable_efficient_fusion",
        "true",
        "--enable_diffusion_shared_vars_cache",
        "true",
        "--need_atom_confidence",
        "true",
        "--use_msa",
        str(use_msa).lower(),
        "--use_template",
        "false",
        "--use_rna_msa",
        "false",
    ]


def _run_protenix_inference_streaming(
    cmd: Sequence[str],
    timeout_s: int,
    on_line: Optional[Any] = None,
    on_tick: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run inference with live stdout/stderr streaming.
    """
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    q: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
    done = {"stdout": False, "stderr": False}
    stdout_tail: deque[str] = deque(maxlen=4000)
    stderr_tail: deque[str] = deque(maxlen=4000)
    start = time.time()
    timed_out = False

    def reader(stream: Any, channel: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                q.put((channel, line))
        finally:
            q.put((channel, None))
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    t_out = threading.Thread(target=reader, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=reader, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    while True:
        now = time.time()
        elapsed = now - start

        if proc.poll() is None and elapsed > timeout_s:
            timed_out = True
            proc.kill()

        if on_tick is not None:
            try:
                on_tick(elapsed)
            except Exception:  # noqa: BLE001
                pass

        try:
            channel, line = q.get(timeout=0.25)
            if line is None:
                done[channel] = True
            else:
                if channel == "stdout":
                    stdout_tail.append(line)
                else:
                    stderr_tail.append(line)

                # Keep container logs live.
                print(f"[inference {channel}] {line.rstrip()}")

                if on_line is not None:
                    try:
                        on_line(channel, line)
                    except Exception:  # noqa: BLE001
                        pass
        except queue.Empty:
            pass

        if proc.poll() is not None and done["stdout"] and done["stderr"] and q.empty():
            break

    t_out.join(timeout=2)
    t_err.join(timeout=2)

    return {
        "returncode": int(proc.returncode) if proc.returncode is not None else -1,
        "stdout_tail": "".join(stdout_tail),
        "stderr_tail": "".join(stderr_tail),
        "timed_out": timed_out,
        "elapsed_s": time.time() - start,
    }


def _parse_sample_rank_from_name(path: Path, prefix: str) -> Optional[int]:
    m = re.search(r"_sample_(\d+)\.json$", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"_sample_(\d+)\.cif$", path.name)
    if m:
        return int(m.group(1))
    return None


def _iter_candidate_file_triples(
    output_dir: Path,
    sample_name: str,
    seeds: Sequence[int],
) -> Sequence[Tuple[int, int, Path, Path, Path]]:
    for seed in seeds:
        pred_dir = output_dir / sample_name / f"seed_{seed}" / "predictions"
        if not pred_dir.exists():
            continue

        summary_files = sorted(pred_dir.glob("*_summary_confidence_sample_*.json"))
        for sf in summary_files:
            rank = _parse_sample_rank_from_name(sf, "summary")
            if rank is None:
                continue

            cif = pred_dir / sf.name.replace("_summary_confidence_sample_", "_sample_").replace(".json", ".cif")
            full_data = pred_dir / sf.name.replace("_summary_confidence_sample_", "_full_data_sample_")
            if not cif.exists() or not full_data.exists():
                continue

            yield int(seed), int(rank), sf, full_data, cif


def _collect_candidates(
    output_dir: Path,
    sample_name: str,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for seed, rank, sf, full_data, cif in _iter_candidate_file_triples(
        output_dir=output_dir,
        sample_name=sample_name,
        seeds=seeds,
    ):
        try:
            summary = _load_json(sf)
        except Exception:  # noqa: BLE001
            continue

        candidates.append(
            {
                "seed": int(seed),
                "sample_rank": int(rank),
                "summary_path": str(sf),
                "full_data_path": str(full_data),
                "cif_path": str(cif),
                "summary": summary,
                "iptm": float(summary.get("iptm", -1.0)),
                "ptm": float(summary.get("ptm", -1.0)),
                "ranking_score": float(summary.get("ranking_score", -1.0)),
            }
        )
    return candidates


def _candidate_sort_key(c: Dict[str, Any]) -> Tuple[float, float, float, float]:
    # Higher iptm/ranking_score first, deterministic tie-break by lower seed/sample.
    return (
        float(c.get("iptm", -1.0)),
        float(c.get("ranking_score", -1.0)),
        -float(c.get("seed", 0)),
        -float(c.get("sample_rank", 0)),
    )


def _select_best_candidate(
    candidates: Sequence[Dict[str, Any]],
    scope: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not candidates:
        raise ValueError("No candidate predictions were found")

    scope_n = (scope or "global").strip().lower()
    if scope_n == "global":
        best = max(candidates, key=_candidate_sort_key)
        return best, []

    # per_seed mode: choose best per seed, then choose final best among those winners.
    per_seed: Dict[int, Dict[str, Any]] = {}
    for c in candidates:
        s = int(c["seed"])
        if s not in per_seed or _candidate_sort_key(c) > _candidate_sort_key(per_seed[s]):
            per_seed[s] = c
    per_seed_best = [per_seed[k] for k in sorted(per_seed.keys())]
    best = max(per_seed_best, key=_candidate_sort_key)
    return best, per_seed_best


def _to_jsonable(payload: Any) -> Any:
    try:
        return json.loads(json.dumps(payload, default=str))
    except Exception:  # noqa: BLE001
        return str(payload)


def _event_key(run_id: str, task_id: str, seq: int) -> str:
    return f"{run_id}:event:{task_id}:{seq:06d}"


def _emit_event_if_needed(
    run_id: Optional[str],
    task_id: str,
    stream: bool,
    event_seq: List[int],
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    if not stream or not run_id:
        return

    seq = int(event_seq[0])
    event_seq[0] += 1
    evt = _to_jsonable(
        {
            "run_id": run_id,
            "task_id": task_id,
            "seq": seq,
            "event_type": event_type,
            "timestamp": time.time(),
            **(payload or {}),
        }
    )
    key = _event_key(run_id, task_id, seq)

    try:
        results_dict[key] = evt
        return
    except Exception as exc:  # noqa: BLE001
        serialized = json.dumps(evt, default=str)
        chunk_size = 180000
        chunks = [serialized[i : i + chunk_size] for i in range(0, len(serialized), chunk_size)]
        header = {
            "run_id": run_id,
            "task_id": task_id,
            "seq": seq,
            "event_type": "chunked_event",
            "timestamp": time.time(),
            "parent_event_type": event_type,
            "parent_seq": seq,
            "chunk_count": len(chunks),
            "error": str(exc),
        }
        try:
            results_dict[key] = header
        except Exception as header_exc:  # noqa: BLE001
            print(f"[stream] failed to emit event and chunk header for {task_id}: {header_exc}")
            return

        for idx, chunk in enumerate(chunks):
            cseq = int(event_seq[0])
            event_seq[0] += 1
            ckey = _event_key(run_id, task_id, cseq)
            try:
                results_dict[ckey] = {
                    "run_id": run_id,
                    "task_id": task_id,
                    "seq": cseq,
                    "event_type": "chunk",
                    "timestamp": time.time(),
                    "parent_seq": seq,
                    "chunk_index": idx,
                    "chunk_count": len(chunks),
                    "data": chunk,
                }
            except Exception as chunk_exc:  # noqa: BLE001
                print(f"[stream] failed to emit chunk {idx+1}/{len(chunks)} for {task_id}: {chunk_exc}")
                break


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")


def _write_stream_artifacts(stream_root: Path, event: Dict[str, Any]) -> None:
    task_id = str(event.get("task_id", "unknown_task"))
    event_type = str(event.get("event_type", ""))
    if event_type not in {"candidate_found", "best_selected"}:
        return

    artifacts_root = stream_root / "artifacts" / task_id
    if event_type == "candidate_found":
        seed = int(event.get("seed", -1))
        sample_rank = int(event.get("sample_rank", -1))
        sample_dir = artifacts_root / f"seed_{seed}" / f"sample_{sample_rank}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        cif_text = event.get("cif_text")
        if isinstance(cif_text, str):
            (sample_dir / "sample.cif").write_text(cif_text)

        summary_json = event.get("summary_confidence_json")
        if isinstance(summary_json, dict):
            _save_json(sample_dir / "summary_confidence.json", summary_json)

        full_data_json = event.get("full_data_json")
        if isinstance(full_data_json, dict):
            _save_json(sample_dir / "full_data.json", full_data_json)
        return

    best_dir = artifacts_root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    cif_text = event.get("cif_text")
    if isinstance(cif_text, str):
        (best_dir / "best_sample.cif").write_text(cif_text)
    summary_json = event.get("summary_confidence_json")
    if isinstance(summary_json, dict):
        _save_json(best_dir / "best_summary_confidence.json", summary_json)
    full_data_json = event.get("full_data_json")
    if isinstance(full_data_json, dict):
        _save_json(best_dir / "best_full_data.json", full_data_json)


def _save_stream_event(
    output_root: Path,
    run_id: str,
    dict_key: str,
    event: Dict[str, Any],
    chunk_state: Dict[Tuple[str, int], Dict[str, Any]],
) -> None:
    stream_root = output_root / "_stream" / run_id
    events_path = stream_root / "events.jsonl"
    _append_jsonl(events_path, {"dict_key": dict_key, "event": event})

    event_type = str(event.get("event_type", ""))
    task_id = str(event.get("task_id", "unknown_task"))

    if event_type == "chunked_event":
        parent_seq = int(event.get("parent_seq", event.get("seq", -1)))
        key = (task_id, parent_seq)
        state = chunk_state.setdefault(key, {"header": None, "chunk_count": 0, "chunks": {}})
        state["header"] = event
        state["chunk_count"] = int(event.get("chunk_count", 0))
        return

    if event_type == "chunk":
        parent_seq = int(event.get("parent_seq", -1))
        chunk_index = int(event.get("chunk_index", -1))
        chunk_count = int(event.get("chunk_count", 0))
        key = (task_id, parent_seq)
        state = chunk_state.setdefault(key, {"header": None, "chunk_count": chunk_count, "chunks": {}})
        state["chunk_count"] = max(int(state.get("chunk_count", 0)), chunk_count)
        if chunk_index >= 0:
            state["chunks"][chunk_index] = str(event.get("data", ""))

        expected = int(state.get("chunk_count", 0))
        if expected > 0 and len(state["chunks"]) >= expected:
            try:
                payload = "".join(state["chunks"].get(i, "") for i in range(expected))
                reconstructed = json.loads(payload)
                _append_jsonl(
                    events_path,
                    {
                        "dict_key": f"{dict_key}:reconstructed",
                        "reconstructed_from_chunk": True,
                        "event": reconstructed,
                    },
                )
                _save_stream_event(
                    output_root=output_root,
                    run_id=run_id,
                    dict_key=f"{dict_key}:reconstructed_payload",
                    event=reconstructed,
                    chunk_state={},
                )
            except Exception as exc:  # noqa: BLE001
                _append_jsonl(
                    events_path,
                    {
                        "dict_key": f"{dict_key}:reconstruct_error",
                        "error": str(exc),
                        "task_id": task_id,
                        "parent_seq": parent_seq,
                    },
                )
            finally:
                chunk_state.pop(key, None)
        return

    if event_type == "log_chunk":
        log_dir = stream_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        channel = str(event.get("channel", "stdout"))
        text = str(event.get("text", ""))
        with open(log_dir / f"{task_id}.log", "a", encoding="utf-8") as f:
            if text:
                f.write(text)
            if text and not text.endswith("\n"):
                f.write("\n")
        with open(log_dir / f"{task_id}.{channel}.log", "a", encoding="utf-8") as f:
            if text:
                f.write(text)
            if text and not text.endswith("\n"):
                f.write("\n")
        return

    _write_stream_artifacts(stream_root=stream_root, event=event)

    status_dir = stream_root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "task_id": task_id,
        "event_type": event_type,
        "timestamp": event.get("timestamp"),
        "status": event.get("status"),
        "stage": event.get("stage"),
        "message": event.get("message"),
        "elapsed_s": event.get("elapsed_s"),
        "best_iptm": event.get("best_iptm"),
        "error": event.get("error"),
    }
    _save_json(status_dir / f"{task_id}.json", _to_jsonable(status_payload))


def _stream_result_if_needed(result: Dict[str, Any], run_id: Optional[str], task_id: str, stream: bool) -> None:
    if not stream or not run_id:
        return
    key = f"{run_id}:{task_id}"
    payload = dict(result)
    payload["timestamp"] = time.time()
    results_dict[key] = payload


# =============================================================================
# MODAL TASK FUNCTION
# =============================================================================

def _run_protenix_task_impl(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["task_id"]
    pair_id = task["pair_id"]
    partner_role = task["partner_role"]
    partner_name = task["partner_name"]
    binder_name = task["binder_name"]
    target_name = task["target_name"]

    result: Dict[str, Any] = {
        "status": "error",
        "task_id": task_id,
        "pair_id": pair_id,
        "row_index": task["row_index"],
        "partner_role": partner_role,
        "partner_name": partner_name,
        "binder_name": binder_name,
        "target_name": target_name,
        "binder_seq": task["binder_seq"],
        "target_seq": task["target_seq"],
        "selection_scope": task["best_sample_scope"],
        "error": None,
    }

    run_id = task.get("run_id")
    stream = bool(task.get("stream_to_dict", False))
    stream_logs = bool(task.get("stream_logs", True))
    log_chunk_seconds = max(0.1, float(task.get("log_chunk_seconds", 1.0)))
    heartbeat_seconds = max(1.0, float(task.get("heartbeat_seconds", 15.0)))
    event_seq = [0]
    pending_logs: Dict[str, List[str]] = {"stdout": [], "stderr": []}
    last_log_emit_time = time.time()

    work_dir = Path(tempfile.mkdtemp())

    def emit(event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        _emit_event_if_needed(
            run_id=run_id,
            task_id=task_id,
            stream=stream,
            event_seq=event_seq,
            event_type=event_type,
            payload=payload,
        )

    def flush_log_chunks(force: bool = False) -> None:
        nonlocal last_log_emit_time
        if not (stream and run_id and stream_logs):
            return
        now = time.time()
        if not force and (now - last_log_emit_time) < log_chunk_seconds:
            return
        for channel in ("stdout", "stderr"):
            if not pending_logs[channel]:
                continue
            text = "".join(pending_logs[channel])
            emit(
                "log_chunk",
                {
                    "channel": channel,
                    "line_count": len(pending_logs[channel]),
                    "text": text,
                },
            )
            pending_logs[channel] = []
        last_log_emit_time = now

    try:
        emit(
            "task_started",
            {
                "status": "running",
                "stage": "bootstrap",
                "pair_id": pair_id,
                "partner_role": partner_role,
                "binder_name": binder_name,
                "partner_name": partner_name,
            },
        )

        _ensure_protenix_runtime(task["model_name"], populate_missing=False)

        emit("msa_started", {"status": "running", "stage": "msa_prepare"})
        binder_msa = _load_msa_ref(task["binder_msa_ref"])
        partner_msa = _load_msa_ref(task["partner_msa_ref"])
        emit(
            "msa_done",
            {
                "status": "running",
                "stage": "msa_prepare",
                "binder_has_msa": bool(binder_msa.get("non_pairing")),
                "partner_has_msa": bool(partner_msa.get("non_pairing")),
            },
        )

        # If no target MSA is available for target role, hard fail.
        if partner_role == "target" and not partner_msa.get("non_pairing"):
            raise RuntimeError("Target MSA is required but missing for this task")

        sample_name = f"{task_id}_pred"
        input_dir = work_dir / "input"
        input_json, chain_map = _build_protenix_input_json(
            sample_name=sample_name,
            binder_seq=task["binder_seq"],
            partner_seq=task["partner_seq"],
            binder_msa=binder_msa,
            partner_msa=partner_msa,
            input_dir=input_dir,
        )

        out_dir = work_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        seeds = [int(x) for x in str(task["seeds_csv"]).split(",") if x.strip()]
        if not seeds:
            raise ValueError("No seeds provided after parsing --seeds")

        seen_candidates: set[Tuple[int, int]] = set()
        last_heartbeat_time = 0.0
        last_candidate_scan_time = 0.0

        def emit_new_candidates() -> None:
            for seed, rank, sf, full_data_path, cif_path in _iter_candidate_file_triples(
                output_dir=out_dir,
                sample_name=sample_name,
                seeds=seeds,
            ):
                candidate_id = (seed, rank)
                if candidate_id in seen_candidates:
                    continue
                try:
                    summary = _load_json(sf)
                    full_data = _load_json(full_data_path)
                    cif_text = cif_path.read_text()
                except Exception:  # noqa: BLE001
                    continue
                seen_candidates.add(candidate_id)
                emit(
                    "candidate_found",
                    {
                        "status": "running",
                        "stage": "inference",
                        "seed": seed,
                        "sample_rank": rank,
                        "iptm": float(summary.get("iptm", -1.0)),
                        "ptm": float(summary.get("ptm", -1.0)),
                        "ranking_score": float(summary.get("ranking_score", -1.0)),
                        "cif_text": cif_text,
                        "summary_confidence_json": summary,
                        "full_data_json": full_data,
                    },
                )

        def on_inference_line(channel: str, line: str) -> None:
            if stream and stream_logs:
                pending_logs[channel].append(line)
                flush_log_chunks(force=False)

        def on_inference_tick(elapsed: float) -> None:
            nonlocal last_heartbeat_time, last_candidate_scan_time
            now = time.time()
            if now - last_candidate_scan_time >= 1.0:
                emit_new_candidates()
                last_candidate_scan_time = now
            if now - last_heartbeat_time >= heartbeat_seconds:
                emit(
                    "heartbeat",
                    {
                        "status": "running",
                        "stage": "inference",
                        "elapsed_s": round(elapsed, 2),
                    },
                )
                last_heartbeat_time = now
            flush_log_chunks(force=False)

        inference_cmd = _build_protenix_inference_cmd(
            input_json=input_json,
            out_dir=out_dir,
            model_name=task["model_name"],
            seeds_csv=task["seeds_csv"],
            n_sample=int(task["n_sample"]),
            n_step=int(task["n_step"]),
            n_cycle=int(task["n_cycle"]),
            use_msa=bool(task["use_msa"]),
        )
        emit(
            "inference_started",
            {
                "status": "running",
                "stage": "inference",
                "command": inference_cmd,
            },
        )
        inference_result = _run_protenix_inference_streaming(
            cmd=inference_cmd,
            timeout_s=int(task["task_timeout_s"]),
            on_line=on_inference_line,
            on_tick=on_inference_tick,
        )
        flush_log_chunks(force=True)
        emit_new_candidates()

        if inference_result.get("timed_out"):
            raise RuntimeError(
                f"Protenix inference timed out after {task['task_timeout_s']}s. "
                f"stderr tail: {str(inference_result.get('stderr_tail', ''))[-1200:]}"
            )

        if int(inference_result.get("returncode", -1)) != 0:
            raise RuntimeError(
                f"Protenix failed (code {inference_result.get('returncode')}): "
                f"{str(inference_result.get('stderr_tail', ''))[-1200:]}"
            )

        candidates = _collect_candidates(
            output_dir=out_dir,
            sample_name=sample_name,
            seeds=seeds,
        )
        if not candidates:
            raise RuntimeError("No Protenix candidates found in output directory")

        best, per_seed_best = _select_best_candidate(
            candidates=candidates,
            scope=task["best_sample_scope"],
        )

        best_summary = best["summary"]
        best_cif_path = Path(best["cif_path"])
        best_full_data_path = Path(best["full_data_path"])
        best_summary_path = Path(best["summary_path"])

        best_cif = best_cif_path.read_text()
        best_full_data = _load_json(best_full_data_path)

        emit(
            "best_selected",
            {
                "status": "running",
                "stage": "selection",
                "seed": int(best["seed"]),
                "sample_rank": int(best["sample_rank"]),
                "best_iptm": float(best_summary.get("iptm", -1.0)),
                "best_ptm": float(best_summary.get("ptm", -1.0)),
                "best_ranking_score": float(best_summary.get("ranking_score", -1.0)),
                "cif_text": best_cif,
                "summary_confidence_json": best_summary,
                "full_data_json": best_full_data,
            },
        )

        candidate_manifest = [
            {
                "seed": int(c["seed"]),
                "sample_rank": int(c["sample_rank"]),
                "iptm": float(c["iptm"]),
                "ptm": float(c["ptm"]),
                "ranking_score": float(c["ranking_score"]),
                "summary": c["summary"],
            }
            for c in sorted(candidates, key=lambda x: (int(x["seed"]), int(x["sample_rank"])))
        ]

        ipsae_metrics: Dict[str, Any] = {}
        try:
            adapter_dir = work_dir / "ipsae_adapter"
            adapted_full_path, _ = _write_ipsae_adapter_files(
                full_data_path=best_full_data_path,
                summary_path=best_summary_path,
                out_dir=adapter_dir,
            )
            ipsae_metrics = _run_ipsae(
                adapted_full_path=adapted_full_path,
                cif_path=best_cif_path,
                pae_cutoff=float(task["pae_cutoff"]),
                dist_cutoff=float(task["dist_cutoff"]),
                binder_chains=chain_map["binder"],
                partner_chains=chain_map["partner"],
            )
        except Exception as exc:  # noqa: BLE001
            ipsae_metrics = {"error": str(exc)}

        result.update(
            {
                "status": "success",
                "best_seed": best["seed"],
                "best_sample_rank": best["sample_rank"],
                "best_iptm": float(best_summary.get("iptm", -1.0)),
                "best_ptm": float(best_summary.get("ptm", -1.0)),
                "best_ranking_score": float(best_summary.get("ranking_score", -1.0)),
                "best_summary": best_summary,
                "best_cif": best_cif,
                "n_candidates": len(candidates),
                "per_seed_best": [
                    {
                        "seed": c["seed"],
                        "sample_rank": c["sample_rank"],
                        "iptm": c["iptm"],
                        "ranking_score": c["ranking_score"],
                    }
                    for c in per_seed_best
                ],
                "ipsae": ipsae_metrics,
                "input_json": _load_json(input_json),
                "protenix_raw": {
                    "inference_stdout_tail": str(inference_result.get("stdout_tail", ""))[-20000:],
                    "inference_stderr_tail": str(inference_result.get("stderr_tail", ""))[-20000:],
                    "inference_elapsed_s": inference_result.get("elapsed_s"),
                    "candidate_summaries": candidate_manifest,
                },
            }
        )
        emit(
            "task_done",
            {
                "status": "success",
                "stage": "complete",
                "n_candidates": len(candidates),
                "best_seed": int(best["seed"]),
                "best_sample_rank": int(best["sample_rank"]),
                "best_iptm": float(best_summary.get("iptm", -1.0)),
            },
        )

    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        flush_log_chunks(force=True)
        emit(
            "task_error",
            {
                "status": "error",
                "stage": "error",
                "error": str(exc),
            },
        )
    finally:
        flush_log_chunks(force=True)
        _stream_result_if_needed(result=result, run_id=run_id, task_id=task_id, stream=stream)
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_T4(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_L4(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_A10G(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="L40S",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_L40S(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_A100_40GB(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_A100_80GB(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_H100(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="H200",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_H200(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


@app.function(
    image=image,
    gpu="B200",
    timeout=7200,
    max_containers=20,
    volumes={str(RUNTIME_ROOT): runtime_volume, str(MSA_CACHE_ROOT): msa_cache_volume},
)
def run_protenix_B200(task: Dict[str, Any]) -> Dict[str, Any]:
    return _run_protenix_task_impl(task)


GPU_FUNCTIONS = {
    "T4": run_protenix_T4,
    "L4": run_protenix_L4,
    "A10G": run_protenix_A10G,
    "L40S": run_protenix_L40S,
    "A100": run_protenix_A100_40GB,
    "A100-40GB": run_protenix_A100_40GB,
    "A100-80GB": run_protenix_A100_80GB,
    "H100": run_protenix_H100,
    "H200": run_protenix_H200,
    "B200": run_protenix_B200,
}


# =============================================================================
# OUTPUT + STREAMING
# =============================================================================

def _save_task_result(output_root: Path, result: Dict[str, Any]) -> None:
    pair_id = result["pair_id"]
    role = result["partner_role"]
    pair_dir = output_root / "pair_runs" / pair_id / role
    pair_dir.mkdir(parents=True, exist_ok=True)

    _save_json(pair_dir / "metrics.json", result)

    if result.get("input_json"):
        _save_json(pair_dir / "input.json", {"input": result["input_json"]})

    if result.get("status") != "success":
        return

    best_dir = pair_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "best_sample.cif").write_text(result.get("best_cif", ""))
    _save_json(best_dir / "best_summary_confidence.json", result.get("best_summary", {}))

    ipsae = result.get("ipsae", {})
    if ipsae:
        ipsae_dir = pair_dir / "ipsae"
        ipsae_dir.mkdir(parents=True, exist_ok=True)
        if ipsae.get("raw_text"):
            (ipsae_dir / "best_sample_ipsae.txt").write_text(ipsae.get("raw_text", ""))
        _save_json(ipsae_dir / "ipsae_metrics.json", ipsae)

    raw = result.get("protenix_raw", {}) or {}
    if raw:
        raw_dir = pair_dir / "protenix_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        if raw.get("inference_stdout_tail"):
            (raw_dir / "inference_stdout_tail.log").write_text(raw["inference_stdout_tail"])
        if raw.get("inference_stderr_tail"):
            (raw_dir / "inference_stderr_tail.log").write_text(raw["inference_stderr_tail"])

        candidate_summaries = raw.get("candidate_summaries", []) or []
        manifest_rows: List[Dict[str, Any]] = []
        for item in candidate_summaries:
            seed = int(item.get("seed", -1))
            sample_rank = int(item.get("sample_rank", -1))
            seed_dir = raw_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            summary = item.get("summary")
            if isinstance(summary, dict):
                summary_path = seed_dir / f"summary_confidence_sample_{sample_rank}.json"
                _save_json(summary_path, summary)

            manifest_rows.append(
                {
                    "seed": seed,
                    "sample_rank": sample_rank,
                    "iptm": item.get("iptm"),
                    "ptm": item.get("ptm"),
                    "ranking_score": item.get("ranking_score"),
                }
            )

        _save_json(
            raw_dir / "manifest.json",
            {
                "n_candidates": len(candidate_summaries),
                "candidates": manifest_rows,
            },
        )

    # Only target role contributes to best_by_target grouping.
    if role == "target":
        binder_slug = _slugify(result.get("binder_name", "binder"))
        target_slug = _slugify(result.get("target_name", "target"))
        pair_hash = _short_hash(
            f"{result.get('binder_name','')}|{result.get('target_name','')}|{result.get('row_index',0)}"
        )
        out_name = f"{binder_slug}__vs__{target_slug}__{pair_hash}.cif"
        grouped_dir = output_root / "best_by_target" / target_slug
        grouped_dir.mkdir(parents=True, exist_ok=True)
        (grouped_dir / out_name).write_text(result.get("best_cif", ""))


def _write_summary_csv(output_root: Path, results: Sequence[Dict[str, Any]]) -> Path:
    out_dir = output_root / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pair_summary.csv"

    header = [
        "task_id",
        "pair_id",
        "row_index",
        "partner_role",
        "partner_name",
        "binder_name",
        "binder_seq",
        "target_name",
        "target_seq",
        "status",
        "best_sample_scope",
        "best_seed",
        "best_sample_rank",
        "best_iptm",
        "best_ptm",
        "best_ranking_score",
        "ipsae",
        "ipsae_d0chn",
        "ipsae_d0dom",
        "ipsae_error",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for r in results:
            ipsae = r.get("ipsae", {}) or {}
            writer.writerow(
                [
                    r.get("task_id"),
                    r.get("pair_id"),
                    r.get("row_index"),
                    r.get("partner_role"),
                    r.get("partner_name"),
                    r.get("binder_name"),
                    r.get("binder_seq"),
                    r.get("target_name"),
                    r.get("target_seq"),
                    r.get("status"),
                    r.get("selection_scope"),
                    r.get("best_seed"),
                    r.get("best_sample_rank"),
                    r.get("best_iptm"),
                    r.get("best_ptm"),
                    r.get("best_ranking_score"),
                    ipsae.get("ipSAE"),
                    ipsae.get("ipSAE_d0chn"),
                    ipsae.get("ipSAE_d0dom"),
                    ipsae.get("error"),
                    r.get("error"),
                ]
            )

    return csv_path


def _sync_worker(
    run_id: str,
    output_dir: Path,
    stop_event: threading.Event,
    interval: float = 5.0,
) -> None:
    synced = set()
    chunk_state: Dict[Tuple[str, int], Dict[str, Any]] = {}
    while not stop_event.is_set():
        try:
            keys = sorted(k for k in results_dict.keys() if k.startswith(f"{run_id}:"))
            for key in keys:
                if key in synced:
                    continue
                payload = results_dict[key]
                if ":event:" in key:
                    _save_stream_event(
                        output_root=output_dir,
                        run_id=run_id,
                        dict_key=key,
                        event=payload,
                        chunk_state=chunk_state,
                    )
                else:
                    _save_task_result(output_dir, payload)
                synced.add(key)
        except Exception:  # noqa: BLE001
            pass
        stop_event.wait(timeout=interval)

    # Final flush.
    try:
        keys = sorted(k for k in results_dict.keys() if k.startswith(f"{run_id}:"))
        for key in keys:
            if key in synced:
                continue
            payload = results_dict[key]
            if ":event:" in key:
                _save_stream_event(
                    output_root=output_dir,
                    run_id=run_id,
                    dict_key=key,
                    event=payload,
                    chunk_state=chunk_state,
                )
            else:
                _save_task_result(output_dir, payload)
            synced.add(key)
    except Exception:  # noqa: BLE001
        pass


# =============================================================================
# LOCAL ENTRYPOINTS
# =============================================================================

def _warn_fixed_msa_compatibility(
    fixed_msa: Dict[str, Optional[str]],
    binder_seq: str,
    binder_name: str,
    threshold: float = 0.90,
) -> None:
    non_pairing = fixed_msa.get("non_pairing") or ""
    query = _extract_query_from_a3m(non_pairing)
    qn = _normalize_for_identity(query)
    bn = _normalize_for_identity(binder_seq)
    if not qn or not bn:
        print(f"[WARN] Fixed MSA compatibility skipped for {binder_name}: could not parse query sequence")
        return

    if len(qn) != len(bn):
        print(
            f"[WARN] Fixed MSA query length {len(qn)} != binder length {len(bn)} for {binder_name}; continuing"
        )
        return

    ident = _identity(qn, bn)
    if ident < threshold:
        print(
            f"[WARN] Fixed MSA query identity {ident:.3f} < {threshold:.2f} for {binder_name}; continuing"
        )


@app.local_entrypoint()
def run_pipeline(
    pair_csv: str,
    output_dir: str = "./results_protenix",
    # Binder/target MSA controls
    binder_mode: str = "de_novo",  # de_novo | fixed_msa | full_msa
    binder_fixed_msa_dir: Optional[str] = None,
    target_msa_source: str = "mmseqs",  # provided | mmseqs
    target_msa_map_csv: Optional[str] = None,
    # Optional extra partner controls
    include_antitarget: bool = False,
    antitarget_csv: Optional[str] = None,
    antitarget_name: Optional[str] = None,
    antitarget_seq: Optional[str] = None,
    antitarget_msa_source: str = "mmseqs",  # provided | mmseqs | none
    antitarget_msa_dir: Optional[str] = None,
    include_self: bool = False,
    # MSA fetch + cache
    mmseqs_mode: str = "colabfold",
    mmseqs_host_url: Optional[str] = None,
    mmseqs_host_policy: str = "strict",  # strict | allow-default
    mmseqs_cache_dir: str = "/msa_cache",
    mmseqs_cache_mode: str = "readwrite",  # none | read | write | readwrite
    mmseqs_store_fetched_msas: bool = False,
    mmseqs_pairing_strategy: str = "greedy",
    mmseqs_db_tag: str = "colabfold_env",
    # Inference controls
    model_name: str = "protenix_base_default_v1.0.0",
    seeds: str = "101",
    sample_diffusion_n_sample: int = 5,
    sample_diffusion_n_step: int = 200,
    n_cycle: int = 10,
    best_sample_scope: str = "global",  # global | per_seed
    task_timeout_s: int = 7200,
    # ipSAE controls
    pae_cutoff: float = 15.0,
    dist_cutoff: float = 15.0,
    # Runtime controls
    gpu: str = DEFAULT_GPU,
    max_parallel: int = 1,
    # Streaming
    no_stream: bool = False,
    stream_logs: bool = True,
    log_chunk_seconds: float = 1.0,
    heartbeat_seconds: float = 15.0,
    run_id: Optional[str] = None,
    sync_interval: float = 5.0,
):
    stream = not _coerce_bool(no_stream)

    if gpu not in GPU_FUNCTIONS:
        raise ValueError(f"Unknown GPU type '{gpu}'. Available: {sorted(GPU_FUNCTIONS.keys())}")

    include_antitarget = _coerce_bool(include_antitarget)
    include_self = _coerce_bool(include_self)
    mmseqs_store_fetched_msas = _coerce_bool(mmseqs_store_fetched_msas)
    stream_logs = _coerce_bool(stream_logs)
    log_chunk_seconds = max(0.1, float(log_chunk_seconds))
    heartbeat_seconds = max(1.0, float(heartbeat_seconds))

    if mmseqs_cache_dir != str(MSA_CACHE_ROOT):
        print(
            f"[WARN] --mmseqs-cache-dir is currently fixed to {MSA_CACHE_ROOT} in Modal workers; "
            f"received '{mmseqs_cache_dir}'. Using {MSA_CACHE_ROOT}."
        )

    binder_mode_n = binder_mode.strip().lower()
    if binder_mode_n not in {"de_novo", "fixed_msa", "full_msa"}:
        raise ValueError("--binder-mode must be one of: de_novo,fixed_msa,full_msa")

    target_msa_source_n = target_msa_source.strip().lower()
    if target_msa_source_n not in {"provided", "mmseqs"}:
        raise ValueError("--target-msa-source must be one of: provided,mmseqs")

    antitarget_msa_source_n = antitarget_msa_source.strip().lower()
    if antitarget_msa_source_n not in {"provided", "mmseqs", "none"}:
        raise ValueError("--antitarget-msa-source must be one of: provided,mmseqs,none")

    scope_n = best_sample_scope.strip().lower()
    if scope_n not in {"global", "per_seed"}:
        raise ValueError("--best-sample-scope must be one of: global,per_seed")

    if mmseqs_mode.strip().lower() != "colabfold":
        raise ValueError("Only --mmseqs-mode colabfold is supported in this pipeline")

    host_url = _resolve_mmseqs_host(mmseqs_host_url, mmseqs_host_policy)

    pair_rows = _load_pair_rows(Path(pair_csv))
    if not pair_rows:
        raise ValueError("No valid rows found in pair CSV")

    for r in pair_rows:
        if not _sequence_is_protein_like(r["binder_seq"]):
            print(f"[WARN] Binder sequence for row {r['row_index']} contains non-standard characters")
        if not _sequence_is_protein_like(r["target_seq"]):
            print(f"[WARN] Target sequence for row {r['row_index']} contains non-standard characters")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Prepare fixed/provided MSA sources.
    fixed_binder_msa: Optional[Dict[str, Optional[str]]] = None
    if binder_mode_n == "fixed_msa":
        if not binder_fixed_msa_dir:
            raise ValueError("--binder-fixed-msa-dir is required for --binder-mode fixed_msa")
        fixed_binder_msa = _read_msa_pair(msa_dir=Path(binder_fixed_msa_dir))
        if not fixed_binder_msa.get("non_pairing"):
            raise ValueError("Fixed binder MSA must provide non_pairing.a3m")
        for row in pair_rows:
            _warn_fixed_msa_compatibility(
                fixed_msa=fixed_binder_msa,
                binder_seq=row["binder_seq"],
                binder_name=row["binder_name"],
            )

    target_msa_map: Dict[str, Dict[str, Optional[str]]] = {}
    if target_msa_source_n == "provided":
        if not target_msa_map_csv:
            raise ValueError("--target-msa-map-csv is required when --target-msa-source provided")
        target_msa_map = _parse_target_msa_map_csv(Path(target_msa_map_csv))

    antitarget_msa_inline: Optional[Dict[str, Optional[str]]] = None
    antitarget_input_mode = "none"
    if include_antitarget:
        if antitarget_csv:
            csv_name, csv_seq = _load_single_name_sequence_csv(Path(antitarget_csv), "antitarget")
            if antitarget_name or antitarget_seq:
                print(
                    "[WARN] Both --antitarget-csv and --antitarget-name/--antitarget-seq were provided; "
                    "using --antitarget-csv."
                )
            antitarget_name, antitarget_seq = csv_name, csv_seq
            antitarget_input_mode = "csv"
        else:
            antitarget_input_mode = "direct"

        if not (antitarget_name and antitarget_seq):
            raise ValueError(
                "--include-antitarget requires either --antitarget-csv or "
                "--antitarget-name/--antitarget-seq"
            )
        antitarget_seq = _normalize_sequence(antitarget_seq)
        antitarget_name = _sanitize_name(antitarget_name)
        if antitarget_msa_source_n == "provided":
            if not antitarget_msa_dir:
                raise ValueError("--antitarget-msa-dir is required for antitarget_msa_source=provided")
            antitarget_msa_inline = _read_msa_pair(msa_dir=Path(antitarget_msa_dir))
            if not antitarget_msa_inline.get("non_pairing"):
                raise ValueError("Provided antitarget MSA must include non_pairing.a3m")

    # Precompute dynamic MSAs (deduped) when needed.
    binder_cache: Dict[str, Dict[str, Any]] = {}
    if binder_mode_n == "full_msa":
        binder_seqs = sorted({r["binder_seq"] for r in pair_rows})
        print(f"Precomputing binder MSAs for {len(binder_seqs)} unique sequence(s)...")
        binder_cache = precompute_msas.remote(
            sequences=binder_seqs,
            role="binder",
            host_url=host_url,
            cache_mode=mmseqs_cache_mode,
            store_fetched_msas=mmseqs_store_fetched_msas,
            msa_mode="colabfold",
            pairing_strategy=mmseqs_pairing_strategy,
            db_tag=mmseqs_db_tag,
        )

    target_cache: Dict[str, Dict[str, Any]] = {}
    if target_msa_source_n == "mmseqs":
        target_seqs = sorted({r["target_seq"] for r in pair_rows})
        print(f"Precomputing target MSAs for {len(target_seqs)} unique sequence(s)...")
        target_cache = precompute_msas.remote(
            sequences=target_seqs,
            role="target",
            host_url=host_url,
            cache_mode=mmseqs_cache_mode,
            store_fetched_msas=mmseqs_store_fetched_msas,
            msa_mode="colabfold",
            pairing_strategy=mmseqs_pairing_strategy,
            db_tag=mmseqs_db_tag,
        )

    antitarget_cache: Dict[str, Dict[str, Any]] = {}
    if include_antitarget and antitarget_msa_source_n == "mmseqs":
        antitarget_cache = precompute_msas.remote(
            sequences=[antitarget_seq],
            role="antitarget",
            host_url=host_url,
            cache_mode=mmseqs_cache_mode,
            store_fetched_msas=mmseqs_store_fetched_msas,
            msa_mode="colabfold",
            pairing_strategy=mmseqs_pairing_strategy,
            db_tag=mmseqs_db_tag,
        )

    def binder_msa_ref_for(seq: str) -> Dict[str, Any]:
        if binder_mode_n == "de_novo":
            return {"source": "none"}
        if binder_mode_n == "fixed_msa":
            return {
                "source": "inline",
                "pairing": fixed_binder_msa.get("pairing") if fixed_binder_msa else None,
                "non_pairing": fixed_binder_msa.get("non_pairing") if fixed_binder_msa else None,
            }
        # full_msa
        info = binder_cache.get(seq, {})
        if info.get("status") == "error":
            raise RuntimeError(f"Binder MSA fetch failed for sequence hash={_short_hash(seq)}: {info.get('error')}")
        if info.get("status") == "fetched_no_cache":
            if not info.get("non_pairing"):
                raise RuntimeError("Binder MSA fetch returned no non_pairing content")
            return {
                "source": "inline",
                "pairing": info.get("pairing"),
                "non_pairing": info.get("non_pairing"),
            }
        ck = info.get("cache_key")
        if not ck:
            raise RuntimeError("Binder MSA cache entry missing cache_key")
        return {"source": "cache", "cache_key": ck}

    def target_msa_ref_for(name: str, seq: str) -> Dict[str, Any]:
        if target_msa_source_n == "provided":
            pair = _target_msa_lookup(target_msa_map, name, seq)
            if not pair or not pair.get("non_pairing"):
                raise RuntimeError(f"No provided target MSA found for target '{name}'")
            return {
                "source": "inline",
                "pairing": pair.get("pairing"),
                "non_pairing": pair.get("non_pairing"),
            }

        info = target_cache.get(seq, {})
        if info.get("status") == "error":
            raise RuntimeError(f"Target MSA fetch failed for target '{name}': {info.get('error')}")
        if info.get("status") == "fetched_no_cache":
            if not info.get("non_pairing"):
                raise RuntimeError(f"Target MSA fetch returned no non_pairing content for '{name}'")
            return {
                "source": "inline",
                "pairing": info.get("pairing"),
                "non_pairing": info.get("non_pairing"),
            }
        ck = info.get("cache_key")
        if not ck:
            raise RuntimeError(f"Target MSA cache key missing for target '{name}'")
        return {"source": "cache", "cache_key": ck}

    def antitarget_msa_ref() -> Dict[str, Any]:
        if antitarget_msa_source_n == "none":
            return {"source": "none"}
        if antitarget_msa_source_n == "provided":
            return {
                "source": "inline",
                "pairing": antitarget_msa_inline.get("pairing") if antitarget_msa_inline else None,
                "non_pairing": antitarget_msa_inline.get("non_pairing") if antitarget_msa_inline else None,
            }
        info = antitarget_cache.get(antitarget_seq, {})
        if info.get("status") == "error":
            raise RuntimeError(f"Antitarget MSA fetch failed: {info.get('error')}")
        if info.get("status") == "fetched_no_cache":
            if not info.get("non_pairing"):
                raise RuntimeError("Antitarget MSA fetch returned no non_pairing content")
            return {
                "source": "inline",
                "pairing": info.get("pairing"),
                "non_pairing": info.get("non_pairing"),
            }
        ck = info.get("cache_key")
        if not ck:
            raise RuntimeError("Antitarget MSA cache key missing")
        return {"source": "cache", "cache_key": ck}

    # Build tasks.
    task_list: List[Dict[str, Any]] = []
    for row in pair_rows:
        binder_ref = binder_msa_ref_for(row["binder_seq"])
        target_ref = target_msa_ref_for(row["target_name"], row["target_seq"])

        base_hash = _short_hash(f"{row['binder_name']}|{row['target_name']}|{row['row_index']}")
        pair_id = (
            f"r{int(row['row_index']):05d}_"
            f"{_slugify(row['binder_name'])}__vs__{_slugify(row['target_name'])}__{base_hash}"
        )

        def add_task(role: str, partner_name: str, partner_seq: str, partner_ref: Dict[str, Any]) -> None:
            task_id = f"{pair_id}__{role}"
            task_list.append(
                {
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "row_index": row["row_index"],
                    "partner_role": role,
                    "partner_name": _sanitize_name(partner_name),
                    "partner_seq": _normalize_sequence(partner_seq),
                    "binder_name": row["binder_name"],
                    "binder_seq": row["binder_seq"],
                    "target_name": row["target_name"],
                    "target_seq": row["target_seq"],
                    "binder_msa_ref": binder_ref,
                    "partner_msa_ref": partner_ref,
                    "model_name": model_name,
                    "seeds_csv": seeds,
                    "n_sample": sample_diffusion_n_sample,
                    "n_step": sample_diffusion_n_step,
                    "n_cycle": n_cycle,
                    "best_sample_scope": scope_n,
                    "use_msa": True,
                    "task_timeout_s": task_timeout_s,
                    "pae_cutoff": pae_cutoff,
                    "dist_cutoff": dist_cutoff,
                    "run_id": run_id,
                    "stream_to_dict": stream,
                    "stream_logs": stream_logs,
                    "log_chunk_seconds": log_chunk_seconds,
                    "heartbeat_seconds": heartbeat_seconds,
                }
            )

        add_task("target", row["target_name"], row["target_seq"], target_ref)

        if include_antitarget:
            add_task(
                "antitarget",
                antitarget_name,
                antitarget_seq,
                antitarget_msa_ref(),
            )

        if include_self:
            add_task("self", row["binder_name"], row["binder_seq"], {"source": "none"})

    effective_run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if stream:
        for task in task_list:
            task["run_id"] = effective_run_id

    print("\n" + "=" * 78)
    print("PROTENIX + ipSAE PIPELINE (Modal)")
    print("=" * 78)
    print(f"Pair rows: {len(pair_rows)}")
    print(f"Tasks: {len(task_list)}")
    print(f"Binder mode: {binder_mode_n}")
    print(f"Target MSA source: {target_msa_source_n}")
    print(f"MSA host: {host_url}")
    print(f"MSA host policy: {mmseqs_host_policy}")
    print(f"MSA cache mode: {mmseqs_cache_mode} (store_alias={mmseqs_store_fetched_msas})")
    print(f"Model: {model_name}")
    print(f"Seeds: {seeds}")
    print(f"N_sample: {sample_diffusion_n_sample}")
    print(f"Best scope: {scope_n}")
    print(f"GPU: {gpu}")
    print(f"Max parallel: {max_parallel}")
    print(f"Output dir: {output_root}")
    if stream:
        print(f"Streaming: ENABLED (run_id={effective_run_id})")
        print(
            f"Stream logs: {'ENABLED' if stream_logs else 'DISABLED'} "
            f"(chunk={log_chunk_seconds:.1f}s, heartbeat={heartbeat_seconds:.1f}s)"
        )
    else:
        print("Streaming: DISABLED")
    print("=" * 78 + "\n")

    print("Running dependency/runtime preflight...")
    preflight = preflight_protenix_runtime.remote(model_name)
    print(f"Preflight: {preflight}")

    configured_fn = GPU_FUNCTIONS[gpu]

    sync_thread = None
    stop_sync = None
    if stream:
        stop_sync = threading.Event()
        sync_thread = threading.Thread(
            target=_sync_worker,
            args=(effective_run_id, output_root, stop_sync, sync_interval),
            daemon=True,
        )
        sync_thread.start()

    results: List[Dict[str, Any]] = []

    def run_safe(t: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return configured_fn.remote(t)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "task_id": t["task_id"],
                "pair_id": t["pair_id"],
                "row_index": t["row_index"],
                "partner_role": t["partner_role"],
                "partner_name": t["partner_name"],
                "binder_name": t["binder_name"],
                "binder_seq": t["binder_seq"],
                "target_name": t["target_name"],
                "target_seq": t["target_seq"],
                "selection_scope": t["best_sample_scope"],
                "error": str(exc),
            }

    print(f"Submitting {len(task_list)} task(s)...")
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(run_safe, t): t for t in task_list}
        for i, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            results.append(res)
            status = res.get("status", "error")
            task_id = res.get("task_id", "unknown")
            if status == "success":
                print(f"[{i}/{len(task_list)}] ✓ {task_id} ipTM={res.get('best_iptm', -1):.4f}")
            else:
                print(f"[{i}/{len(task_list)}] ✗ {task_id} {str(res.get('error', 'error'))[:120]}")

    if sync_thread is not None and stop_sync is not None:
        stop_sync.set()
        sync_thread.join(timeout=30)

    # Persist all results (also covers no-stream mode).
    for r in results:
        _save_task_result(output_root, r)

    csv_path = _write_summary_csv(output_root, results)

    metadata = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "run_id": effective_run_id,
        "pair_csv": str(pair_csv),
        "num_rows": len(pair_rows),
        "num_tasks": len(task_list),
        "binder_mode": binder_mode_n,
        "target_msa_source": target_msa_source_n,
        "include_antitarget": bool(include_antitarget),
        "antitarget_input_mode": antitarget_input_mode,
        "antitarget_csv": str(antitarget_csv) if antitarget_csv else None,
        "include_self": bool(include_self),
        "mmseqs_mode": "colabfold",
        "mmseqs_host_url": host_url,
        "mmseqs_host_policy": mmseqs_host_policy,
        "mmseqs_cache_mode": mmseqs_cache_mode,
        "mmseqs_store_fetched_msas": bool(mmseqs_store_fetched_msas),
        "model_name": model_name,
        "seeds": seeds,
        "sample_diffusion_n_sample": sample_diffusion_n_sample,
        "sample_diffusion_n_step": sample_diffusion_n_step,
        "n_cycle": n_cycle,
        "best_sample_scope": scope_n,
        "gpu": gpu,
        "max_parallel": max_parallel,
        "stream_logs": bool(stream_logs),
        "log_chunk_seconds": float(log_chunk_seconds),
        "heartbeat_seconds": float(heartbeat_seconds),
    }

    meta_path = output_root / "summaries" / "run_metadata.json"
    _save_json(meta_path, metadata)

    n_success = sum(1 for r in results if r.get("status") == "success")
    n_fail = len(results) - n_success

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Success: {n_success}/{len(results)}")
    print(f"Failed : {n_fail}/{len(results)}")
    print(f"Summary CSV: {csv_path}")
    print(f"Run metadata: {meta_path}")


@app.local_entrypoint()
def init_protenix_runtime(model_name: str = "protenix_base_default_v1.0.0"):
    print(f"Initializing Protenix runtime cache for model: {model_name}")
    msg = _init_runtime_remote.remote(model_name)
    print(msg)


@app.function(
    image=image,
    timeout=3600,
    volumes={str(RUNTIME_ROOT): runtime_volume},
)
def _init_runtime_remote(model_name: str) -> str:
    _ensure_protenix_runtime(model_name, populate_missing=True)
    files = list(RUNTIME_ROOT.rglob("*"))
    total_size = sum(p.stat().st_size for p in files if p.is_file())
    return (
        f"Runtime initialized for {model_name}. "
        f"Files: {len(files)}, Size: {total_size / 1e9:.2f} GB"
    )


@app.local_entrypoint()
def list_gpus():
    print("\nSupported GPU Types")
    print("=" * 60)
    for gpu_type, desc in GPU_TYPES.items():
        default = " (DEFAULT)" if gpu_type == DEFAULT_GPU else ""
        print(f"  {gpu_type:12s} - {desc}{default}")


def _gpu_probe_output() -> str:
    proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else proc.stderr


@app.function(image=image, gpu="T4", timeout=120)
def _test_gpu_T4() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="L4", timeout=120)
def _test_gpu_L4() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="A10G", timeout=120)
def _test_gpu_A10G() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="L40S", timeout=120)
def _test_gpu_L40S() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="A100", timeout=120)
def _test_gpu_A100_40GB() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="A100-80GB", timeout=120)
def _test_gpu_A100_80GB() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="H100", timeout=120)
def _test_gpu_H100() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="H200", timeout=120)
def _test_gpu_H200() -> str:
    return _gpu_probe_output()


@app.function(image=image, gpu="B200", timeout=120)
def _test_gpu_B200() -> str:
    return _gpu_probe_output()


GPU_TEST_FUNCTIONS = {
    "T4": _test_gpu_T4,
    "L4": _test_gpu_L4,
    "A10G": _test_gpu_A10G,
    "L40S": _test_gpu_L40S,
    "A100": _test_gpu_A100_40GB,
    "A100-40GB": _test_gpu_A100_40GB,
    "A100-80GB": _test_gpu_A100_80GB,
    "H100": _test_gpu_H100,
    "H200": _test_gpu_H200,
    "B200": _test_gpu_B200,
}


@app.local_entrypoint()
def test_connection(gpu: str = DEFAULT_GPU):
    if gpu not in GPU_TEST_FUNCTIONS:
        raise ValueError(f"Unknown GPU '{gpu}'. Available: {sorted(GPU_TEST_FUNCTIONS.keys())}")
    print(f"Testing Modal GPU connection ({gpu})...")
    print(GPU_TEST_FUNCTIONS[gpu].remote())
