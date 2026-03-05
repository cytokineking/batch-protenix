#!/usr/bin/env python3
"""
Standalone Modal bootstrap utility for a UniRef100 MMseqs database volume.

This script is intentionally separate from the main pipeline so database
bootstrap can be run independently and pointed at an alternate volume.
"""

from __future__ import annotations

import datetime
import json
import os
import queue
import threading
import time
from typing import Optional

import modal


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


DEFAULT_VOLUME_NAME = "mmseqs-uniref100-db"
DEFAULT_TMP_VOLUME_NAME = "mmseqs-tmp"
DEFAULT_VOLUME_VERSION = _env_int("MMSEQS_DB_VOLUME_VERSION", 1)
DEFAULT_TMP_VOLUME_VERSION = _env_int("MMSEQS_TMP_VOLUME_VERSION", DEFAULT_VOLUME_VERSION)
DEFAULT_DB_PROFILE = "uniref100_only"
DEFAULT_UNIREF100_FASTA_URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz"
DEFAULT_MMSEQS_PREBUILT_URL = "https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"

APP_NAME = "mmseqs-uniref100-bootstrap"
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("bash", "coreutils", "findutils", "tar", "wget", "ca-certificates")
)


def _build_bootstrap_script(
    *,
    db_profile: str,
    uniref_fasta_url: str,
    mmseqs_prebuilt_url: str,
    mmseqs_threads: int,
    force_rebuild: bool,
    allow_fallback_download: bool,
    heartbeat_seconds: int,
) -> str:
    force_flag = "1" if force_rebuild else "0"
    fallback_flag = "1" if allow_fallback_download else "0"
    hb_s = max(10, int(heartbeat_seconds))
    now = datetime.datetime.utcnow().isoformat() + "Z"
    return f"""\
set -euo pipefail

log() {{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}}

STATUS_FILE="/mmseqs_db/bootstrap_status.txt"
CURRENT_STEP="init"

update_status() {{
  local state="$1"
  local detail="$2"
  printf "%s state=%s step=%s detail=%s\\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$CURRENT_STEP" "$detail" > "$STATUS_FILE"
}}

on_error() {{
  local rc=$?
  update_status "error" "exit_code=${{rc}}"
  log "bootstrap failed at step=$CURRENT_STEP rc=$rc"
  exit "$rc"
}}
trap on_error ERR

run_with_heartbeat() {{
  local step="$1"
  shift
  CURRENT_STEP="$step"
  update_status "running" "started"
  local started
  started=$(date +%s)
  "$@" &
  local pid=$!
  while kill -0 "$pid" >/dev/null 2>&1; do
    local now elapsed db_sz tmp_sz fasta_sz
    now=$(date +%s)
    elapsed=$((now - started))
    db_sz=$(du -sh "$DB_ROOT" 2>/dev/null | awk '{{print $1}}' || true)
    tmp_sz=$(du -sh "$TMP_ROOT" 2>/dev/null | awk '{{print $1}}' || true)
    fasta_sz=$(stat -c %s "$FASTA_GZ" 2>/dev/null || echo 0)
    log "[heartbeat] step=$step elapsed_s=$elapsed db_size=${{db_sz:-0}} tmp_size=${{tmp_sz:-0}} fasta_bytes=$fasta_sz"
    sleep {hb_s}
  done
  wait "$pid"
}}

recover_split_target_db() {{
  CURRENT_STEP="recover_split_db"
  update_status "running" "started"
  local shard_data shard_idx
  shard_data=$(find "$DB_ROOT" -maxdepth 1 -type f -name 'target_db.[0-9]*' | wc -l | tr -d ' ')
  shard_idx=$(find "$DB_ROOT" -maxdepth 1 -type f -name 'target_db.index.[0-9]*' | wc -l | tr -d ' ')
  if [[ "$shard_data" -eq 0 || "$shard_idx" -eq 0 ]]; then
    log "split-db recovery skipped (data_shards=$shard_data index_shards=$shard_idx)"
    return 1
  fi
  log "attempting split-db recovery via no-scratch k-way merge (data_shards=$shard_data index_shards=$shard_idx)"
  python3 - "$DB_ROOT" <<'PY'
import heapq
import os
import shutil
import sys
from pathlib import Path

db_root = Path(sys.argv[1])
target = db_root / "target_db"
target_index = db_root / "target_db.index"
tmp_data = db_root / "target_db.rebuild"
tmp_index = db_root / "target_db.index.rebuild"

data_shards = []
for p in db_root.glob("target_db.[0-9]*"):
    n = p.name
    if ".index." in n:
        continue
    data_shards.append(p)
data_shards.sort(key=lambda p: int(p.name.split(".")[-1]))

index_shards = sorted(
    db_root.glob("target_db.index.[0-9]*"),
    key=lambda p: int(p.name.split(".")[-1]),
)

if not data_shards or not index_shards:
    raise SystemExit("no split shards found for recovery")

if len(data_shards) != len(index_shards):
    raise SystemExit(
        "data/index split count mismatch: %d vs %d"
        % (len(data_shards), len(index_shards))
    )

prefix = dict()
offset = 0
with open(tmp_data, "wb") as out_data:
    for p in data_shards:
        sid = int(p.name.split(".")[-1])
        prefix[sid] = offset
        with open(p, "rb") as in_data:
            shutil.copyfileobj(in_data, out_data, length=16 * 1024 * 1024)
        offset += p.stat().st_size

idx_files = dict()
heap = []
for p in index_shards:
    sid = int(p.name.split(".")[-1])
    fh = open(p, "r", encoding="utf-8", buffering=1024 * 1024)
    idx_files[sid] = fh
    line = fh.readline()
    if not line:
        continue
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        raise SystemExit("malformed index line in %s" % str(p))
    key = int(parts[0])
    off = int(parts[1]) + int(prefix[sid])
    size = parts[2]
    heapq.heappush(heap, (key, sid, off, size))

last_key = -1
count = 0
with open(tmp_index, "w", encoding="utf-8", buffering=4 * 1024 * 1024) as out_idx:
    while heap:
        key, sid, off, size = heapq.heappop(heap)
        if key < last_key:
            raise SystemExit("non-monotonic key %d after %d" % (key, last_key))
        out_idx.write("%d\t%d\t%s\n" % (key, off, size))
        last_key = key
        count += 1

        line = idx_files[sid].readline()
        if line:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise SystemExit("malformed index line in shard %d" % sid)
            nkey = int(parts[0])
            noff = int(parts[1]) + int(prefix[sid])
            nsize = parts[2]
            heapq.heappush(heap, (nkey, sid, noff, nsize))

for fh in idx_files.values():
    fh.close()

os.replace(tmp_data, target)
os.replace(tmp_index, target_index)

dbtype = db_root / "target_db.dbtype"
if not dbtype.exists():
    dbtype.write_bytes(bytes([0, 0, 0, 0]))

for p in data_shards + index_shards:
    try:
        p.unlink()
    except OSError:
        pass

print("recovered split DB entries=%d target_bytes=%d index_bytes=%d" % (
    count,
    target.stat().st_size if target.exists() else -1,
    target_index.stat().st_size if target_index.exists() else -1,
), flush=True)
PY
}}

DB_PROFILE="{db_profile}"
DB_ROOT="/mmseqs_db/${{DB_PROFILE}}"
TOOLS_DIR="/mmseqs_db/tools"
TMP_ROOT="/mmseqs_tmp/bootstrap_${{DB_PROFILE}}"
MMSEQS_BIN="$TOOLS_DIR/mmseqs"
MMSEQS_THREADS="{int(mmseqs_threads)}"
FASTA_GZ="$TMP_ROOT/uniref100.fasta.gz"
MANIFEST_JSON="/mmseqs_db/db_manifest.json"
FORCE_REBUILD="{force_flag}"
ALLOW_FALLBACK_DOWNLOAD="{fallback_flag}"
UNIREF_FASTA_URL="{uniref_fasta_url}"
MMSEQS_PREBUILT_URL="{mmseqs_prebuilt_url}"
export MMSEQS_FORCE_MERGE=1
export MMSEQS_NUM_THREADS="$MMSEQS_THREADS"
export OMP_NUM_THREADS="$MMSEQS_THREADS"

mkdir -p "$DB_ROOT" "$TOOLS_DIR" "$TMP_ROOT"
update_status "running" "bootstrap_started"

if [[ "$FORCE_REBUILD" == "1" ]]; then
  log "force rebuild enabled, clearing prior DB artifacts under $DB_ROOT"
  rm -rf "$DB_ROOT/target_db"* "$DB_ROOT/target_db_padded"* || true
fi

if [[ ! -x "$MMSEQS_BIN" ]]; then
  log "downloading MMseqs prebuilt from $MMSEQS_PREBUILT_URL"
  run_with_heartbeat "download_mmseqs" wget -c --progress=dot:giga -O "$TMP_ROOT/mmseqs-linux-gpu.tar.gz" "$MMSEQS_PREBUILT_URL"
  log "extracting MMseqs prebuilt"
  tar xzf "$TMP_ROOT/mmseqs-linux-gpu.tar.gz" -C "$TMP_ROOT"
  install -m 0755 "$TMP_ROOT/mmseqs/bin/mmseqs" "$MMSEQS_BIN"
fi

# With `set -o pipefail`, `mmseqs --version | head -n 1` can fail due to SIGPIPE.
MMSEQS_VERSION_LINE=$("$MMSEQS_BIN" --version 2>/dev/null | head -n 1 || true)
log "mmseqs version: ${{MMSEQS_VERSION_LINE}}"
log "mmseqs thread cap: MMSEQS_NUM_THREADS=$MMSEQS_NUM_THREADS"

if [[ ! -f "$DB_ROOT/target_db.dbtype" || ! -f "$DB_ROOT/target_db.index" ]]; then
  shard_data=$(find "$DB_ROOT" -maxdepth 1 -type f -name 'target_db.[0-9]*' | wc -l | tr -d ' ')
  shard_idx=$(find "$DB_ROOT" -maxdepth 1 -type f -name 'target_db.index.[0-9]*' | wc -l | tr -d ' ')
  if [[ "$shard_data" -gt 0 && "$shard_idx" -gt 0 ]]; then
    log "target_db index missing but split shards detected; attempting shard recovery first"
    if ! recover_split_target_db; then
      if [[ "$ALLOW_FALLBACK_DOWNLOAD" == "1" ]]; then
        log "shard recovery failed; fallback download is enabled"
      else
        log "ERROR: shard recovery failed and fallback download is disabled"
        exit 1
      fi
    fi
  fi
fi

if [[ ! -f "$DB_ROOT/target_db.dbtype" || ! -f "$DB_ROOT/target_db.index" ]]; then
  log "downloading UniRef100 FASTA from $UNIREF_FASTA_URL"
  run_with_heartbeat "download_uniref100_fasta" wget -c --progress=dot:giga -O "$FASTA_GZ" "$UNIREF_FASTA_URL"
  log "building target_db via mmseqs createdb (MMSEQS_FORCE_MERGE=1)"
  run_with_heartbeat "mmseqs_createdb" "$MMSEQS_BIN" createdb "$FASTA_GZ" "$DB_ROOT/target_db"
else
  log "target_db already exists, skipping createdb"
fi

if [[ ! -f "$DB_ROOT/target_db.dbtype" || ! -f "$DB_ROOT/target_db.index" ]]; then
  shard_data=$(find "$DB_ROOT" -maxdepth 1 -type f -name 'target_db.[0-9]*' | wc -l | tr -d ' ')
  shard_idx=$(find "$DB_ROOT" -maxdepth 1 -type f -name 'target_db.index.[0-9]*' | wc -l | tr -d ' ')
  avail=$(df -h /mmseqs_db | awk 'NR==2{{print $4}}')
  log "ERROR: incomplete target_db after createdb/recovery (dbtype/index missing). split_shards data=$shard_data index=$shard_idx avail=$avail"
  exit 1
fi

if [[ -f "$FASTA_GZ" ]]; then
  fasta_bytes=$(stat -c %s "$FASTA_GZ" 2>/dev/null || echo 0)
  rm -f "$FASTA_GZ" || true
  log "removed source FASTA after createdb bytes=$fasta_bytes path=$FASTA_GZ"
fi

if [[ ! -f "$DB_ROOT/target_db_padded.dbtype" ]]; then
  log "building padded DB via mmseqs makepaddedseqdb"
  run_with_heartbeat "mmseqs_makepaddedseqdb" "$MMSEQS_BIN" makepaddedseqdb "$DB_ROOT/target_db" "$DB_ROOT/target_db_padded"
else
  log "target_db_padded already exists, skipping makepaddedseqdb"
fi

if [[ ! -f "$DB_ROOT/target_db_padded.idx.dbtype" ]]; then
  log "building index via mmseqs createindex --index-subset 2"
  run_with_heartbeat "mmseqs_createindex" "$MMSEQS_BIN" createindex "$DB_ROOT/target_db_padded" "$TMP_ROOT/index_tmp" --index-subset 2
else
  log "target_db_padded idx already exists, skipping createindex"
fi

DB_BYTES=$(du -sb "$DB_ROOT" | awk '{{print $1}}')
MMSEQS_VERSION=$(printf "%s" "${{MMSEQS_VERSION_LINE:-}}" | tr -d '\\n')
cat > "$MANIFEST_JSON" <<JSON
{{
  "created_at": "{now}",
  "db_profile": "{db_profile}",
  "source": "uniref100_fasta",
  "fasta_url": "{uniref_fasta_url}",
  "mmseqs_prebuilt_url": "{mmseqs_prebuilt_url}",
  "mmseqs_version": "$MMSEQS_VERSION",
  "db_root": "$DB_ROOT",
  "target_db": "$DB_ROOT/target_db",
  "padded_db": "$DB_ROOT/target_db_padded",
  "size_bytes": $DB_BYTES
}}
JSON

CURRENT_STEP="finalize"
update_status "complete" "bootstrap_complete"
log "bootstrap complete: db_root=$DB_ROOT size_bytes=$DB_BYTES"
"""


def _stream_container_process(proc: modal.container_process.ContainerProcess, prefix: str = "") -> None:
    q: "queue.Queue[tuple[str, str]]" = queue.Queue()

    def _pump(tag: str, stream: object) -> None:
        for line in stream:
            q.put((tag, str(line).rstrip("\n")))
        q.put((tag, "__EOF__"))

    t_out = threading.Thread(target=_pump, args=("stdout", proc.stdout), daemon=True)
    t_err = threading.Thread(target=_pump, args=("stderr", proc.stderr), daemon=True)
    t_out.start()
    t_err.start()

    eof = 0
    while eof < 2:
        tag, line = q.get()
        if line == "__EOF__":
            eof += 1
            continue
        pfx = f"[{prefix} {tag}] " if prefix else ""
        print(f"{pfx}{line}", flush=True)

    proc.wait()


@app.local_entrypoint()
def main(
    volume_name: str = DEFAULT_VOLUME_NAME,
    tmp_volume_name: str = DEFAULT_TMP_VOLUME_NAME,
    volume_version: int = DEFAULT_VOLUME_VERSION,
    tmp_volume_version: int = DEFAULT_TMP_VOLUME_VERSION,
    db_profile: str = DEFAULT_DB_PROFILE,
    uniref_fasta_url: str = DEFAULT_UNIREF100_FASTA_URL,
    mmseqs_prebuilt_url: str = DEFAULT_MMSEQS_PREBUILT_URL,
    mmseqs_threads: int = 0,
    force_rebuild: bool = False,
    allow_fallback_download: bool = False,
    timeout_s: int = 60 * 60 * 24,
    heartbeat_seconds: int = 30,
    sandbox_cpu: int = int(os.environ.get("MMSEQS_BOOTSTRAP_CPU", "16")),
    sandbox_memory_mb: int = int(os.environ.get("MMSEQS_BOOTSTRAP_MEMORY_MIB", "65536")),
    detach: bool = True,
) -> None:
    if db_profile != "uniref100_only":
        raise ValueError("Currently only --db-profile uniref100_only is supported")
    if int(volume_version) not in (1, 2):
        raise ValueError("--volume-version must be 1 or 2")
    if int(tmp_volume_version) not in (1, 2):
        raise ValueError("--tmp-volume-version must be 1 or 2")

    db_vol = modal.Volume.from_name(
        volume_name,
        create_if_missing=True,
        version=int(volume_version),
    )
    tmp_vol = modal.Volume.from_name(
        tmp_volume_name,
        create_if_missing=True,
        version=int(tmp_volume_version),
    )
    lookup_app = modal.App.lookup(APP_NAME, create_if_missing=True)

    resolved_threads = int(mmseqs_threads) if int(mmseqs_threads) > 0 else max(1, int(sandbox_cpu))

    script = _build_bootstrap_script(
        db_profile=db_profile,
        uniref_fasta_url=uniref_fasta_url,
        mmseqs_prebuilt_url=mmseqs_prebuilt_url,
        mmseqs_threads=resolved_threads,
        force_rebuild=force_rebuild,
        allow_fallback_download=allow_fallback_download,
        heartbeat_seconds=heartbeat_seconds,
    )

    sb = modal.Sandbox.create(
        app=lookup_app,
        image=image,
        timeout=max(600, int(timeout_s)),
        cpu=max(1, int(sandbox_cpu)),
        memory=max(1024, int(sandbox_memory_mb)),
        volumes={"/mmseqs_db": db_vol, "/mmseqs_tmp": tmp_vol},
    )
    proc = sb.exec("bash", "-lc", script, text=True)

    print(f"Started sandbox: {sb.object_id}", flush=True)
    print(f"Target DB volume: {volume_name} (v{int(volume_version)})", flush=True)
    print(f"Scratch volume: {tmp_volume_name} (v{int(tmp_volume_version)})", flush=True)
    print(
        f"Sandbox resources: cpu={max(1, int(sandbox_cpu))} "
        f"memory_mib={max(1024, int(sandbox_memory_mb))}",
        flush=True,
    )
    print(f"MMseqs thread cap: {resolved_threads}", flush=True)
    print("Mounts: /mmseqs_db and /mmseqs_tmp", flush=True)

    if detach:
        print("Detach mode enabled; sandbox continues server-side if local session disconnects.", flush=True)
        print(f"Follow logs via: modal container logs {sb.object_id}", flush=True)
        return

    _stream_container_process(proc, prefix=sb.object_id)
    rc: Optional[int] = proc.returncode
    if rc not in (0, None):
        raise RuntimeError(f"Sandbox bootstrap process failed with return code {rc}")
