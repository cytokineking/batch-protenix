#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time

import modal

APP_NAME = "mmseqs-volume-sync"

SRC_DB_VOLUME_NAME = os.environ.get("MMSEQS_SRC_DB_VOLUME_NAME", "mmseqs-uniref100-db-2")
DST_DB_VOLUME_NAME = os.environ.get("MMSEQS_DST_DB_VOLUME_NAME", "mmseqs-uniref100-db-2-v2")
SRC_TMP_VOLUME_NAME = os.environ.get("MMSEQS_SRC_TMP_VOLUME_NAME", "mmseqs-tmp-2")
DST_TMP_VOLUME_NAME = os.environ.get("MMSEQS_DST_TMP_VOLUME_NAME", "mmseqs-tmp-2-v2")
SYNC_CPU = int(os.environ.get("MMSEQS_VOLUME_SYNC_CPU", "8"))
SYNC_MEMORY_MIB = int(os.environ.get("MMSEQS_VOLUME_SYNC_MEMORY_MIB", "32768"))
SYNC_TIMEOUT_S = int(os.environ.get("MMSEQS_VOLUME_SYNC_TIMEOUT_S", str(12 * 60 * 60)))

SRC_DB_MOUNT = "/src_db"
DST_DB_MOUNT = "/dst_db"
SRC_TMP_MOUNT = "/src_tmp"
DST_TMP_MOUNT = "/dst_tmp"

SRC_DB_VOLUME = modal.Volume.from_name(SRC_DB_VOLUME_NAME)
DST_DB_VOLUME = modal.Volume.from_name(DST_DB_VOLUME_NAME)
SRC_TMP_VOLUME = modal.Volume.from_name(SRC_TMP_VOLUME_NAME)
DST_TMP_VOLUME = modal.Volume.from_name(DST_TMP_VOLUME_NAME)

image = modal.Image.debian_slim().apt_install("rsync", "coreutils")
app = modal.App(APP_NAME, image=image)


def _run(cmd: str) -> None:
    print(f"[cmd] {cmd}", flush=True)
    proc = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}")


@app.function(
    cpu=SYNC_CPU,
    memory=SYNC_MEMORY_MIB,
    timeout=SYNC_TIMEOUT_S,
    volumes={
        SRC_DB_MOUNT: SRC_DB_VOLUME,
        DST_DB_MOUNT: DST_DB_VOLUME,
        SRC_TMP_MOUNT: SRC_TMP_VOLUME,
        DST_TMP_MOUNT: DST_TMP_VOLUME,
    },
)
def sync_mmseqs_volumes() -> None:
    start = time.time()
    print(
        f"[sync] src_db={SRC_DB_VOLUME_NAME} -> dst_db={DST_DB_VOLUME_NAME}; "
        f"src_tmp={SRC_TMP_VOLUME_NAME} -> dst_tmp={DST_TMP_VOLUME_NAME}; "
        f"cpu={SYNC_CPU} mem_mib={SYNC_MEMORY_MIB} timeout_s={SYNC_TIMEOUT_S}",
        flush=True,
    )
    _run(
        "set -euo pipefail; "
        f"mkdir -p {DST_DB_MOUNT} {DST_TMP_MOUNT}; "
        f"du -sh {SRC_DB_MOUNT} {SRC_TMP_MOUNT} || true; "
        f"du -sh {DST_DB_MOUNT} {DST_TMP_MOUNT} || true"
    )
    _run(
        "set -euo pipefail; "
        f"rsync -aH --inplace --partial --info=progress2 {SRC_DB_MOUNT}/ {DST_DB_MOUNT}/"
    )
    _run(
        "set -euo pipefail; "
        f"rsync -aH --inplace --partial --info=progress2 {SRC_TMP_MOUNT}/ {DST_TMP_MOUNT}/"
    )
    # Explicit commits so follow-up jobs can immediately read copied state.
    DST_DB_VOLUME.commit()
    DST_TMP_VOLUME.commit()
    elapsed = time.time() - start
    _run(
        "set -euo pipefail; "
        f"du -sh {DST_DB_MOUNT} {DST_TMP_MOUNT} || true"
    )
    print(f"[sync] complete in {elapsed:.1f}s", flush=True)


@app.local_entrypoint()
def main() -> None:
    call = sync_mmseqs_volumes.spawn()
    print(
        f"Spawned volume sync call_id={call.object_id}. "
        "Use `modal app list` and `modal app logs <app_id>` to monitor."
    )
