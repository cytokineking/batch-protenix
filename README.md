# Protenix Modal Batch Pipeline

Batch protein interface prediction pipeline on Modal using Protenix v1 + ipSAE scoring.

This repository now centers on:

- `modal_protenix_batch.py` - Modal cloud pipeline entrypoints.
- `ipsae.py` - ipSAE scorer used after Protenix inference.
- `Protenix/` - local Protenix checkout embedded into the Modal image.

## Overview

The pipeline is row-centric:

1. Read a pair CSV where each row is one binder-target pair.
2. Resolve required MSAs per-task (dependency-gated, non-blocking), then run Protenix for ready rows.
3. Select the best structure by highest `iptm` (`global` or `per_seed` scope).
4. Run ipSAE on the selected structure using a compatibility adapter.
5. Write per-task outputs plus a run summary CSV.

## Prerequisites

1. Install Modal and authenticate:

```bash
pip install modal
modal token new
```

2. Keep these paths present in repo root:

- `modal_protenix_batch.py`
- `ipsae.py`
- `Protenix/`

## Main Command

```bash
modal run modal_protenix_batch.py::run_pipeline --help
```

## Required Input CSV

`--pair-csv` should contain one pair per row (header optional). The first 4 columns are used:

1. binder name
2. binder sequence
3. target name
4. target sequence

Example:

```csv
binder_name,binder_sequence,target_name,target_sequence
b1,EVQLVESGGGLVQPGGSLRLSCAAS...,targetA,MKAILVVLLYTFATANAD...
b2,DIQMTQSPSSLSASVGDRVTITCR...,targetB,MSPQTETKASVGFKAGVKEY...
```

## Quick Start

### 0) Initialize Protenix runtime once

Populate checkpoint + required common runtime artifacts on Modal Volume:

```bash
modal run modal_protenix_batch.py::init_protenix_runtime \
  --model-name protenix_base_default_v1.0.0
```

`run_pipeline` now uses validate-only runtime checks and will fail fast if these files are missing.

### 1) Target MSA fetched from MMseqs/ColabFold

Strict host policy is default, so provide an allowed host explicitly:

```bash
modal run modal_protenix_batch.py::run_pipeline \
  --pair-csv ./pairs.csv \
  --output-dir ./results \
  --binder-mode de_novo \
  --target-msa-source mmseqs \
  --mmseqs-mode colabfold \
  --mmseqs-host-url https://api.colabfold.com
```

### 2) Target MSA provided via mapping CSV

```bash
modal run modal_protenix_batch.py::run_pipeline \
  --pair-csv ./pairs.csv \
  --output-dir ./results \
  --binder-mode full_msa \
  --target-msa-source provided \
  --target-msa-map-csv ./target_msa_map.csv \
  --mmseqs-mode colabfold \
  --mmseqs-host-url https://api.colabfold.com
```

## CLI Reference (`run_pipeline`)

### Core inputs

- `--pair-csv <path>` (required)
- `--output-dir <path>` (default: `./results_protenix`)

### Binder MSA controls

- `--binder-mode {de_novo,fixed_msa,full_msa}` (default: `de_novo`)
- `--binder-fixed-msa-dir <dir>` required for `fixed_msa`  
  Expected files: `pairing.a3m`, `non_pairing.a3m`

### Target MSA controls

- `--target-msa-source {provided,mmseqs}` (default: `mmseqs`)
- `--target-msa-map-csv <path>` required when `target_msa_source=provided`

Supported columns in target MSA map CSV:

- `target_name` or `name` or `target`
- `target_sequence` or `sequence`
- `msa_dir` or `msa_path` (directory containing `pairing.a3m` and `non_pairing.a3m`)
- or explicit `pairing_path` and `non_pairing_path`/`unpaired_msa_path`

Lookup priority is: `(name + sequence)` -> `sequence` -> `name`.

### Optional antitarget/self controls

- `--include-antitarget <bool>` (default: `false`)
- Antitarget input:
  - `--antitarget-csv <path>` (single row, `name,sequence`; header optional), or
  - `--antitarget-name <str>` + `--antitarget-seq <seq>`
- `--antitarget-msa-source {provided,mmseqs,none}` (default: `mmseqs`)
- `--antitarget-msa-dir <dir>` required for `antitarget_msa_source=provided`
- `--include-self <bool>` (default: `false`)

### MMseqs/ColabFold controls

- `--mmseqs-mode colabfold` (only supported mode)
- `--mmseqs-host-url <url>`
- `--mmseqs-host-policy {strict,allow-default}` (default: `strict`)
  - `strict`: host URL is required and must match allowlist policy.
  - `allow-default`: if no host URL is given, defaults to `https://api.colabfold.com`.
- `--mmseqs-cache-dir <path>`  
  Note: currently fixed to `/msa_cache` inside Modal workers; other values are ignored with warning.
- `--mmseqs-cache-mode {none,read,write,readwrite}` (default: `readwrite`)
- `--mmseqs-store-fetched-msas <bool>` (forces cache writes)
- `--mmseqs-pairing-strategy {greedy,query_only,copy_non_pairing}` (default: `greedy`)
- `--mmseqs-db-tag <str>` (default: `colabfold_env`)

### Inference controls

- `--model-name <name>` (default: `protenix_base_default_v1.0.0`)
- `--seeds <csv>` (default: `101`, example: `101,102`)
- `--sample-diffusion-n-sample <int>` (default: `5`)
- `--sample-diffusion-n-step <int>` (default: `200`)
- `--n-cycle <int>` (default: `10`)
- `--best-sample-scope {global,per_seed}` (default: `global`)
- `--task-timeout-s <int>` (default: `7200`)

### ipSAE controls

- `--pae-cutoff <float>` (default: `15.0`)
- `--dist-cutoff <float>` (default: `15.0`)

### Runtime + streaming controls

- `--gpu <type>` (default: `A100-80GB`)
- `--max-parallel <int>` (default: `1`)
  - Max concurrent inference task executions.
- Streaming is always enabled for this pipeline.
- `--stream-logs <bool>` (default: `true`)
- `--log-chunk-seconds <float>` (default: `1.0`)
- `--heartbeat-seconds <float>` (default: `15.0`)
- `--run-id <id>` (optional)
- `--sync-interval <seconds>` (default: `5.0`)

### Resume + lease controls

- `--resume {auto,always,never}` (default: `auto`)
- `--retry-errors <bool>` (default: `true`)
- `--stale-running-minutes <int>` (default: `30`)
- `--overwrite-existing <bool>` (default: `false`)
- `--resume-manifest-override <bool>` (default: `false`)
- `--lease-timeout-s <int>` (default: `900`)
- `--lease-heartbeat-interval-s <int>` (default: `60`)

### Worker lifecycle controls

- `--worker-max-runtime-s <int>` (default: `6800`)
- `--worker-max-tasks <int>` (default: `0`, disabled)
  - Max number of task dispatches from this local scheduler process before pending tasks are terminalized.
- `--worker-idle-timeout-s <int>` (default: `120`)
  - If no scheduler progress occurs and nothing is runnable, pending tasks are terminalized after this timeout.

### MSA scheduler controls

- `--msa-min-submit-interval-s <float>` (default: `1.0`)
- `--msa-global-rate-key <str>` (default: `protenix_msa_global`)
- `--msa-max-inflight <int>` (currently must be `1`)

### Long-run reliability

Do not use detached mode with `run_pipeline`:

```bash
modal run -d modal_protenix_batch.py::run_pipeline ...
```

`run_pipeline` is a local entrypoint orchestrator; in detached mode, only the last triggered remote function is kept alive after local disconnect. Run it attached from a stable shell/session instead.

Recommended:

```bash
modal run modal_protenix_batch.py::run_pipeline ...
```

## Utility Entry Points

### Prewarm Protenix runtime cache

Downloads/validates checkpoint + required common cache artifacts to the runtime volume.
Inference workers do not auto-download these files.

```bash
modal run modal_protenix_batch.py::init_protenix_runtime \
  --model-name protenix_base_default_v1.0.0
```

### List supported GPUs

```bash
modal run modal_protenix_batch.py::list_gpus
```

### Test Modal GPU connection

```bash
modal run modal_protenix_batch.py::test_connection --gpu A100-80GB
```

## Output Layout

`--output-dir` contains:

```text
<output_dir>/
  run_metadata.json
  pair_summary.csv
  events.jsonl
  logs/
    <task_id>.log
    <task_id>.stdout.log
    <task_id>.stderr.log
  pairs/
    <pair_id>/
      pair.json
      target.status.json
      target.metrics.json
      target.best.cif
      target.best_summary.json
      target.ipsae.json
      target.candidates/
        s<seed>_n<rank>.cif
        s<seed>_n<rank>.summary.json
        s<seed>_n<rank>.full_data.json
      antitarget.* / self.* (when enabled)
  by_target/
    <target_slug>/
      <binder_slug>__vs__<target_slug>__<short_hash>.cif
```

`pair_summary.csv` columns:

- `task_id`, `pair_id`, `row_index`
- `partner_role`, `partner_name`
- `binder_name`, `binder_seq`
- `target_name`, `target_seq`
- `status`
- `best_sample_scope`, `best_seed`, `best_sample_rank`
- `best_iptm`, `best_ptm`, `best_ranking_score`
- `ipsae`, `ipsae_d0chn`, `ipsae_d0dom`, `ipsae_error`
- `error`

## Notes

- Target MSA is enforced per target task.
- MSA fetching is dependency-gated and non-blocking: inference starts as soon as any task becomes ready.
- Streaming artifacts and logs are written incrementally during task execution.
- `run_metadata.json` includes `run_status` as one of: `complete_success`, `complete_with_errors`, `incomplete`.
- `fixed_msa` mode performs warn-only compatibility checks against binder sequences.
- Best-structure selection is based on highest `iptm`.
- `run_metadata.json` captures key run configuration and run-level status for reproducibility.
