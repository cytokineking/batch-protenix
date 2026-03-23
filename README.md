# Protenix Modal Batch Pipeline

This repository provides a production-style Modal pipeline for large batch binder-target
screening with Protenix v1, plus ipSAE post-scoring for interface quality analysis.

It is designed for long-running, restartable batch jobs where you want:

- GPU-parallel structure prediction across many binder-target rows.
- Flexible MSA backends (`colabfold` API or local MMseqs2 GPU mode).
- Incremental streaming of logs/artifacts during execution.
- Resume-safe behavior after interruption or disconnects.

Core files:

- `modal_protenix_batch.py` - Modal cloud pipeline entrypoints.
- `modal_build_uniref100_db.py` - standalone detached bootstrap utility for local MMseqs UniRef100 volumes.
- `ipsae.py` - ipSAE scorer used after Protenix inference.
- Protenix + MMseqs2 are cloned into the Modal image at build time (pinned refs).

## Overview

The pipeline is row-centric and dependency-gated:

1. Parse `--pair-csv` where each row defines one binder-target task.
2. Resolve MSA dependencies for each task (without blocking unrelated ready tasks).
3. Dispatch Protenix inference in parallel as soon as each task becomes runnable.
4. Select best sample by top `iptm` (`global` or `per_seed` policy).
5. Run ipSAE on the selected candidate through a Protenix-to-ipSAE adapter.
6. Stream status/log/events during run; write structured outputs and summaries.
7. Support resume/retry workflows for interrupted runs.

Pipeline paradigm:

- This is a pairwise interface pipeline, not a monomer-only pipeline.
- Each CSV row is treated as a complex prediction request (`binder` vs `target`).
- Optional `antitarget`/`self` tasks add extra partner contexts for the same binder, but do not replace the required binder-target row.

## Prerequisites

1. Install Modal and authenticate:

```bash
pip install modal
modal token new
```

2. Keep these paths present in repo root:

- `modal_protenix_batch.py`
- `ipsae.py`

Optional override env vars for image source refs:

- `PROTENIX_GIT_URL`, `PROTENIX_GIT_REF`
- `MMSEQS2_GIT_URL`, `MMSEQS2_GIT_REF`

## Main Command

```bash
modal run modal_protenix_batch.py::run_pipeline --help
```

## VHH Template Prep

For VHH binder libraries, you can pre-analyze binders into reusable MSA template groups
without running the full structure pipeline.

Modal analyze-only workflow:

```bash
modal run modal_protenix_batch.py::prepare_vhh_binder_msas \
  --pair-csv ./test_batch.csv \
  --output-dir ./runs/vhh_grouping \
  --analyze-only \
  --framework-mode lengths_only
```

Local helper workflow:

```bash
python vhh_msa_templates.py \
  --pair-csv ./test_batch.csv \
  --output-dir ./runs/vhh_grouping_local \
  --framework-mode lengths_only
```

Framework grouping modes:

- `--framework-mode {exact,lengths_only}` for VHH template grouping
- Default: `lengths_only`
- `exact`: split templates by exact `FR1/FR2/FR3/FR4` sequences plus IMGT `CDR1/2/3` registers
- `lengths_only`: split templates by `FR1/FR2/FR3/FR4` lengths plus IMGT `CDR1/2/3` registers

Use `lengths_only` when you want framework variants with the same length/register architecture
to share a single MSA template representative.

## Required Input CSV

`--pair-csv` should contain one pair per row (header optional). The first 4 columns are used:

1. binder name
2. binder sequence
3. target name
4. target sequence

Rows missing any of these four fields are skipped.
If no valid rows remain, the run fails fast.

Single-protein predictions:

- A true monomer-only mode is not currently supported in this pipeline.
- You must provide both binder and target fields for each row.

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

### 3) Local MMseqs GPU backend (optional)

Initialize local UniRef100 MMseqs DB once (quick path):

```bash
modal run modal_protenix_batch.py::init_mmseqs_uniref100_db \
  --db-tag uniref100_v1 \
  --db-profile uniref100_only
```

Or use the standalone bootstrap utility (recommended for long detached builds and explicit volume targeting):

```bash
modal run --detach modal_build_uniref100_db.py \
  --volume-name mmseqs-uniref100-db \
  --tmp-volume-name mmseqs-tmp
```

For parallel/backfill bootstrap, target a second volume:

```bash
modal run --detach modal_build_uniref100_db.py \
  --volume-name mmseqs-uniref100-db-2 \
  --tmp-volume-name mmseqs-tmp-2
```

Notes:

- Bootstrap is storage-heavy. Default guardrails require large free space (default `--min-required-gb 300`, plus safety margin) and temp free space (default `--tmp-min-free-gb 20`).
- Current supported local DB profile is `uniref100_only` (`uniref100_plus_env` is intentionally fail-fast for now).
- `modal_build_uniref100_db.py` is a standalone app/bootstrap path and does not interfere with `run_pipeline` orchestration.

Then run pipeline in local mode:

```bash
modal run modal_protenix_batch.py::run_pipeline \
  --pair-csv ./pairs.csv \
  --output-dir ./results_local_gpu \
  --binder-mode full_msa \
  --target-msa-source mmseqs \
  --mmseqs-mode local_gpu \
  --mmseqs-db-tag uniref100_v1 \
  --mmseqs-local-gpu A100-80GB \
  --mmseqs-local-workers 1 \
  --mmseqs-local-batch-size 8 \
  --inference-max-batch-size 8
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

- `--mmseqs-mode {colabfold,local_gpu}`
- `--mmseqs-host-url <url>`
  - Required for `colabfold` mode.
  - In `local_gpu` mode, only needed when `--mmseqs-fallback colabfold` is enabled (or when explicitly forcing host resolution).
- `--mmseqs-host-policy {strict,allow-default}` (default: `strict`)
  - `strict`: host URL is required and must match allowlist policy.
  - `allow-default`: if no host URL is given, defaults to `https://api.colabfold.com`.
- `--mmseqs-cache-dir <path>`  
  Note: currently fixed to `/msa_cache` inside Modal workers; other values are ignored with warning.
- `--mmseqs-cache-mode {none,read,write,readwrite}` (default: `readwrite`)
- `--mmseqs-store-fetched-msas <bool>` (forces cache writes)
- `--mmseqs-pairing-strategy {greedy,query_only,copy_non_pairing}` (default: `greedy`)
- `--mmseqs-db-tag <str>` (default: `auto`)
  - `auto` resolves to `colabfold_env` in `colabfold` mode and `uniref100_v1` in `local_gpu` mode.
- `--mmseqs-fallback {none,colabfold}` (default: `none`)
  - `local_gpu` mode is fail-fast by default; set `colabfold` only if you explicitly want fallback.

### Local MMseqs controls (`--mmseqs-mode local_gpu`)

- `--mmseqs-local-workers <int>` (default: `1`)
- `--mmseqs-local-batch-size <int>` (default: `8`)
- `--mmseqs-local-db-profile {uniref100_only,uniref100_plus_env}` (default: `uniref100_only`)
  - `uniref100_plus_env` is reserved for a later phase; current implementation supports `uniref100_only`.
  - Must match the initialized DB manifest profile.
- `--mmseqs-local-gpu <type>` (default: `A10G`; available: `A10G`, `L40S`, `A100-40GB`, `A100-80GB`, `H100`)
  - Recommended: Ampere or newer (`A100-80GB` or `H100`) for local MMseqs GPU search performance.
- `--mmseqs-local-max-seqs <int>` (default: `300`)
- `--mmseqs-local-prefilter-mode <int>` (default: `1`)
- `--mmseqs-local-tmp-dir <path>` (default: `/tmp/mmseqs_work`)
- `--mmseqs-local-tmp-min-free-gb <float>` (default: `5.0`)
- `--mmseqs-local-commit-every-n <int>` (default: `1`)
- `--mmseqs-local-commit-interval-s <float>` (default: `30.0`)

Scratch-space behavior:

- Local mode checks free space before each fetch batch and fails fast when insufficient.
- If preferred tmp path is too full, workers can fall back to `/mmseqs_tmp`.

### Inference controls

- `--model-name <name>` (default: `protenix_base_default_v1.0.0`)
- `--seeds <csv>` (default: `101`, example: `101,102`)
- `--sample-diffusion-n-sample <int>` (default: `5`)
- `--sample-diffusion-n-step <int>` (default: `200`)
- `--n-cycle <int>` (default: `10`)
- `--best-sample-scope {global,per_seed}` (default: `global`)
- `--task-timeout-s <int>` (default: `7200`)

### Inference batching

Protenix accepts an `input.json` containing multiple samples; the pipeline can bundle multiple
ready tasks into a single Protenix invocation to amortize model load.

- `--inference-max-batch-size <int>` (default: `1`)
  - Upper bound only: inference workers start as soon as at least 1 task is ready.
  - Increase this to reduce repeated checkpoint loads.
- `--inference-batch-fill-window-s <float>` (default: `0`)
  - Optional: wait up to this many seconds to fill the batch (default is no waiting).

### ipSAE controls

- `--pae-cutoff <float>` (default: `15.0`)
- `--dist-cutoff <float>` (default: `15.0`)

### Runtime + streaming controls

- `--gpu <type>` (default: `A100-80GB`)
- `--max-parallel <int>` (default: `1`)
  - Max concurrent inference task executions.
- `--by-target-include-best-full-data <bool>` (default: `false`)
  - When enabled, exports the selected top-sample `full_data.json` next to the selected `.cif` in `by_target/<target>/`.
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
- `--msa-max-inflight <int>` (currently must be `1` in `colabfold` mode)
  - In `local_gpu` mode, submit throttling is disabled and this knob is ignored (batching/concurrency are controlled by local worker flags).

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

### Initialize local MMseqs UniRef100 DB

```bash
modal run modal_protenix_batch.py::init_mmseqs_uniref100_db \
  --db-tag uniref100_v1 \
  --db-profile uniref100_only
```

### Standalone UniRef100 volume bootstrap (detached-friendly)

This is the preferred operator workflow for large first-time DB builds, explicit target volume naming, and live progress checks.

```bash
modal run --detach modal_build_uniref100_db.py \
  --volume-name mmseqs-uniref100-db \
  --tmp-volume-name mmseqs-tmp
```

Useful flags:

- `--volume-name` (default: `mmseqs-uniref100-db`)
- `--tmp-volume-name` (default: `mmseqs-tmp`)
- `--force-rebuild`
- `--no-detach` (foreground mode)

How it relates to local MMseqs pipeline:

- `run_pipeline --mmseqs-mode local_gpu` reads DB artifacts/manifest from the MMseqs DB volume.
- Keep `--mmseqs-db-tag` aligned with the initialized DB in that target volume.
- You can bootstrap an alternate volume (for example `mmseqs-uniref100-db-2`) in parallel, then cut over by config/redeploy.

### Local MMseqs smoke benchmark

```bash
modal run modal_protenix_batch.py::smoke_local_mmseqs \
  --pair-csv ./test_batch.csv \
  --n-sequences 10 \
  --mmseqs-db-tag uniref100_v1 \
  --mmseqs-local-gpu A100-80GB
```

### List supported GPUs

```bash
modal run modal_protenix_batch.py::list_gpus
```

### Test Modal GPU connection

```bash
modal run modal_protenix_batch.py::test_connection --gpu A100-80GB
```

## Epitope Analysis Utility

For post-run structural epitope binning on grouped `mmCIF` outputs, use:

```bash
python scripts/epitope_binning_analysis.py \
  --cif-dir runs/PV2_cluster88/pv2_cluster88_gpa33_ontarget/by_target/gpa33 \
  --pair-summary runs/PV2_cluster88/pv2_cluster88_gpa33_ontarget/pair_summary.csv \
  --comparison-csv runs/PV2_cluster88/pv2_cluster88_analysis/vhh_target_decoy_comparison.csv \
  --target-name GPA33 \
  --outdir runs/PV2_cluster88/pv2_cluster88_gpa33_ontarget/epitope_analysis
```

Defaults:

- binder chain: `A`
- target chain: `B`
- contact cutoff: `4.5 A`
- `ipSAE` threshold: `0.6`

### Analysis modes

Default mode analyzes only successful complexes with `best_ipsae > --ipsae-threshold`:

```bash
python scripts/epitope_binning_analysis.py ... --ipsae-threshold 0.6
```

Optional exploratory mode analyzes all successful complexes and marks which ones pass the same threshold:

```bash
python scripts/epitope_binning_analysis.py ... \
  --ipsae-threshold 0.6 \
  --include-all-successful
```

Interpretation:

- default mode: `--ipsae-threshold` is used for filtering
- `--include-all-successful` mode: `--ipsae-threshold` is used for pass/fail annotation and bin-level quality summaries

Helpful flags:

- `--binder-chain <id>` / `--target-chain <id>`
- `--contact-cutoff <angstrom>`
- `--jaccard-distance-threshold <float>`
- `--approach-angle-threshold-deg <float>`
- `--interface-centroid-threshold <float>`
- `--representative-mode {highest_ipsae,medoid}`
- `--write-aligned-cifs` / `--no-write-aligned-cifs`
- `--comparison-csv <path>` for optional off-target annotation

### Epitope analysis outputs

`--outdir` from `epitope_binning_analysis.py` contains:

- `analyzed_complexes.csv`
  - one row per analyzed complex
  - includes `ipsae_threshold_used` and `passes_ipsae_threshold`
- `filtered_binders.csv`
  - compatibility alias of `analyzed_complexes.csv`
  - in `--include-all-successful` mode, this file still includes complexes that do not pass the threshold; use `passes_ipsae_threshold` to distinguish them
- `per_complex_geometry.csv`
  - one row per analyzed complex with contact counts, aligned centroids, approach vector, bin IDs, and threshold annotation fields
- `epitope_bins.csv`
  - binder-to-bin assignment table with confidence annotations, including `ipsae_threshold_used` and `passes_ipsae_threshold`
- `bin_summary.csv`
  - one row per final bin with:
    - `bin_size_all`
    - `bin_size_ipsae_pass`
    - `fraction_ipsae_pass`
    - summary stats for `best_ipsae`, `best_iptm`, and `best_pdockq2`
    - representative binder fields
    - consensus epitope residue sets
    - within-bin geometry summaries
- `target_residue_occupancy.csv`
  - campaign-level target residue occupancy map
- `bin_offtarget_annotation.csv`
  - optional off-target / decoy summaries per `(final_bin_id, target_name_off)`
- `bin_consensus_epitopes.json`
- `analysis_metadata.json`
- `aligned_representatives/`
  - optional aligned representative `mmCIF` files, one per final bin when aligned CIF writing is enabled
- `figures/`
  - occupancy, clustering, and geometry summary plots

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
      <binder_slug>__vs__<target_slug>__<short_hash>.full_data.json  # optional; when enabled
```

`pair_summary.csv` columns:

- `task_id`, `pair_id`, `row_index`
- `partner_role`, `partner_name`
- `binder_name`, `binder_seq`
- `target_name`, `target_seq`
- `status`
- `best_sample_scope`, `best_seed`, `best_sample_rank`, `n_candidates`, `n_interface_scored_candidates`
- `best_iptm`, `iptm_mean`, `iptm_std`
- `best_ptm`, `ptm_mean`, `ptm_std`
- `best_ranking_score`, `ranking_score_mean`, `ranking_score_std`
- `best_ipsae`, `ipsae_mean`, `ipsae_std`
- `best_ipsae_d0chn`, `ipsae_d0chn_mean`, `ipsae_d0chn_std`
- `best_ipsae_d0dom`, `ipsae_d0dom_mean`, `ipsae_d0dom_std`
- `best_pdockq`, `pdockq_mean`, `pdockq_std`
- `best_pdockq2`, `pdockq2_mean`, `pdockq2_std`
- `best_lis`, `lis_mean`, `lis_std`
- `ipsae_error`
- `error`

Count semantics:

- `n_candidates` is the number of successful Protenix candidate structures collected for the pair.
- `n_interface_scored_candidates` is the number of candidates whose stored interface metrics include a valid numeric `ipSAE`.

Interface metric note:

- `pDockQ`, `pDockQ2`, and `LIS` are prediction-derived interface metrics from `ipsae.py`, not true `DockQ`.

## Notes

- Target MSA is enforced per target task.
- MSA fetching is dependency-gated and non-blocking: inference starts as soon as any task becomes ready.
- Streaming artifacts and logs are written incrementally during task execution.
- `run_metadata.json` includes `run_status` as one of: `complete_success`, `complete_with_errors`, `incomplete`.
  - Tasks marked `running_elsewhere` (resume auto-skip) count toward `incomplete`, not `failed`.
- Local mode (`--mmseqs-mode local_gpu`) requires initialized MMseqs DB manifest on the MMseqs DB volume.
- In local mode, `--mmseqs-db-tag` should match the tag used with either `init_mmseqs_uniref100_db` or `modal_build_uniref100_db.py`.
- `fixed_msa` mode performs warn-only compatibility checks against binder sequences.
- Best-structure selection is based on highest `iptm`.
- `target.ipsae.json` remains the best-candidate interface-scoring artifact; per-candidate interface metrics are stored in `protenix_raw.candidate_summaries`.
- `run_metadata.json` captures key run configuration and run-level status for reproducibility.
