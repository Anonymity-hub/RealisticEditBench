# Evaluation Module

Docker-based evaluation harness for RealisticEditBench: apply model (or gold) patches, run project tests, and compute pass/fail and code-similarity metrics. Supports single-run, batch (multiple models/datasets), and summary/analysis with optional filtering.

## Overview

- **run** — For each (dataset, predictions, run_id): build Docker env, apply patch, run eval script, collect report and metrics (applied/resolved, CodeBLEU, edit distance, Jaccard, TF-IDF cosine, file hit rate).
- **summary** — Print human-readable summary from an existing output JSON; optional filter by instance IDs or cutoff date.
- **batch** — Convenience to run evaluation for multiple (dataset, model, run_id) combinations using default prediction paths under `experiment_results`.

**Requirements**

- Docker installed and running (see main [README](../../README.md)); evaluation is **not supported on Windows**; ARM64 (e.g. Mac M-series) may have compatibility issues.
- Sufficient resources: recommended ≥120GB free storage, 16GB RAM, 8 CPU cores; increase `--open_file_limit` if you hit “too many open files”.

---

## 1. Scenario: Evaluate gold (oracle) patches

Use this to establish an upper bound: patches are the ground-truth from the dataset.

```bash
python -m editbench.evaluation.run_evaluation run \
  --dataset_name all \
  --predictions_path gold \
  --run_id 0.2 \
  --max_workers 2 \
  [--timeout 600] [--cache_level eval] [--open_file_limit 1048576]
```

- **dataset_name**: `all` or `owner/repo` or path to infbench JSONL; used to resolve `crawled_data/infbench/{name}-task-instances_{run_id}.jsonl` when you pass `all` or repo name.
- **predictions_path** `gold`: use each instance’s `ground_truth` as the patch; no prediction file needed.
- Output report path follows: `experiment_results/gold/{name}-{run_id}-output.json` (see config `SRC_EXPERIMENTS`).

---

## 2. Scenario: Evaluate one model’s predictions (single run_id)

Predictions are in a JSONL (or JSON) produced by the inference module.

```bash
python -m editbench.evaluation.run_evaluation run \
  --dataset_name all \
  --predictions_path ./experiment_results/deepseek-v3.2/T=0/n=1/all-task-instances_0.2.jsonl \
  --run_id 0.2 \
  --max_workers 4 \
  [--timeout 600]
```

- **dataset_name** and **run_id** must match the infbench and the prediction file (e.g. `all` + `0.2` → `all-task-instances_0.2.jsonl`).
- Report and metrics are written under `experiment_results/{model}/T=.../n=.../` and a single `{name}-{run_id}-output.json` is produced (path printed at the end).

---

## 3. Scenario: Batch evaluate multiple models and/or run_ids

Let the harness infer prediction paths from `--dataset_name`, `--model`, and `--run_id` (paths under `experiment_results/{model}/T=.../n=.../`).

```bash
python -m editbench.evaluation.run_evaluation run \
  --dataset_name all \
  --model claude-sonnet-4-5-20250929 deepseek-v3.2 \
  --run_id 0.2 0.4 \
  [--max_workers 4] [--timeout 600] [--sampled_ids_file ./crawled_data/infbench/sampled_instance_ids_0.2.json]
```

- For each (dataset_name, model, run_id), predictions path is set to  
  `experiment_results/{model}/T={temperature}/n={n}/{name}-task-instances_{run_id}.jsonl`.
- **sampled_ids_file**: optional JSON with `sampled_instance_ids`; only those instances are evaluated (same as inference subset).

---

## 4. Scenario: Custom dataset path and instance list

Use a direct path to infbench and optionally restrict instance IDs.

```bash
python -m editbench.evaluation.run_evaluation run \
  --dataset_name ./crawled_data/infbench/all-task-instances_0.2.jsonl \
  --predictions_path ./experiment_results/my-model/T=0/n=1/all-task-instances_0.2.jsonl \
  --run_id 0.2 \
  --instance_ids id1 id2 id3 \
  [--max_workers 4]
```

- **dataset_name** can be a full path to a `.jsonl` file; **run_id** is still required (used for log dirs and output naming).
- **instance_ids**: only these instances are loaded and evaluated; already-completed instances (existing `report.json`) are skipped by default.

---

## 5. Summary and analysis (no Docker run)

Print a concise summary from an existing output JSON. Optionally filter by instance IDs or by date.

```bash
# Basic summary
python -m editbench.evaluation.run_evaluation summary \
  --report_path ./experiment_results/deepseek-v3.2/T=0/n=1/all-0.2-output.json

# Exclude instances from a filter list (e.g. paper’s union of filtered IDs)
python -m editbench.evaluation.run_evaluation summary \
  --report_path ./experiment_results/deepseek-v3.2/T=0/n=1/all-0.2-output.json \
  --filter_ids_file ./union-all.json

# Restrict to instances before/after a cutoff date (dataset must exist for run_id to resolve created_at)
python -m editbench.evaluation.run_evaluation summary \
  --report_path ./experiment_results/deepseek-v3.2/T=0/n=1/all-0.2-output.json \
  --cutoff_date 20250101 \
  --date_mode before
```

- **filter_ids_file**: JSON with key `union_filter_instance_ids`; these instances are excluded before recomputing metrics.
- **cutoff_date**: `YYYYMMDD` or `YYYY-MM-DD`.
- **date_mode**: `before` = only instances with `created_at` before cutoff; `after` = only after.

---

## Common arguments (run)

| Argument | Default | Description |
|----------|--------|-------------|
| `--dataset_name` | `all` | Dataset: `all`, `owner/repo`, or path to `.jsonl`. |
| `--predictions_path` | `gold` if no `--model` | Path to prediction JSONL/JSON or `gold`. |
| `--run_id` | `0.2` | Run ID (must match infbench and prediction file naming). |
| `--model` | — | One or more model names; used to build prediction path and output dir. |
| `--instance_ids` | — | Restrict to these instance IDs. |
| `--sampled_ids_file` | — | JSON with `sampled_instance_ids`; overrides/combines with instance list. |
| `--max_workers` | 4 | Parallel evaluation workers. |
| `--timeout` | 600 | Per-instance test timeout (seconds). |
| `--force_rebuild` | false | Force rebuild Docker images. |
| `--cache_level` | `eval` | `none` \| `base` \| `env` \| `eval`; higher = more caching. |
| `--clean` | false | Remove images above cache level after run. |
| `--open_file_limit` | 1048576 | RLIMIT_NOFILE (raise if you see “too many open files”). |
| `--max_eval_images` | — | When `cache_level=eval`, keep at most this many eval images (optional). |

---

## Output and metrics

- **Per-instance:** Under `logs/run_evaluation/{run_id}/{model}/{instance_id}/`: `report.json`, `patch.diff`, `eval.sh`, test output, etc.
- **Aggregate:** A single JSON file per run, e.g. `experiment_results/{model}/T=0/n=1/all-0.2-output.json`, containing:
  - Counts: total, completed, resolved, applied, unresolved, empty patch.
  - Rates: `%applied`, `%resolved`.
  - Averages: CodeBLEU, normalized edit distance, Jaccard similarity, TF-IDF cosine similarity, file hit rate (recall).
  - Lists: `resolved_ids`, `unresolved_ids`, `completed_ids`, etc., and `instance_details` with per-instance metrics.

**Resolved** means the patch applied and the eval script reported all target tests as passed (or equivalent). **Applied** means the patch applied successfully; tests may still fail.

---

## Usage notes

1. **Dataset vs run_id:** The evaluation script resolves the infbench file from `dataset_name` and `run_id` (e.g. `all` + `0.2` → `crawled_data/infbench/all-task-instances_0.2.jsonl`). Ensure that file exists and matches the prediction file’s run_id.
2. **Skip completed:** By default, instances that already have a `report.json` for the same run_id and model are skipped. To re-run, delete the corresponding log subdir or use a different run_id.
3. **Gold run:** Use `--predictions_path gold` and do **not** pass `--model` when you want to evaluate ground-truth patches; the output path will be under `experiment_results/gold/`.
4. **Resource limits:** On large runs, set `--open_file_limit` high (e.g. 1048576) and consider `--max_eval_images` when using `cache_level=eval` to avoid disk and inode exhaustion.
5. **Platform:** Evaluation is intended for Linux (e.g. Ubuntu 22.04 x86_64). Windows is not supported; Mac ARM may have Docker issues.

For end-to-end flow (collection → editing split → gather bench → merge → infbench → inference → evaluation), see the [Collection](../collection/README.md), [Editing Split](../editing_split/README.md), and [Inference](../inference/README.md) READMEs.
