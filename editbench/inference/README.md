# Inference Module

Build **inference (infbench)** datasets from the final **bench** and run model inference. Predictions are written under a fixed directory layout and can be consumed by the evaluation harness.

## Overview

1. **Bench → Infbench** — Convert bench JSONL to inference variants (with `prompt`, `pre_edits`, `ground_truth`) using `prompt_builder`. Multiple variants (e.g. info ratio, BM25, with/without issue body) correspond to different **run_id**s.
2. **Run inference** — Call `run_api` to run one or more models on one or more (dataset, run_id) combinations. Outputs are stored under **experiment_results** by model and run_id.

**Default paths**

- **Infbench data:** `crawled_data/infbench/` (see `editbench.config`: `SRC_INF_BENCHMARK_DATA`).
- **Prediction output:** `experiment_results/{model_name}/T={temperature}/n={n}/{dataset_stem}.jsonl` (see `editbench.inference.constants`: `EXPERIMENTAL_RESULTS`).

---

## 1. Convert bench to inference (infbench)

The **bench** has full patch history (`work_patch_list`). The **infbench** adds a split into “previous edits” (`pre_edits`) and “ground truth” (`ground_truth`), plus a constructed `prompt` for the model. One bench can produce several infbench files (e.g. different info ratios or retrieval settings), each identified by a **run_id**.

### Command: `prompt_builder`

```bash
# Basic: 20% of edit steps as history (info_pct=0.2), PR + issue in requirement
python -m editbench.inference.prompt_builder \
  --dataset-name ./crawled_data/bench/all-task-instances.jsonl \
  --save-path ./crawled_data/infbench/all-task-instances_0.2.jsonl \
  --info-pct 0.2

# No PR message, no issue message (requirement can be empty for “only changes” prompt)
python -m editbench.inference.prompt_builder \
  --dataset-name ./crawled_data/bench/django-django-task-instances.jsonl \
  --save-path ./crawled_data/infbench/django-django-task-instances_0.2.jsonl \
  --info-pct 0.2 \
  --no-pr-mes \
  --no-issue-mes

# BM25 retrieval for code context instead of oracle; top-5 files
python -m editbench.inference.prompt_builder \
  --dataset-name ./crawled_data/bench/all-task-instances.jsonl \
  --save-path ./crawled_data/infbench/all-task-instances_0.2_bm25_5.jsonl \
  --info-pct 0.2 \
  --use-bm25 \
  --topn 5
```

**Arguments**

- `--dataset-name`: Input bench JSONL path (usually after merge: `all-task-instances.jsonl` or per-repo).
- `--save-path`: Output infbench JSONL path. Naming convention for later use: e.g. `all-task-instances_0.2.jsonl`, `all-task-instances_0.2_bm25_5.jsonl`, `all-task-instances_0.2_body_issue.jsonl` so that **run_id** can be derived (e.g. `0.2`, `0.2_bm25_5`, `0.2_body_issue`).
- `--info-pct`: Fraction of edit steps used as history (e.g. `0.2`, `0.4`, `0.6`, `0.8`). Drives split into `pre_edits` vs `ground_truth`.
- `--no-pr-mes` / `--no-issue-mes`: Omit PR title/body or issue message from the requirement text.
- `--use-bm25`: Use BM25 retrieval for code context (requires repo in evaluation’s `MAP_INSTALLED_REPO` and `base_commit`).
- `--topn`: Number of files for BM25 (default: 10).

**Usage note:** Run this **per variant** you need; then either merge per-repo infbench files with `merge_utils merge-infbench` (see [Collection](../collection/README.md)) or place single-repo files under `crawled_data/infbench/` with names like `{name}-task-instances_{run_id}.jsonl` so that `run_api` and evaluation can resolve paths by `dataset_name` and `run_id`.

---

## 2. Run inference (`run_api`)

Uses OpenAI-compatible API (OpenAI, Claude, Qwen, Gemini, DeepSeek, etc.) to generate a patch per instance. Reads from **infbench** JSONL and writes one JSONL per (dataset, model, run_id) under **experiment_results**.

**Environment**

- **OPENAI_KEYS** or **OPENAI_KEY**: Comma-separated API keys (round-robin across workers).
- **BASE_URL** (optional): Override API base URL for compatible endpoints.

### Commands

```bash
# Single dataset (e.g. "all"), one model, one run_id (default 0.2)
python -m editbench.inference.run_api \
  --dataset_name all \
  --model deepseek-v3.2 \
  --run_id 0.2

# Multiple models and run_ids
python -m editbench.inference.run_api \
  --dataset_name all \
  --model qwen3-235b-a22b deepseek-v3.2 \
  --run_id 0.2 0.2_bm25_5 \
  --max_workers 10 \
  --timeout 600

# Restrict to sampled instance IDs (e.g. for development or leaderboard subset)
python -m editbench.inference.run_api \
  --dataset_name all \
  --model claude-sonnet-4-5-20250929 \
  --run_id 0.2 \
  --sampled_ids_file ./crawled_data/infbench/sampled_instance_ids_0.2.json
```

**Arguments**

- `--dataset_name`: One or more of `all`, `owner/repo`, or a path to a `.jsonl` file. With `all` or repo name, input path is `crawled_data/infbench/{name}-task-instances_{run_id}.jsonl`.
- `--model`: One or more model names (e.g. `gpt-5-codex`, `claude-sonnet-4-5-20250929`, `gemini-2.5-pro`, `deepseek-v3.2`, `qwen3-235b-a22b`, `gpt-4.1`). Must be supported in `run_api` (see `MAP_MODEL_TO_COFIG`, `MODEL_LIMITS`, etc.).
- `--run_id`: One or more run IDs (default: `0.2`). Choices typically include: `0.2`, `0.4`, `0.6`, `0.8`, `0.2_bm25_1`, `0.2_bm25_3`, `0.2_bm25_5`, `0.2_body_issue`, `None_body_issue`.
- `--sampled_ids_file`: Optional JSON file with key `sampled_instance_ids`; only these instances are run.
- `--max_workers`: Concurrent threads (default: 10).
- `--timeout`: Per-request timeout in seconds (default: 600).

**Output (default location)**

- Directory: `experiment_results/{model_name}/T={temperature}/n={n}/`.
- File: `{dataset_stem}.jsonl` (e.g. `all-task-instances_0.2.jsonl`). Each line is a JSON object with at least `instance_id`, `model_name`, `prompt`, `full_output`, `model_patch`, and optionally `cost`, `response_time`, `status`, `error`.

**Model selection and defaults**

- Model-specific config (temperature, top_p, n) is in `MAP_MODEL_TO_COFIG` in `run_api.py` (e.g. temperature 0 for most, 1 for `gpt-5-codex`).
- Token limits and cost tables are in `MODEL_LIMITS`, `MODEL_COST_PER_INPUT`, `MODEL_COST_PER_OUTPUT`. If prompt exceeds the limit, the code may strip the last file(s) from the prompt and retry once.
- Predictions are **appended** to the same JSONL; already-seen `instance_id`s in the file are skipped, so you can resume runs.

---

## 3. Merging inference results across repos

If you ran inference per-repo (different `dataset_name` or per-repo infbench files), merge result JSONL with:

```bash
python -m editbench.utils.merge_utils merge-results \
  --model-name your-model-name \
  [--temperature 0] [--n 1] [--run-id 0.2] \
  [--repos repo1 repo2]
```

Output is written under the same `experiment_results/{model}/T=.../n=.../` tree (e.g. `all-task-instances_0.2.jsonl`). See [Collection README](../collection/README.md) and `merge_utils` for details.

---

## Summary

| Step | Command / script | Input | Output |
|------|------------------|--------|--------|
| Bench → Infbench | `prompt_builder` | Bench JSONL | Infbench JSONL (e.g. `*_0.2.jsonl`) |
| Inference | `run_api` | Infbench path(s) + model + run_id | `experiment_results/{model}/T=.../n=.../*.jsonl` |
| Merge results | `merge_utils merge-results` | Per-repo result JSONL | Single `all-task-instances_{run_id}.jsonl` per run_id |

Default storage for inference outputs is **experiment_results** (project root). Use the same `run_id` and dataset naming as in the evaluation module so that `run_evaluation run` can find predictions by `--model` and `--run_id`.
