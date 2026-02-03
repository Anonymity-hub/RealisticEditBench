# Collection Module

Data collection pipeline for building RealisticEditBench from GitHub: collect PR/commit activity, filter by execution (Docker + pass/fail), optionally verify manually, then gather split tasks into bench and merge across repos.

## Overview

The collection flow is **per-repository**; you typically run each step for one or more repos, then merge all repo results at the end.

1. **run_collection** — Fetch PRs and build activity-format JSONL per repo.
2. **execute_filter** — Run Docker for each activity instance; keep only those that run and have **pass_to_fail** (and related) test outcomes.
3. **Manual verification (optional)** — Follow the paper’s criteria to manually verify/curate instances (e.g. using `report_valid.json` from execution filter).
4. **editing_split** — For each execution-filtered instance, run split pipeline (see [Editing Split](../editing_split/README.md)) so that `work_patch_list` is available.
5. **gather_bench** — From execution-filtered JSONL + split results, collect bench instances (with `work_patch_list`) into a bench file per repo.
6. **merge_utils** — Merge all per-repo bench (and optionally infbench) into `all-task-instances.jsonl`.

---

## 1. Run collection (`run_collection`)

Collects PRs (and can support commits) for given repos and writes **activity-format** JSONL: one file per repo under `path-prs` (raw PR data) and one per repo under `path-tasks` (task instances in activity format).

**Requirements**

- `GITHUB_TOKENS`: comma-separated GitHub tokens (used in parallel). Example: `GITHUB_TOKENS=$(gh auth token)` or set in `.env`.

**Commands**

```bash
# Basic: one or more repos, output dirs for raw PRs and task instances
python -m editbench.collection.run_collection \
  --repos django/django scikit-learn/scikit-learn \
  --path-prs ./crawled_data/raw \
  --path-tasks ./crawled_data/activity

# Limit tasks per repo and add a cutoff date suffix to filenames
python -m editbench.collection.run_collection \
  --repos astropy/astropy \
  --path-prs ./crawled_data/raw \
  --path-tasks ./crawled_data/activity \
  --max-tasks 100 \
  --cutoff-date 2024-06-01

# Do not overwrite existing PR/task files
python -m editbench.collection.run_collection \
  --repos owner/repo \
  --path-prs ./crawled_data/raw \
  --path-tasks ./crawled_data/activity \
  --no-override
```

**Output**

- `path-prs`: `{owner-repo}-prs.jsonl` (and optionally `-{cutoff_date}.jsonl`).
- `path-tasks`: `{owner-repo}-task-instances.jsonl` (activity format, one JSON object per line).

---

## 2. Execution filter (`execute_filter`)

Reads **activity** JSONL (from step 1), runs each instance in Docker (build env, apply patch, run tests before/after), and produces pass/fail reports. Only instances that execute successfully and have the desired test outcome (e.g. **pass_to_fail**) should be kept for the benchmark; the rest are filtered out at the “gather” step or after manual verification.

**Input:** Path(s) to activity task-instance JSONL (e.g. `./crawled_data/activity/{repo}-task-instances.jsonl`).

**Output**

- Per-instance logs under `logs/execution_filter/{instance_id}/` (e.g. `report.json`, test output before/after).
- Aggregated: `logs/execution_filter/report.json` and `report_valid.json` (instances with `f2p` non-empty).

**Commands**

```bash
# Single repo activity file
python -m editbench.collection.execute_filter \
  --dataset-name ./crawled_data/activity/astropy-astropy-task-instances.jsonl \
  --max-workers 4 \
  --timeout 1800

# Multiple repo files (run in sequence)
python -m editbench.collection.execute_filter \
  --dataset-name ./crawled_data/activity/django-django-task-instances.jsonl ./crawled_data/activity/scikit-learn-task-instances.jsonl \
  --max-workers 4

# Optional: force rebuild images, stricter cache, cleanup
python -m editbench.collection.execute_filter \
  --dataset-name ./crawled_data/activity/owner-repo-task-instances.jsonl \
  --max-workers 4 \
  --force-rebuild \
  --cache-level env \
  --clean
```

**Options**

- `--dataset-name`: One or more paths to activity JSONL.
- `--instance-ids`: Optional; restrict to listed instance IDs.
- `--max-workers`: Parallel workers (default: 4).
- `--timeout`: Per-instance timeout in seconds (default: 1800).
- `--cache-level`: `none` | `base` | `env` | `eval` (default: `env`).
- `--clean`: Remove images above cache level after run.
- `--open-file-limit`: RLIMIT_NOFILE (default: 4096).

---

## 3. Manual verification (paper-aligned)

After execution filter, refer to the paper for exact criteria (e.g. which test outcomes to require, quality checks). Use:

- `logs/execution_filter/report.json` — All instances and their p2p, p2f, f2f, f2p.
- `logs/execution_filter/report_valid.json` — Subset with non-empty `f2p`.

You can maintain an “approved” list of instance IDs or filtered JSONL and pass that into **gather_bench** (e.g. via a curated execution-filtered file that only contains verified instances).

---

## 4. Editing split (prerequisite for gather_bench)

Before **gather_bench**, the **editing_split** module must be run so that each instance has split patch history on disk (see [Editing Split](../editing_split/README.md)). Input for editing_split is the **execution-filtered** activity JSONL (or your verified subset). After splitting and validation, `gather_bench` can collect instances that have valid `work_patch_list`.

---

## 5. Gather bench (`gather_bench`)

Reads **execution-filtered** (and optionally manually verified) activity JSONL and, for each instance, loads the split result from `patch_histories/{instance_id}`. Writes bench instances (with `work_patch_list`) to a single bench JSONL file.

**Input**

- **ref-path**: Path to execution-filtered (and optionally verified) task-instances JSONL.
- **tar-path**: Output bench JSONL path (e.g. `./crawled_data/bench/owner-repo-task-instances.jsonl`).

**Commands**

```bash
# Collect all split instances from one execution-filtered file into one bench file
python -m editbench.collection.gather_bench \
  --ref-path ./crawled_data/execution_filter/owner-repo-task-instances.jsonl \
  --tar-path ./crawled_data/bench/owner-repo-task-instances.jsonl

# Overwrite target file
python -m editbench.collection.gather_bench \
  --ref-path ./crawled_data/execution_filter/owner-repo-task-instances.jsonl \
  --tar-path ./crawled_data/bench/owner-repo-task-instances.jsonl \
  --overwrite

# Restrict to specific instance IDs
python -m editbench.collection.gather_bench \
  --ref-path ./crawled_data/execution_filter/owner-repo-task-instances.jsonl \
  --tar-path ./crawled_data/bench/owner-repo-task-instances.jsonl \
  --instance-ids id1 id2 id3
```

Run this **per repository**; each repo produces one bench JSONL under `crawled_data/bench/`.

---

## 6. Merge all repos (`merge_utils`)

After you have one bench JSONL per repo in `crawled_data/bench/`, merge them into a single `all-task-instances.jsonl`.

**Commands**

```bash
# Merge all bench JSONL files in crawled_data/bench/ into all-task-instances.jsonl
python -m editbench.utils.merge_utils merge-bench

# Custom output path and/or only certain repos
python -m editbench.utils.merge_utils merge-bench \
  --output-path ./crawled_data/bench/all-task-instances.jsonl \
  --repos django/django scikit-learn/scikit-learn
```

**Infbench merge (after building infbench per condition)**  
If you later build inference variants (see [Inference](../inference/README.md)) per repo and condition, you can merge them with:

```bash
# Merge one condition (e.g. 0.2)
python -m editbench.utils.merge_utils merge-infbench --condition 0.2

# Merge all conditions
python -m editbench.utils.merge_utils merge-all-infbench
```

---

## End-to-end flow (summary)

1. **run_collection** → `path-tasks` = activity JSONL per repo.
2. **execute_filter** on each repo’s activity JSONL → Docker runs, `report.json` / `report_valid.json`.
3. **Manual verification** (paper criteria) → optional curated list or filtered JSONL.
4. **editing_split** on execution-filtered (or verified) JSONL → split + validate → `patch_histories/` populated.
5. **gather_bench** per repo → `crawled_data/bench/{repo}-task-instances.jsonl`.
6. **merge_utils merge-bench** → `crawled_data/bench/all-task-instances.jsonl`.

Then you can build infbench from the merged bench and run inference/evaluation (see [Inference](../inference/README.md) and [Evaluation](../evaluation/README.md)).
