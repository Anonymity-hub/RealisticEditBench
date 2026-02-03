# Editing Split Module

Tools to split an activity’s full patch into step-wise sub-diffs, validate that they apply correctly, and expose structured patch history for the benchmark. The **editing_split** step is required before **gather_bench**: it consumes **execution-filter** JSONL and produces per-instance split data under `patch_histories/`.

## Overview

1. **run_split** — Initialize split for each instance: write `whole.diff`, per-file sub-diffs (`whole-1.diff`, …), and original file snapshots; run initial validation.
2. **Split commands** (via `diff_utils`) — Manually or scriptedly refine splits (e.g. `quick_diff`, `gene`, `apply`, `diff_minus`, `trim`, etc.).
3. **validation** — Re-run patch application and collect success/failure per instance (and optionally per repo).

**Important paths (from `editing_split/constants.py`)**

- **EDITING_SPLIT_DIR** = `patch_histories/` — Per-instance split outputs: `{instance_id}/`, per-file dirs, `whole.diff`, `{step}.diff`, `original.*`, `final.*`.
- **REPO_AND_LOG_DIR** = `tmp/` — Cloned repos and validation logs (e.g. `tmp/{owner__repo}/testbed`, `tmp/{owner__repo}/pull-{num}/validation.log`).

---

## 1. Initial run: `run_split`

Reads **execution-filter** activity JSONL and, for each instance, writes the full patch and per-file sub-diffs under `patch_histories/{instance_id}/`, fetches original files at `base_commit`, and runs one validation pass.

**Input:** Path to execution-filtered task-instances JSONL (e.g. `./crawled_data/execution_filter/owner-repo-task-instances.jsonl`).

**Commands**

```bash
# Run on one repo’s execution-filter output (default time_window = 20241201)
python -m editbench.editing_split.run_split \
  ./crawled_data/execution_filter/astropy-astropy-task-instances.jsonl

# With custom time window (YYYYMMDD): only instances with created_at >= that date
python -m editbench.editing_split.run_split \
  ./crawled_data/execution_filter/owner-repo-task-instances.jsonl \
  --time-window 20240101

# Restrict to specific instance IDs (when run as script, you can pass instance_ids via main())
# From CLI you may need to patch run_split to accept --instance-ids; otherwise run on full dataset.
```

**Note:** `run_split`’s `main()` in the script block uses a hardcoded list of `dataset_names` and `SRC_EXECUTION_FILTER_DATA` paths. For a single file, call the module with that file path as the first argument, or adjust the `if __name__ == "__main__"` block to accept a path and optional `--instance-ids`.

**Output under `patch_histories/{instance_id}/`**

- `whole.diff` — Full combined patch.
- For each `file_work`: a directory `{file_path_sanitized}/` with:
  - `whole-1.diff`, `whole-2.diff`, … — Sub-diffs per step for that file.
  - `original.{ext}` — File content at `base_commit`.
- Validation is run once per instance; logs under `tmp/`.

---

## 2. Split commands (`diff_utils`)

All of these are invoked as subcommands of `editbench.editing_split.diff_utils`. Use them to generate, apply, or refine diffs (e.g. to produce step-wise `1.diff`, `2.diff`, … that `validation` and `load_patch_list_instance` expect).

### 2.1 `gene` — Generate git-style diff between two files

```bash
python -m editbench.editing_split.diff_utils gene \
  --filename path/to/foo.py \
  --file1 old.py \
  --file2 new.py \
  [--res out.diff]
```

### 2.2 `apply` — Apply a single diff to a file

```bash
python -m editbench.editing_split.diff_utils apply \
  --file base.txt \
  --diff patch.diff \
  [--fuzz 1] [--strip 1] \
  [--res patched.txt]
```

### 2.3 `batch_apply` — Apply multiple diffs in order

```bash
python -m editbench.editing_split.diff_utils batch_apply \
  --file base.txt \
  --diffs 1.diff 2.diff \
  [--fuzz 1] [--strip 0] \
  [--res out.txt]
```

### 2.4 `diff_minus` — Diff between “mid” and “final” (e.g. sub-diff = full − previous)

```bash
python -m editbench.editing_split.diff_utils diff_minus \
  --filename x.py \
  --file orig.py \
  --diff1 sub.diff \
  --diff2 full.diff \
  [--res result.diff]
```

### 2.5 `quick_diff` — Generate next step sub-diff for one instance (under EDITING_SPLIT_DIR)

```bash
python -m editbench.editing_split.diff_utils quick_diff \
  --instance_id astropy__astropy-pull-123 \
  --step_index 2
```

Used to build step-wise diffs (e.g. `2.diff` from `1.diff` and `whole-1.diff`). Repeat for later steps by incrementing `step_index`.

### 2.6 `trim` — Trim diff hunks to limited context lines

```bash
python -m editbench.editing_split.diff_utils trim \
  --input in.diff \
  [--output out.diff] \
  [--context 3]
```

---

## 3. Validation (`validation`)

Re-runs the apply pipeline for each instance: clone (if needed), checkout `base_commit` for work files, apply each step’s patches in order, and check for apply errors. Results are summarized per repo.

**Input:** Same execution-filtered (or split-ready) task-instances JSONL path.

**Commands**

```bash
# Validate all instances in the dataset
python -m editbench.editing_split.validation \
  --dataset_name ./crawled_data/execution_filter/owner-repo-task-instances.jsonl

# Restrict to specific instance IDs
python -m editbench.editing_split.validation \
  --dataset_name ./crawled_data/execution_filter/owner-repo-task-instances.jsonl \
  --instance_ids astropy__astropy-pull-123 astropy__astropy-pull-456
```

**Output**

- Per-instance: `tmp/{owner__repo}/pull-{num}/validation.log`, `apply.sh`.
- Console summary: success vs fail sets (and optionally first 10 IDs). No separate JSON output; use logs to fix failing instances and re-run.

**Naming convention for patch history**

- `validation.load_patch_list_instance()` (and thus **gather_bench**) looks for files under `patch_histories/{instance_id}/{work_file_str}/` whose names **start with digits and end with `diff`** (e.g. `1.diff`, `2.diff`, `001_step.diff`), ordered by numeric prefix. Ensure your split workflow writes such files so that bench instances get a non-empty `work_patch_list`.

---

## Recommended workflow

1. **run_split** on execution-filter JSONL → initial `patch_histories/` and one validation pass.
2. Use **diff_utils** subcommands as needed to adjust or add steps (e.g. `quick_diff` for step 2, 3, …; `gene`/`apply`/`diff_minus` for custom splits).
3. **validation** on the same JSONL → confirm all instances apply cleanly; fix any that fail.
4. Run **gather_bench** (see [Collection](../collection/README.md)) so that only instances with valid `work_patch_list` are written to bench JSONL.

After that, you can build infbench and run inference/evaluation (see [Inference](../inference/README.md) and [Evaluation](../evaluation/README.md)).
