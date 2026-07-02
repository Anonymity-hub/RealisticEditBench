# Editing Split Module

Split an activity’s full patch into step-wise sub-diffs, validate apply chains, and produce structured patch history for **gather_bench**. Consumes **activity-execution** JSONL (`crawled_data/activity_execution/`) and writes to `patch_histories/`.

> [!NOTE]
> This repository includes `patch_histories/` for all 712 instances. Those step-wise splits were produced by **manual expert annotation**. Use this module to validate existing splits, regenerate from activity-execution JSONL, or refine individual instances.

## Overview

1. **run_split** — Initialize split for each instance: write `whole.diff`, per-file sub-diffs (`whole-1.diff`, …), and original file snapshots; run initial validation.
2. **Split commands** (via `diff_utils`) — Manually or scriptedly refine splits (e.g. `quick_diff`, `gene`, `apply`, `diff_minus`, `trim`, etc.).
3. **validation** — Re-run patch application and collect success/failure per instance (and optionally per repo).

**Important paths (from `editing_split/constants.py`)**

- **EDITING_SPLIT_DIR** = `patch_histories/` — Per-instance split outputs: `{instance_id}/`, per-file dirs, `whole.diff`, `{step}.diff`, `original.*`, `final.*`.
- **REPO_AND_LOG_DIR** = `tmp/` — Cloned repos and validation logs (e.g. `tmp/{owner__repo}/testbed`, `tmp/{owner__repo}/pull-{num}/validation.log`).

---

## 1. Initial run: `run_split`

Reads **activity-execution** JSONL and, for each instance, writes the full patch and per-file sub-diffs under `patch_histories/{instance_id}/`, fetches original files at `base_commit`, and runs one validation pass.

**Input:** Path to activity-execution task-instances JSONL (e.g. `./crawled_data/activity_execution/owner-repo-task-instances.jsonl`), or a repo name resolved under that directory.

**Commands**

```bash
# Run on one repo (resolved to crawled_data/activity_execution/astropy-astropy-task-instances.jsonl)
python -m editbench.editing_split.run_split \
  --dataset-name astropy/astropy

# Or pass a JSONL path directly
python -m editbench.editing_split.run_split \
  --dataset-name ./crawled_data/activity_execution/astropy-astropy-task-instances.jsonl

# With custom time window (YYYYMMDD): only instances with created_at >= that date
python -m editbench.editing_split.run_split \
  --dataset-name astropy/astropy \
  --time-window 20240101

# Restrict to specific instance IDs
python -m editbench.editing_split.run_split \
  --dataset-name astropy/astropy \
  --instance-ids astropy__astropy-pull-123 astropy__astropy-pull-456
```

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

**Input:** Same activity-execution (or split-ready) task-instances JSONL path.

**Commands**

```bash
# Validate all instances in the dataset
python -m editbench.editing_split.validation \
  --dataset_name ./crawled_data/activity_execution/owner-repo-task-instances.jsonl

# Restrict to specific instance IDs
python -m editbench.editing_split.validation \
  --dataset_name ./crawled_data/activity_execution/owner-repo-task-instances.jsonl \
  --instance_ids astropy__astropy-pull-123 astropy__astropy-pull-456
```

**Output**

- Per-instance: `tmp/{owner__repo}/pull-{num}/validation.log`, `apply.sh`.
- Console summary: success vs fail sets (and optionally first 10 IDs). No separate JSON output; use logs to fix failing instances and re-run.

**Naming convention for patch history**

- `validation.load_patch_list_instance()` (and thus **gather_bench**) looks for files under `patch_histories/{instance_id}/{work_file_str}/` whose names **start with digits and end with `diff`** (e.g. `1.diff`, `2.diff`, `001_step.diff`), ordered by numeric prefix. Ensure your split workflow writes such files so that bench instances get a non-empty `work_patch_list`.

---

## 4. PR sub-edit helper (`scripts/pr_subedit_instance.py`)

> [!IMPORTANT]
> The benchmark's `patch_histories/` were **manually decomposed by experts**. The `skills/` directory and this script are **reference tooling** for splitting or validating PR sub-edits—they document principles we found useful, but they **did not produce** the released 712-instance dataset. Use them when exploring splits, curating new data, or checking apply chains—not as a description of how RealisticEditBench was built.

For optional manual or agent-assisted refinement of step-wise splits, the helper script at the repo root supports status checks, strict validation (global `git apply` on the testbed), per-file validation, prompt printing, and optional `cursor agent` batch runs.

**Reference skills** (for agents or human annotators): `skills/pr-subedit-principles/` and `skills/pr-subedit-workflow/`.

**Commands**

```bash
# Instance status (strict validate when activity_execution jsonl is available)
python scripts/pr_subedit_instance.py status astropy__astropy-pull-19055

# Strict validation (default): global step order + git apply on testbed
python scripts/pr_subedit_instance.py validate astropy__astropy-pull-19055

# Per-file apply_patch_batch only (fast, no cross-file step conflicts)
python scripts/pr_subedit_instance.py validate astropy__astropy-pull-19055 --per-file

# Print the standard Cursor Agent prompt
python scripts/pr_subedit_instance.py prompt astropy__astropy-pull-19055

# Run cursor agent for one instance (then validate)
python scripts/pr_subedit_instance.py agent astropy__astropy-pull-19055
```

By default, strict validation scans `crawled_data/activity_execution/*-task-instances.jsonl` for the instance. Pass `--dataset path/to.jsonl` to override.

---

## Recommended workflow

1. **run_split** on activity-execution JSONL → initial `patch_histories/` and one validation pass.
2. Use **diff_utils** subcommands and/or **pr_subedit_instance** (with reference `skills/`) to adjust or add steps when building new data—not required for using the released benchmark.
3. **validation** or `scripts/pr_subedit_instance.py validate` on the same JSONL → confirm all instances apply cleanly; fix any that fail.
4. Run **gather_bench** (see [Collection](../collection/README.md)) so that only instances with valid `work_patch_list` are written to bench JSONL.

After that, you can build infbench and run inference/evaluation (see [Inference](../inference/README.md) and [Evaluation](../evaluation/README.md)).
