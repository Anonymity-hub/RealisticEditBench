---
name: pr-subedit-workflow
description: Automatically split a PR instance under patch_histories from whole.diff into an ordered sequence of sub-edits, such as 1.diff and 2.diff. Use this when the user asks to split patch_histories tasks, run the sub-edit workflow, or mentions run_split, quick_diff, or validation.
---

# PR Sub-Edit Splitting Workflow

Goal: convert `patch_histories/{instance_id}/` from a PR-level `whole.diff` into a verifiable and predictable sub-edit sequence: `1.diff`, `2.diff`, and later steps. For splitting principles, see [pr-subedit-principles](../pr-subedit-principles/SKILL.md).

There are three core constraints:

- Treat each file directory's `whole.diff` as authoritative; `whole-N.diff` is only commit-chain reference.
- Choose the shortest sequence that preserves semantic predictability; normal PRs often land around 4-5 steps, while 6+ is reserved for genuinely large refactors, long verified commit chains, or clearly separable edit phases.
- Every `N.diff` must apply after global step `N-1`, and the final result must match `final.*`.

## Directory Prerequisites

`run_split` has already produced this structure:

```text
patch_histories/{instance_id}/
  whole.diff
  {file_sanitized}/
    original.{ext}
    final.{ext}
    whole.diff          # authoritative: base_commit to final state
    whole-1.diff ...    # reference: commit-chain fragments, not guaranteed to compose
```

Strict validation usually uses:

```bash
python scripts/pr_subedit_instance.py validate {instance_id}
```

## A. Read the Global Diff and Plan

Read `patch_histories/{instance_id}/whole.diff` before entering individual file directories. Determine:

- what problem the PR solves;
- how many narrative lines exist, and which files depend on each other;
- which files repeat the same pattern, and which changes are unrelated parallel edits;
- whether `whole-N.diff` commit references are present.

Write a global timeline before splitting:

```markdown
| Global Step | File | Intent |
|-------------|------|--------|
| 1 | file_a | Add the minimal semantic code change first |
| 2 | file_a | Continue the same branch or add import |
| 3 | file_b | Update a dependent call site |
```

Step-numbering rules:

- Unrelated narrative lines start in parallel at the same `1.diff`; later steps continue independently.
- When definition and use have a dependency, the definition comes first.
- Do not split repeated-pattern files in parallel. Let the demonstration file run through `1..k`, then continue later files at `k+1..`.
- Generated files lag behind source files: edit lexer/parser sources first, then update `*_lextab.py`, `*_parsetab.py`, or similar generated files.

## B. Initialize File Splitting

For each changed file, start from authoritative `whole.diff` by default:

```bash
cp patch_histories/{instance_id}/{file}/whole.diff \
   patch_histories/{instance_id}/{file}/1.diff
```

If `whole-1.diff`, `whole-2.diff`, and later commit references exist, run compose validation first. Reuse `whole-N.diff` directly only when validation passes and the commit chain is semantically coherent.

Run this inside the file directory:

```bash
python3 - <<'PY'
from pathlib import Path
from editbench.editing_split.diff_utils import apply_diff_with_file, apply_patch_batch

d = Path('.')
orig = next(d.glob('original.*'))
whole = apply_diff_with_file(orig, d / 'whole.diff')
parts = sorted(d.glob('whole-*.diff'), key=lambda p: int(p.stem.split('-')[1]))
seq = apply_patch_batch(orig, parts)
assert seq == whole, 'whole-N chain != whole.diff; split from whole.diff instead'
print('OK: whole-N chain matches whole.diff')
PY
```

If validation fails, do not copy `whole-N.diff`. You may use commit order to understand the narrative, but actual hunks must be split from `whole.diff` or rebased onto the correct intermediate state.

## C. Use quick_diff to Split One File Iteratively

`quick_diff` works as follows: manually shrink the current complete diff into `{N-1}.t.diff`; the tool saves that as `{N-1}.diff` and writes the remaining changes into `{N}.diff`.

Command:

```bash
python -m editbench.editing_split.diff_utils quick_diff \
  --instance_id {instance_id} \
  --step_index {N}
```

To generate step 2:

1. Confirm `{file}/1.diff` is the current complete segment.
2. Keep only the hunks that belong in step 1 and save them as `1.t.diff`.
3. Ensure only this file directory in the instance contains `1.t.diff`.
4. Run `quick_diff --step_index 2`.
5. Check that `1.diff + 2.diff` still covers the original `whole.diff`.

To generate step 3, repeat the same process on `2.diff`, save `2.t.diff`, and run `--step_index 3`.

While splitting, prioritize these checks:

- Step 1 must be a semantic code edit, not import-only, docs-only, comments-only, or formatting-only.
- New `def`, class, or method edits should not be filled all at once; start with the signature or first real body fragment, then fill the body.
- Place imports after first use when possible; merge them into the first-use step only when necessary.
- Docs and comments should not stand alone, especially not as the first or final step.
- If a step adds more than about 15 net lines or contains two natural phases, split it further.
- Intermediate states may be syntactically incomplete; do not add `pass` or `...` that is absent from the final file.

`quick_diff` scans all `{N-1}.t.diff` files under the instance. Keep a `.t.diff` in only one file directory at a time to avoid accidentally processing other files.

## D. Align Global Step Numbers

After each file is split, local numbers may need to be renamed into global step numbers.

Common patterns:

```text
# Unrelated narratives in parallel
file_a/: 1.diff, 2.diff
file_b/: 1.diff

# Repeated pattern; later file continues numbering
file_a/: 1.diff, 2.diff, 3.diff
file_b/: 4.diff, 5.diff, 6.diff

# Definition to use
models/: 1.diff, 2.diff
options/: 3.diff, 4.diff
```

When renaming, move from larger numbers to smaller numbers to avoid overwrites. Keep `whole.diff`, `whole-N.diff`, `original.*`, and `final.*`. Remove temporary files such as `*.t.diff`, `_old.*`, and `_new.*`.

## E. Run Strict Validation

Preferred command:

```bash
python scripts/pr_subedit_instance.py validate {instance_id}
```

This applies diffs by global step number in the testbed using `git apply`, then compares final files against `final.*`.

A coarse per-file check is available:

```bash
python scripts/pr_subedit_instance.py validate {instance_id} --per-file
```

However, `--per-file` does not replace strict validation because it cannot catch cross-file global-step conflicts.

Passing criteria:

- Every changed file has correctly numbered numeric diffs matching the global timeline.
- Strict validation reports no `patch failed`, `corrupt patch`, or `version error`.
- Applying the full chain produces files identical to all `final.*` files.
- No `.t.diff` or temporary diff-generation files remain.

## Common Failures

| Symptom | Fix |
|---------|-----|
| `whole-N` does not apply on `original.*` | Expected in some cases; split from authoritative `whole.diff` |
| `whole-1...whole-N` does not equal `whole.diff` | Do not copy `whole-N.diff`; re-split from `whole.diff` |
| Per-file validation passes but strict validation fails | Some `N.diff` was not generated from the global `N-1` state; regenerate from the intermediate state |
| `corrupt patch` | Check hunk headers, context, and final newline |
| Final result differs from `final.*` | A hunk is missing, duplicated, or overwritten by step numbering |

## Deliverable Checklist

```text
patch_histories/{instance_id}/
  {file}/
    original.{ext}     keep
    final.{ext}        keep
    whole.diff         keep; authoritative anchor
    whole-{n}.diff     keep; commit reference
    1.diff ... k.diff  globally numbered and strictly validatable
```

Finally, perform a manual predictability check: given `1..N-1`, is step `N` a natural next edit? If not, adjust the narrative order or split the step further.
