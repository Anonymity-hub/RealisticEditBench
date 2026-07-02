---
name: pr-subedit-principles
description: Split a PR-level whole.diff in patch_histories into a developer-like, predictable, and verifiable sub-edit sequence. Treat whole.diff as authoritative, use whole-N only as reference, plan global step numbers, prefer code-first edits, split structure before body, reject invalid doc/format-only steps, and choose step counts by semantic predictability rather than a fixed lower bound.
---

# PR Sub-Edit Splitting Principles

This skill teaches an agent how to split a coarse PR diff into `1.diff`, `2.diff`, and later sub-edits. The goal is not to divide lines evenly. The goal is to produce a sequence that resembles a real developer's editing process: each step should follow naturally from prior steps, and applying the full chain should reproduce the final PR.

## 0. Execution Order Overview

When splitting, reason in this order. Do not process files mechanically by directory order.

```text
1. Confirm hard constraints: authoritative whole.diff, N.diff base state, final validation.
2. Understand the PR: read the root whole.diff, identify narrative lines, dependencies, and repeated patterns.
3. Plan global step numbers: decide which files proceed in parallel, which continue a template, and which must be ordered.
4. Split single-file hunks: keep only one predictable editing phase per step.
5. Check anti-patterns: invalid steps, oversized steps, wrong patch bases, and step-number conflicts.
```

## 1. Hard Constraints First

### 1.1 `whole.diff` Is Authoritative

For each file directory, `{file}/whole.diff` represents the change from `base_commit` to the final file state. It is the only authoritative input for splitting and validation.

| File | Purpose | Authoritative? |
|------|---------|----------------|
| `{file}/whole.diff` | File state from `base_commit` to final PR state | Yes |
| `whole-{n}.diff` | Commit-chain fragment, relative to each commit parent | No |
| `work_patch_list` | Reconstructed GitHub commit chain | No |

Do not assume `whole-N.diff` files compose to `whole.diff`. You may directly reuse them only after verifying:

```text
apply(original, whole-1.diff, ..., whole-N.diff) == apply(original, whole.diff)
```

If validation fails, use the commit order only to understand the intended narrative. The actual hunks must be split from authoritative `whole.diff`, or rebased onto the correct intermediate state.

### 1.2 `N.diff` Must Use the Global Intermediate State

In multi-file instances, global step `N` means all edits that happen at the same time point. Every `N.diff` must apply on top of the repository state after global steps `1..N-1`.

Do not treat `N.diff` as the Nth patch relative to `original.*`. Prefer one of these generation methods:

- Use `quick_diff --step_index N`, which resolves the correct base automatically.
- Or apply `1..N-1` to obtain the intermediate state, then use `generate_diff` against the target state.

Avoid hand-written nonstandard hunk headers such as `@@ ... @@ def foo():`; they often cause `git apply` to report `corrupt patch`.

### 1.3 Strict Validation Must Pass

The final chain must satisfy:

```text
apply(original, 1.diff, 2.diff, ..., k.diff) == apply(original, whole.diff)
```

Strict validation is based on global `git apply`. Per-file `apply_patch_batch` is only a coarse check. Intermediate states may be syntactically incomplete. Do not add `pass`, `...`, or other placeholder code that does not appear in the final file just to make an intermediate state runnable.

## 2. Decide the Split in This Order

### 2.1 Identify Narrative Lines and Dependencies First

After reading the PR-level `whole.diff`, answer these questions:

- What problem does the PR solve?
- How many unrelated narrative lines are present?
- Which files have hard definition-to-use dependencies?
- Which files repeat the same editing template?
- Are there generated files that must lag behind source-file edits?

### 2.2 Plan Global Step Numbers Next

Plan the global timeline before splitting individual files in detail. Follow these rules:

- Unrelated narratives start in parallel: each line begins at global `1.diff`.
- Hard dependencies are ordered: definitions, helpers, and interface changes come before call sites.
- Repeated patterns continue sequentially: the demonstration file runs through `1..k`, then later files continue at `k+1..`.
- Generated files lag behind source files: update source files first, then `*_lextab.py`, `*_parsetab.py`, or similar generated outputs.

### 2.3 Then Split Single-File Hunks

Within a file, each step should keep one predictable editing phase. Prefer natural stopping points:

- New helper: prepare local values first, then update the return or call-site exit.
- New function: add the signature or first real body fragment, then fill the body.
- New loop: introduce the loop or collect data first, then update inner use.
- Large hunk: split around comment blocks, branches, returns, and registration points.
- Repeated multi-file pattern: split the first file finely to demonstrate the template, then continue later files using the same template.

### 2.4 Finally Check Predictability

For every step, ask: given `1..N-1`, is step `N` the natural next edit? If not, adjust ordering or split the step further.

## 3. Four Splitting Principles

Use these four principles when ordering or granularity decisions conflict.

### 3.1 Context Coherence Principle

Each sub-edit should preserve local code context. By default, keep consecutive edits within the same file, function, or nearby code region, and reduce cross-file, cross-function, or long-distance jumps. This matches normal cursor movement and helps the model retain useful context from prior edits.

### 3.2 Atomic Intent Principle

Each sub-edit should represent one core editing intent, not a mechanical line-count slice. Consecutive changes in the same region and serving the same purpose may be merged. If one diff contains multiple independent purposes, such as preparing data and then changing return logic, split them into separate sub-edits.

### 3.3 Strong Logical Dependency Principle

Later sub-edits should rely on or build upon earlier ones. When adding a function, class, helper, interface, or variable, define it before using it. Call sites, registrations, and complex implementations should appear after the corresponding structure or symbol exists.

### 3.4 Cognitive Load Stratification Principle

The sequence should move from lower cognitive load to higher cognitive load. Earlier steps usually contain mechanical, local, or skeletal edits such as declaring variables, adding function signatures, building structural shells, or making simple replacements. Later steps fill in complex branches, variable interactions, edge cases, core algorithms, and parameter-dependent logic.

## 4. Step Count and Granularity

### 4.1 Step Count Guidance

Step count is a descriptive outcome, not a fixed target. Choose the shortest sequence that preserves semantic predictability and avoids oversized or multi-phase steps.

| PR Type | Suggested Step Count |
|---------|----------------------|
| PR-level `whole.diff` is only a few lines | 2-3 steps; use 4 only if there are distinct semantic phases |
| Very small symmetric change | 2-3 steps; do not pad mechanically |
| Normal single- or two-file PR | commonly 3-5 steps; 4 is a normal center point |
| Moderate multi-file change | commonly 4-6 steps, depending on dependencies and repeated patterns |
| Large refactor, multiple helpers, verified long commit chain | 6+ steps only when the edit phases truly warrant it |

### 4.2 Signals That a Step Should Be Split Further

If any of these are true, prefer adding another step:

- The step adds more than about 15 net lines.
- The step fills an entire new function, class, multi-branch block, or large loop.
- The step contains two natural phases.
- The previous step established a pattern that the next step could continue in a smaller piece.

Do not create meaningless fragments just to reach a target count. A step that only changes blank lines, formatting, comments, or docs should be merged into a neighboring code step.

## 5. Single-Step Content Rules

### 5.1 Step 1 Must Be a Semantic Code Edit

Step 1 must be a semantic code change, such as an implementation replacement, a new branch, a call-site update, or a core hunk. These must not appear alone as step 1:

- import;
- docstring, parameter text, or comment;
- pure formatting or blank lines;
- low-value cleanup.

The final step should also be a testable or checkable semantic code change. Pure unused-import removal, doc adjustment, or formatting cleanup should not be the final step by itself.

### 5.2 Place Imports Near First Use

Imports are not meaningless, but they usually should not come first. Default behavior:

- First write the business-code use of the new symbol.
- Add the corresponding import in the next step.
- If the diff cannot be naturally split without the import, or the step must remain runnable, include the import in the first-use step.

Pure import cleanup may be a middle cleanup step or merged into a nearby code step. It should not be the first or final step.

### 5.3 Merge Docs and Comments Into Code Steps

Docstrings, comments, parameter descriptions, tilde formatting fixes, and similar text-only changes should not stand alone. Merge them into a nearby code step in the same function, class, or editing cluster.

Acceptable: code change plus related doc update in the same `N.diff`.

Not acceptable:

```text
1.diff  docs only
2.diff  code
3.diff  comments only
```

### 5.4 Structure Before Body

When adding or rewriting a `class`, `def`, method, solver, handler, or adapter, do not write the entire structure in one step. Recommended split:

1. Add the signature, class shell, class attributes, or first real body fragment.
2. Fill validation, local variables, branches, loops, and core computation.
3. Add returns, registrations, call sites, or import wiring.

Do not add placeholder `pass` or `...` for syntax only. If those statements are not in the final file, they should not appear in any sub-edit.

### 5.5 Grow Multi-Branch Logic Gradually

When adding multiple paths inside an existing function, write the simplest path first, then add more complex branches. Do not write a full `if/elif/else`, both read and write sides, or multiple type cases in one step.

Control-flow shells are optional. Use them only when the logic is large enough to deserve fine splitting and the edit resembles real typing behavior:

| Case | Possible Split |
|------|----------------|
| Guard wraps a large old block | Write `if not` first, then complete the condition and indented body |
| `for` replaces one old line | Write `for`, then complete the loop header, then add the body and delete the old line |
| New large `if/else` branch | Write the minimal condition shell, then fill branch bodies |

Do not force shell steps for small blocks. A one- or two-line change such as `if x: return y` can be one step.

## 6. Multi-File Step Numbering Rules

### 6.1 Unrelated Narratives Start in Parallel

If the PR contains unrelated changes, each line should start at global `1.diff`. Do not finish file A as `1..3` and then start unrelated file B at `4.diff`.

```text
file_a/: 1.diff, 2.diff, 3.diff
file_b/: 1.diff, 2.diff
```

### 6.2 Repeated Patterns Continue Sequentially

If multiple files apply the same refactor template, later files should not restart at `1.diff` and should not run in parallel with the demonstration file. Let the first file demonstrate the full pattern, then continue global numbering in the next file.

```text
file_a/: 1.diff, 2.diff, 3.diff
file_b/: 4.diff, 5.diff, 6.diff
```

Typical cases: two serializer files adding the same guard/body/return pattern; two reader backends making the same import, entry-point replacement, and helper call.

### 6.3 Dependency Order

Definitions, helpers, and interface changes should come before call sites. If a file uses a symbol or behavior introduced by an earlier step, the use-site file must come later.

```text
models/: 1.diff, 2.diff
options/: 3.diff, 4.diff
```

### 6.4 Generated Files Lag Behind

Change source files first, then generated tables or outputs. Update lexer changes before `*_lextab.py`; update parser grammar or docstrings before `*_parsetab.py`. Generated files should not be synchronized with the first source-file step by default.

## 7. Common Anti-Patterns

| Anti-Pattern | Correct Handling |
|--------------|------------------|
| Starting with import, docs, comments, or formatting | Start with a semantic code edit |
| Final step is only cleanup or docs | Merge into an earlier semantic step, or reorder so the final step is code |
| One step fills an entire new def/class | Split into signature or first body fragment, then body |
| Adding `pass` or `...` for an intermediate state | Allow incomplete intermediate syntax; do not add placeholders outside the final code |
| Repeated-pattern files restart from `1.diff` in parallel | Finish the demonstration file first, then continue numbering |
| Hand-writing `N.diff` relative to `original.*` | Generate it from the state after global `1..N-1` |
| Only running per-file validation | Run strict validation |

## 8. Checklist

Before splitting:

```text
- [ ] Read the PR-level whole.diff and summarize the PR intent.
- [ ] Confirm each file uses {file}/whole.diff as authoritative.
- [ ] If reusing whole-N, verify that its composition equals whole.diff.
- [ ] Identify unrelated narratives, repeated patterns, and hard dependencies.
- [ ] Plan global step numbers instead of relying on local file numbering.
```

During splitting:

```text
- [ ] Step 1 is a semantic code edit.
- [ ] Imports are not placed first without reason; they stay near first use.
- [ ] Docs, comments, and formatting do not stand alone.
- [ ] New def/class edits are not filled all at once and do not use pass/ellipsis placeholders.
- [ ] No step contains two naturally separable phases.
- [ ] Steps larger than about 15 lines have been reconsidered for further splitting.
- [ ] Repeated-pattern files use continued numbering rather than parallel restart.
```

Before validation:

```text
- [ ] No leftover *.t.diff, _old.*, or _new.* files.
- [ ] Numeric diff names match the global timeline.
- [ ] Every N.diff applies after the global N-1 state.
- [ ] Strict validation passes.
- [ ] Manual predictability check: after reading 1..N-1, N is the natural next edit.
```

## 9. Reference Cases

Consult cases only when a concrete pattern is needed. Do not mechanically apply one case to every PR.

Prioritize: `astropy__astropy-pull-19199`, `astropy__astropy-pull-19064`, `astropy__astropy-pull-19055`, `astropy__astropy-pull-19025`, `astropy__astropy-pull-19023`, `astropy__astropy-pull-19001`, `astropy__astropy-pull-18902`, `astropy__astropy-pull-18862`, `astropy__astropy-pull-18849`, `astropy__astropy-pull-18776`, `astropy__astropy-pull-18743`, `astropy__astropy-pull-18712`, `astropy__astropy-pull-18703`, `astropy__astropy-pull-18666`, `astropy__astropy-pull-17777`, `django__django-pull-19303`, `django__django-pull-19214`, `django__django-pull-19152`, `sympy__sympy-pull-28805`, `matplotlib__matplotlib-pull-30746`, `matplotlib__matplotlib-pull-29716`, `matplotlib__matplotlib-pull-29533`.
