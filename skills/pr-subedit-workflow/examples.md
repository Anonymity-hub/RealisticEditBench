# Workflow Examples

## Example 1: Single File, 4 Steps — django__django-pull-19303

```text
django__db__migrations__writer__py/
  whole-1.diff -> copy as 1.diff
  split 1.diff -> 1.diff + 2.diff + 3.diff + 4.diff
  no phase C needed because this is a single-file instance
```

| Operation | File / Action |
|-----------|---------------|
| Write `1.t.diff` for template placeholder keys | Trim and run `--step_index 2` |
| Write `2.t.diff` for population logic | Run `--step_index 3` |
| Write `3.t.diff` for concatenation order | Run `--step_index 4` |

## Example 2: Two Files, Sequential 4 Steps — django__django-pull-19214

**Phase A3 plan:**

| Global Step | File | Intent |
|-------------|------|--------|
| 1-2 | `django__contrib__admin__models__py` | API change |
| 3-4 | `django__contrib__admin__options__py` | call-site cleanup |

**Phase B:** split models first as local steps 1-2, then split options as local steps 1-2.

**Phase C:** rename options `1.diff -> 3.diff` and `2.diff -> 4.diff`.

## Example 3: Three Files, 5 Steps — matplotlib__matplotlib-pull-29716

| Global Step | File |
|-------------|------|
| 1-2 | `lib__matplotlib__axes___axes__py` |
| 3-4 | `lib__matplotlib__pyplot__py` |
| 5 | `examples/.../broken_barh.py` |

The example file participates only in step 5. After splitting, rename its local `1.diff` to `5.diff`.

## Example 4: Synchronized Step — matplotlib__matplotlib-pull-29533

Step 1 contains `1.diff` in two files at the same time:

```text
art3d.py/1.diff      -\
dviread.py/1.diff    -- global step 1; no rename needed
axes_grid.py/1..3.diff -> global steps 2-4
grid_finder.py/1..2.diff -> global steps 5-6
```

Phase C only needs to rename step numbers for axes_grid and grid_finder.

## Example 5: Multiple whole-N Files — astropy__astropy-pull-17777

```text
whole-1 ... whole-5 correspond to five commits
After cp whole-k.diff k.diff, run quick_diff again if a segment is still too large
Final 1.diff ... 5.diff match commit order one-to-one
```

## Safe quick_diff Template for a Single File

```bash
INSTANCE=matplotlib__matplotlib-pull-29716
TARGET=lib__matplotlib__axes___axes__py

# Put .t.diff only in the target directory.
# Confirm that no other subdirectory has 1.t.diff.

python -m editbench.editing_split.diff_utils quick_diff \
  --instance_id $INSTANCE \
  --step_index 2
```

## Example 6: Parallel Narratives, 3 Steps — astropy__astropy-pull-18666

**Phase A3 plan:**

| Global Step | nddata | FITS |
|-------------|--------|------|
| 1 | `nddata/1.diff` | `compressed/1.diff` |
| 2 | `nddata/2.diff` | `section/2.diff` |
| 3 | `nddata/3.diff` | `section/3.diff` |

**Phase B:** split nddata into three local steps; split the FITS line as one compressed step plus two section steps.

**Phase C:** each file's local numbering is already global. Step 1 has both lines in parallel, so no offset is needed.

**Anti-pattern:** nddata 1-3 -> compressed 4. Step 4 cannot be predicted from either narrative line, so it is not acceptable for the benchmark.

## Example 7: Same Pattern Across Files, 6 Steps — astropy__astropy-pull-18712

**Phase A3 plan:**

| Global Step | ecsv | pyarrow/csv | Phase |
|-------------|------|-------------|-------|
| 1-3 | `1-3.diff` | - | import -> doc -> `get_header_lines` |
| 4-6 | - | `4-6.diff` | same template: import -> `read_csv` -> `strip_comment_lines` |

**Phase B:** split each file locally into 1-3, one hunk or phase per step.

**Phase C:** keep ecsv as `1-3`; rename pyarrow as **`1 -> 4, 2 -> 5, 3 -> 6`**, not `1,4,5`.

**Anti-pattern:** keeping `1.diff, 4.diff, 5.diff` inside the pyarrow directory. Global step 4 cannot be inferred from ecsv steps 1-3.

## Example 8: Complex PR with Commit Chain — astropy__astropy-pull-18703

**A0:** read the root `whole.diff`: Table index serialization; four files; connect is the main area.

**A3 plan:**

| Global Step | File | Intent |
|-------------|------|--------|
| 1 | soco + sorted_array | engine `__len__`, soco `unique` |
| 2 | soco + sorted_array + index | type annotations + `SlicedIndex` len fix |
| 3 | index | `ENGINE_CLS_DEFAULT` |
| 4-23 | connect | `cp whole-N.diff` -> rename to `N+3.diff` |

**Phase B:** do not run quick_diff for connect; for small files, copy verified `whole-N.diff` to `N.diff`.

**Phase C:** connect `1 -> 4, ..., 20 -> 23`; index `1 -> 2, 2 -> 3`; soco and sorted_array local numbers are already global.

## Example 9: PLY Source + Generated Tables — astropy__astropy-pull-18776

**A0:** read the root `whole.diff`: OGIP lexer/parser refactor; three files in one narrative line.

**A3 plan:**

| Global Step | ogip.py | lextab | parsetab |
|-------------|---------|--------|----------|
| 1 | lexer | - | - |
| 2 | scale_factor + numeric_power grammar docs | sync | - |
| 3 | p_number grammar doc, old body kept temporarily | - | scale+numeric segment |
| 4 | delete signed_* + p_number implementation | - | signature + productions |
| 5 | - | - | action/goto table |

**Phase B for ogip.py:**

```bash
# 1.diff = lexer, first half of whole-1
# 2.diff = merged old 3+4: scale_factor + numeric_power docstrings
# 3.diff = p_number production docstring part from old 5
# 4.diff = delete signed_* + new body from old 5
```

**Phase C:** lextab has only `2.diff`; parsetab has `3-5.diff` aligned with ogip steps 3-4, with step 5 only for action/goto.

**Predictability:** step 1 lexer -> step 2 lextab; step 2 grammar references `number` -> step 3 adds `p_number` doc + first parsetab segments; step 3 -> step 4 cleans implementation + swaps token/number rules in parsetab; step 5 finishes LALR tables.

## Example 10: New Helper, Signature Before Body — astropy__astropy-pull-18849

**A0:** single-file `arange` refactor; nine `whole-N` files are **not reused** because they include abandoned enum work.

**A3 plan, 7 steps:**

| Step | Content |
|------|---------|
| 1 | new `@function_helper arange` signature |
| 2 | `unwrap_arange_args` stub: def + first match segment, **not pass** |
| 3 | `unwrap` function body |
| 4 | `wrap_arange_args` stub + match qty_args, without return |
| 5 | second half of `wrap`: unit, arrays, return |
| 6 | connect `arange_impl` to unwrap + error checks, old conversion may remain temporarily |
| 7 | connect wrap + `return args, kwargs` |

**Phase B notes:** do not make steps 2-3 each write a full helper in one go. Use the intermediate state plus `diff -u` to generate each step diff.

**Common mistake:** splitting three helpers into three **large** steps rather than 3 x (stub + body).

## Example 11: Code First, Branches Gradually — astropy__astropy-pull-18862

**A0:** single-file `ccddata.py`; FITS read/write support for flags; two `whole-N` files are **not reused mechanically**. Commit 2 is only doc tilde fixes plus a small reader refactor, so fold it into the final narrative.

**A3 plan, 6 steps:**

| Step | Content | Type |
|------|---------|------|
| 1 | `to_hdu` writes only ndarray path | code |
| 2 | import `FlagCollection` + FlagCollection write loop + ValueError | code |
| 3 | all docstrings: class, to_hdu, reader, writer | doc as a middle step |
| 4 | reader single extension + `flags=` | code |
| 5 | reader `flag_extensions` prefix | code |
| 6 | reader reconstructs `FlagCollection` | code |

**Phase B notes:**

- Step 1 must not be import + docs.
- Step 2 must not write both write-side branches at once, about 30 lines.
- Docs should not occupy both step 1 and step 6.
- Generate each diff from the intermediate state with `diff -u`.

## Example 12: Structure Wrapping — astropy__astropy-pull-18902

**A0:** single-file `fitswcs.py`; three helpers add the same guard. One `whole-1`, no need to reuse it directly.

**A3 plan, 4 steps:**

| Step | Content |
|------|---------|
| 1 | `redshift`: comment + bare `if not`, without `(...)`, +6 |
| 2 | `redshift`: complete `(observer is None and spectralcoord.observer is None):` + indent body |
| 3 | `beta`: full template in one step |
| 4 | `value`: full template in one step |

**Phase B notes:**

- The comment belongs with the `if not` starter in step A, not as a standalone comment step.
- Split condition `(...)` out of the guard line into step B, so the benchmark predicts the full condition in B.
- B also completes indentation; later functions can be written in one step.

## Example 13: Single-Function Mechanical Prefix — astropy__astropy-pull-19001

**A0:** single-file `spectral_coordinate.py`; one function changes from `if/elif` to `match`. One `whole-1`, no need to reuse it directly.

**A3 plan, 2 steps:**

| Step | Content |
|------|---------|
| 1 | `beta` -> `.to_value(u.dimensionless_unscaled)` + `1` -> `1.0`, +2/-2 |
| 2 | `if/elif` -> `match physical_type` + delete `else:`, +11/-15 |

**Phase B notes:**

- Step 1 is a mechanical prefix change. It changes the implied type of `beta` and hints at switching dispatch style.
- Generate step 2 directly with `diff -u`; there is no import/doc interference.

## Validation Dataset Paths

| Repository | jsonl Path |
|------------|------------|
| astropy | `./crawled_data/activity_execution/astropy-astropy-task-instances.jsonl` |
| django | `./crawled_data/activity_execution/django-django-task-instances.jsonl` |
| sympy | `./crawled_data/activity_execution/sympy-sympy-task-instances.jsonl` |
| matplotlib | `./crawled_data/activity_execution/matplotlib-matplotlib-task-instances.jsonl` |

Instance ID format: `{owner}__{repo}-pull-{number}`, for example `django__django-pull-19303`.
