# Reference Case Quick Lookup

All paths are under `patch_histories/{instance_id}/`.

## Complex PR: Read whole.diff First + Reuse a Verified Commit Chain

### astropy__astropy-pull-18703

**PR intent:** add index serialization/deserialization to the unified Table I/O interface (`write_indices`, `construct_indices`, `represent_indices`) and complete the index engine infrastructure.

**Narrative lines:**

| Line | Files | Global Steps | Notes |
|------|-------|--------------|-------|
| engine support | `soco.py`, `sorted_array.py` | 1-2 | `__len__`, `unique` property; step 2 adds type annotations |
| index infrastructure | `index.py` | 2-3 | `SlicedIndex` uses `len(index)`; `ENGINE_CLS_DEFAULT` |
| main connect feature | `connect.py` | **4-23** | keep 20 commits as-is: `whole-1...20` -> `4...23.diff` |

**Phase A3 global timeline:**

| Global Step | Files | Intent |
|-------------|-------|--------|
| 1 | soco + sorted_array | add engine `__len__`; add soco `unique` property |
| 2 | soco + sorted_array + index | type annotations; change `SlicedIndex` to `len(index)` |
| 3 | index | add `ENGINE_CLS_DEFAULT`, referenced by connect step 16 |
| 4-23 | connect | real 20-step commit chain: imports -> I/O wiring -> construct/represent iterations |

**Principle:** read the root `whole.diff` first and identify narrative lines. connect is large and has 20 verified commits, so preserve its commit steps. Smaller dependency files come before connect as steps 1-3.

## Multi-Narrative PR: Parallel Start + Aligned Continuation, 3 Steps

### astropy__astropy-pull-18666

The PR contains two **unrelated** narrative lines, nddata and FITS. They must start in **parallel at step 1** with their own `1.diff`; do not run nddata 1-3 and then start compressed at step 4.

| Global Step | nddata Line | FITS Line |
|-------------|-------------|-----------|
| 1 | `nddata/1.diff`: introduce `extracted_array_large` in `np.zeros` | `compressed/1.diff`: add `_update_header_scale_info` to `data` |
| 2 | `nddata/2.diff`: continue renaming the `fill_value` line | `section/2.diff`: first `__getitem__` cleanup |
| 3 | `nddata/3.diff`: slice assignment logic | `section/3.diff`: second cleanup |

**Principle:** predictability. Step 1 starts two independent lines at the same global time. Step 2 in each line is naturally inferred from that line's step 1. compressed has only step 1; section continues the FITS line from step 2.

## Same Pattern Across Files, Sequential Continuation, 6 Steps — astropy__astropy-pull-18712

**PR intent:** use `get_readable_fileobj` to unify compressed file reading for ECSV and PyArrow CSV, replacing hand-written `ExitStack` plus `gzip`/`bz2` handling.

**One narrative, same template across two files:** demonstrate the three phases in `ecsv.py`, then apply the same template to `pyarrow/csv.py` with continued global numbering. Do not write `1, 4, 5` inside the pyarrow directory, and do not restart from `1`.

| Global Step | `ecsv.py` | `pyarrow/csv.py` | Phase, same shape in both files |
|-------------|-----------|------------------|---------------------------------|
| 1 | `1.diff` | - | replace import, introduce `get_readable_fileobj` |
| 2 | `2.diff` | - | docstring, only present in ecsv |
| 3 | `3.diff` | - | first read entry point `get_header_lines` |
| 4 | - | `4.diff` | same phase: replace import, inferable from step 1 |
| 5 | - | `5.diff` | wrap `read_csv`, inferable from step 3 |
| 6 | - | `6.diff` | `strip_comment_lines`, inferable from step 5 |

**Phase C:** if pyarrow was locally split as `1,2,3`, rename it to **`4,5,6`**, continuing after ecsv step 3. It is not `1,4,5`.

**Anti-pattern:** pyarrow directory has `1.diff + 4.diff + 5.diff`. Step 4 cannot be inferred from the global history `1-3`, so it fails the benchmark narrative.

## Single File: Import at the End, 5 Steps

### astropy__astropy-pull-18743

**PR intent:** support masked input/output in the high-level WCS API.

**Step order: import last; split the demonstration function in half, then write the symmetric function in one step:**

| Step | Intent |
|------|--------|
| 1 | allow `MaskedNDArray` in `high_level_objects_to_values` |
| 2 | symmetric relaxation in `values_to_high_level_objects` |
| 3 | first half of `world_to_pixel`: `_get_data_and_masks` + pass `*values` into the transform |
| 4 | second half of `world_to_pixel`: wrap back with `Masked` via `combine_masks` |
| 5 | entire `pixel_to_world` segment; same pattern as 3+4, predictable after one demonstration |
| 6 | file header `from astropy.utils.masked import ...`, import at the end across multiple functions |

**Anti-pattern:** step 1 only adds the import; step 3 combines mask extraction and rewrapping in one oversized edit; after splitting steps 3/4, splitting `pixel_to_world` in half again is redundant because the pattern is already predictable.

### astropy__astropy-pull-18752, 5 Steps, Import Immediately After First Use

| Step | Intent |
|------|--------|
| 1 | `ECSVHeaderSplitter` |
| 2 | `ECSVHeaderSplitterQuoteAll`, first appearance of `csv.QUOTE_ALL` |
| 3 | `import csv`, immediately after step 2 as a quick fix |
| 4 | wire `splitter_class` |
| 5 | choose splitter conditionally in `write()` |

**Anti-pattern:** step 1 imports `csv` first, or step 5 finally imports it, too far from first use.

## Single File: 4-5 Steps

### django__django-pull-19303, 4 Steps

File: `django__db__migrations__writer__py/`

| Step | Intent |
|------|--------|
| 1 | add placeholder keys to the template dict |
| 2-3 | fill `run_before` / `atomic` logic |
| 4 | adjust template string concatenation order |

### matplotlib__matplotlib-pull-30408, 5 Steps

File: `lib__mpl_toolkits__mplot3d__axes3d__py/`

| Step | Intent |
|------|--------|
| 1 | `@_api.delete_parameter` decorator |
| 2-3 | large docstring update + remove `super().set_aspect` |
| 4-5 | default `adjustable`, validation, call |

### astropy__astropy-pull-17777, 5 Steps, Multiple whole-N Anchors

File: `astropy__io__fits__connect__py/`

Source data already has `whole-1` through `whole-5`, corresponding to five commits. Substeps `1.diff` through `5.diff` match them. Pattern: parameters/docs -> memmap branch -> pass parameter -> introduce intermediate variable -> `rstrip` implementation.

## Single File: Layered Helper, 6 Steps

### django__django-pull-19152

File: `django__db__models__expressions__py/`

Typical pattern: wire first, then fill helper:

1. Change `identity()` to call `_identity()` even though the helper does not yet exist.
2. Through 6. Fill `_identity` layer by layer: empty shell -> tuple -> dict -> Field segment -> complete version with `make_hashable`.

**Principle:** hard dependency can tolerate an intermediate non-runnable state; cognitive load rises from simple skeleton to complex implementation.

## Single File: Very Small, 2 Steps

### sympy__sympy-pull-28063

File: `sympy__physics__quantum__pauli__py/`

- Step 1: add `is_annihilation = True` to `SigmaMinus`.
- Step 2: add `is_annihilation = False` to `SigmaPlus`.

Symmetric intent; do not force this to four steps.

## Two Files: Sequential Phases, 4 Steps

### django__django-pull-19214

| Global Step | File | Intent |
|-------------|------|--------|
| 1-2 | `django__contrib__admin__models__py` | caller logic + delete `single_object` parameter |
| 3-4 | `django__contrib__admin__options__py` | delete call-site `single_object=True` |

**Principle:** update API/logic first, then update call sites.

### django__django-pull-19042

| Global Step | File | Intent |
|-------------|------|--------|
| 1-2 | `related_descriptors.py` | `filter` -> `add_q(..., reuse_all=True)` |
| 3-4 | `query.py` | add `reuse_all` to signature + implementation |

## Two Files: Implementation to Printer, 5 Steps

### sympy__sympy-pull-27482

| Global Step | File | Intent |
|-------------|------|--------|
| 1-3 | `numpy_nodes.py` | imports + convert `amin`/`amax` to `Token` structure |
| 4-5 | `numpy.py` | wire `_print_amin` / `_print_amax` |

## Three Files: Implementation to API to Example, 5 Steps

### matplotlib__matplotlib-pull-29716

| Global Step | File | Intent |
|-------------|------|--------|
| 1-2 | `lib__matplotlib__axes___axes__py` | signature + docs + `align` logic |
| 3-4 | `lib__matplotlib__pyplot__py` | public API signature + pass-through |
| 5 | `examples/.../broken_barh.py` | gallery example |

Step 1 snippet, signature + docs in a local cluster:

```diff
-    def broken_barh(self, xranges, yrange, **kwargs):
+    def broken_barh(self, xranges, yrange, align="bottom", **kwargs):
+        align : {"bottom", "center", "top"}, default: 'bottom'
```

## Four Files: Includes a Synchronized Step, 6 Steps

### matplotlib__matplotlib-pull-29533

| Global Step | File | Intent |
|-------------|------|--------|
| **1** | `art3d.py` **+** `dviread.py` | two independent small fixes, synchronized step |
| 2-4 | `axes_grid.py` | delete mgrid -> change loop -> new display logic |
| 5-6 | `grid_finder.py` | for loop to vectorized logic -> list comprehension |

**Principle:** when there is no hard cross-file dependency, different files may share global step 1.

## Early vs. Late Hunk Summary

| Phase | Typical Content |
|-------|-----------------|
| Early | signature, new parameter, docstring, decorator, class property, template placeholder |
| Middle | helper skeleton, core algorithm, branch implementation |
| Late | call-site parameter deletion/passing, printer wiring, gallery, doctest |

## Import Handling Examples

- **Default:** place the import hunk after the last logical cluster in that file, as a soft constraint.
- **Exception that becomes hard:** a `+` line references a type name that first appears only in an import hunk's `+` line, and the patch tool cannot fuzzy-apply without it.

## PLY Source File + Generated Tables — astropy__astropy-pull-18776, 5 Steps

**PR intent:** in the OGIP formatter, merge `SIGN`/`UFLOAT`/`UINT` into signed `FLOAT`/`INT`, and replace `signed_*` rules with a unified `number` rule to support signed fractional exponents.

**Single narrative line, three files with hard dependencies:** `ogip.py` source -> `ogip_lextab.py` -> `ogip_parsetab.py`.

| Global Step | ogip.py | lextab | parsetab | Intent |
|-------------|---------|--------|----------|--------|
| 1 | `1.diff` | - | - | token list + lexer rule |
| 2 | `2.diff` | `2.diff` | - | grammar docstring batch: `scale_factor` + `numeric_power` now reference `number`; regenerate lextab |
| 3 | `3.diff` | - | `3.diff` | `p_number` production docstring, `p_sign` -> `p_number`, keep old function body temporarily; sync scale+numeric parsetab segment |
| 4 | `4.diff` | - | `4.diff` | delete `p_signed_*`, change `p_number` implementation; parsetab `_lr_signature` changes token + `number` rule and `_lr_productions` |
| 5 | - | - | `5.diff` | parsetab action/goto table |

**Principles:**

- lextab is **not synchronized in step 1** with the lexer. Step 1 edits source; step 2 can predict regenerated lextab.
- Adjacent parser rule **grammar docstrings can be merged into one step** as in step 2; scale and numeric do not need separate steps.
- New rule **signature/docstring and implementation are separated**: step 3 then step 4.
- parsetab is split by parser phase into steps 3-5; do not change the entire LALR table in one step.

## New Helper: Signature Before Body — astropy__astropy-pull-18849, 7 Steps

**PR intent:** rewrite the `Quantity` `arange` helper for NumPy 2.0+: new positional signature, extract `unwrap`/`wrap`, and make impl return `args` rather than only kwargs.

**Do not reuse the nine whole-N commits**, because they include abandoned enum intermediate states. Split by final narrative and developer stopping points.

| Global Step | Intent | Prediction Basis |
|-------------|--------|------------------|
| 1 | new signatures for two `@function_helper def arange` functions | PR changes NumPy 2.0 entry point |
| 2 | `unwrap_arange_args`: def + doc + first match segment, no `pass` | step 1 changed calls; next write helper start |
| 3 | `unwrap`: match + assert + return | step 2 stub; doc already describes pattern matching |
| 4 | `wrap_arange_args`: def + assert + match qty_args, no return yet | symmetric with unwrap; first write signature and first half |
| 5 | `wrap`: unit conversion + `_quantities2arrays` + return | step 4 has qty_args, now fill second half |
| 6 | new `arange_impl` signature + call unwrap + TypeError; old qty conversion remains temporarily | first wire unwrap, wrap not yet wired |
| 7 | delete old conversion -> call `wrap_arange_args` -> `return args, kwargs` | natural finish after steps 5-6 |

**Counterexample, initial 4-step version:** steps 2-3 each wrote a whole helper, about 20-35 lines. With only the arange signature as prior context, the benchmark cannot predict a full unwrap/wrap implementation.

**Principle:** signature before body; large refactors may take **6-7** steps. Do not compress “extract three helpers” into three large steps.

## Single-Function Mechanical Micro-Steps — astropy__astropy-pull-19001, 4 Steps

**PR intent:** in `_apply_relativistic_doppler_shift`, convert `beta` to dimensionless, change `1` to `1.0`, convert `if/elif` to `match physical_type`, and remove `else:` so the `raise` is bare.

**Four-step narrative:**

| Global Step | Intent | Diff |
|-------------|--------|------|
| 1 | `beta = (velocity/c).to_value(u.dimensionless_unscaled)` | +1 -1 |
| 2 | `1` -> `1.0` in `doppler_factor` | +1 -1 |
| 3 | `if/elif` -> `match physical_type`, three cases, keep `else: raise` | +7 -10 |
| 4 | `else: raise RuntimeError(...)` -> bare `raise RuntimeError(...)` | +4 -5 |

**Prediction rhythm:** step 1 removes units -> step 2 uses float constant -> step 3 rewrites structure, predictably to `match physical_type` -> step 4 finishes match syntax by removing the redundant `else:`.

**Principle:** independent mechanical edits on the same line, such as `beta.to_value` and `1 -> 1.0`, can each take one step. `if -> match` may be split by syntax layer: first cases, then remove else. This is a structural rewrite micro-step pattern, not the §3c read/write six-step pattern.

## Control Flow Shell Before Body, Optional §3d Technique

Use this only when a hunk is too large, the step count is too low and worth micro-splitting, or the edit resembles typing in an IDE. Simple two-line logic in a small PR does not need a forced shell step.

### astropy__astropy-pull-18902 — guard, 4 Steps

**PR intent:** three spectralcoord helper functions repeat the same pattern: wrap an `if observer is None: ...` warning block in `if not (observer is None and spectralcoord.observer is None):` plus comment, so no warning appears when both observers are None.

**Four-step narrative:**

| Global Step | Function | Intent | Diff |
|-------------|----------|--------|------|
| 1 | `redshift` | comment + bare `if not`, without `(...)` condition; old body remains at old indentation | +6 |
| 2 | `redshift` | complete `(observer is None and spectralcoord.observer is None):` + indent body | +13 -13 |
| 3 | `beta` | full template: comment + guard + indentation, in one step | +18 -12 |
| 4 | `value` | full template in one step | +18 -12 |

**Prediction rhythm:** step 1 gives the explanatory comment plus `if not` starter -> step 2 predicts condition and indented body -> steps 3-4 repeat the same pattern fully.

### astropy__astropy-pull-19023 — `for`, 3 Steps

**PR intent:** in `getdata`, replace `data.dtype.names = [trans(n) for n in ...]` with explicit `for` plus `data.columns.change_name`.

**Three-step narrative; same level as the guard case, shell is the bare keyword, not the whole line:**

| Global Step | Intent | Diff |
|-------------|--------|------|
| 1 | add only bare **`for`**, old list comprehension line remains | +1 |
| 2 | complete it to **`for n in data.dtype.names:`**, still no loop body and old line remains | +1 -1 |
| 3 | add body `data.columns.change_name(n, trans(n))` and delete old `data.dtype.names = [...]` | +2 -1 |

**Counterexample, common initial error:**

| Mistake | Problem |
|---------|---------|
| step 1 = `for n in data.dtype.names:` | treats the shell as the full loop header; step 1 should contain only the word `for` |
| step 2 = loop body, step 3 = delete old line | order is close, but shell granularity is wrong; B should predict the full line header, C should predict the body |
| leftover `_old.py` / `_new.py` | remove temporary files after generating diffs |

**Principle:** use §3d only when applicable. Step A is the **minimal syntax starter**. `19025` is a small 3-step PR: first simplest path, then bare `if format.format`.

### astropy__astropy-pull-19025 — `if` branch, 3 Steps

**PR intent:** in `_convert_ascii`, split TNULL fill into string plus encode. Float columns (`DEF`) use `"nan"`; others still use `ASCIITNULL`.

| Global Step | Intent | Diff |
|-------------|--------|------|
| 1 | delete old line + update comment + `null_fill = str(ASCIITNULL)` + `encode`, simplest path with no if | +4 -4 |
| 2 | add only bare **`if format.format`**, old assignment still below | +1 |
| 3 | complete **`in "DEF":`** + `else` + `nan`/`ASCIITNULL` branches | +4 -2 |

**Counterexample, initial 2-step version:** step 2 wrote the entire if/else at once. It should be step 2 only **`if format.format`**, then step 3 completes it.

**Prediction rhythm:** step 1 decomposes null_fill -> step 2 sees the comment mention NaN/float -> step 3 predicts `in "DEF":` plus branch bodies.

## Code First, Branches Gradually — astropy__astropy-pull-18862, 6 Steps

**PR intent:** FITS read/write support for `flags` in `CCDData`. Do **not** treat this six-step table as the universal template for all multi-branch PRs. The general rule is only: simplest path first, then complex path; docs should not be first or final.

| Global Step | Intent | Prediction Basis |
|-------------|--------|------------------|
| 1 | `to_hdu`: only `np.ndarray` -> one `ImageHDU`, code | after deleting `NotImplementedError`, first write the simplest write path |
| 2 | import `FlagCollection` + `elif FlagCollection` write loop + `ValueError` | step 1 has ndarray branch; naturally add complex write path; import appears with first reference |
| 3 | all docstring updates, middle step | write-side code has landed, now add API docs |
| 4 | reader: single extension flags + pass `flags=` into constructor | after write side, add simplest read side |
| 5 | reader: `flag_extensions` prefix detection + adjust simple branch | step 4 single extension, now extend to multi-HDU naming |
| 6 | reader: `elif flag_extensions` reconstructs `FlagCollection`; final step remains code | step 5 has prefix list, now assemble Collection |

**Counterexample, initial 6-step version:**

| Mistake | Problem |
|---------|---------|
| step 1 = import + docs | step 1 has no code, so benchmark cannot predict code |
| step 2 = full `to_hdu` write path, ndarray + FlagCollection together | too large; violates gradual multi-branch rule |
| step 1 docs + step 6 docs | first and final are comments, while code is in the middle |
| import in step 1, business logic in step 2 | violates import phase rule |

---

## When Writing Skills or Splits: Do Not Turn Cases Into Universal Rules

| Mistaken Universal Rule | Correct Wording |
|-------------------------|-----------------|
| Every function must be split into setup/return | One step per **phase**; 19055 is only one prepare-then-exit example |
| Every multi-branch PR uses the 18862 six-step write/write/doc/read pattern | §3c says **simplest before complex**; the six-step plan is only for 18862 |
| Every if starts with `if format.format` | §3d is optional; the shell depends on the case: `if not`, `for`, and others |
| Every symmetric method is split in half, as in 18743 | For the same mask pipeline, demonstrate once then write the symmetric method in one step; not all symmetry needs splitting |
| Every cross-file change is import -> doc -> entry point, as in 18712 | Use continued numbering only when multiple files share the same refactor template |
| Every PLY change is lexer -> lextab -> parsetab in five steps | Only when source files and generated tables are in the same PR |
| Small PRs must have four steps | Use the smallest predictable sequence; two-line symmetric changes may be 2-3 steps |
| Normal PRs must satisfy a fixed step floor, or large PRs must stay below a fixed ceiling | There is no fixed default lower bound or upper bound; normal PRs often land around 4-5, while large refactors use more steps only when phases warrant it |
| Docs and code should be separate steps | Merge docs into code steps, as in `19064` |
| Add `pass` after an `if` shell | Shell step should be only `if self.elidable:`; fill body next, as in `20614` |

**Self-check:** if a rule written into SKILL cannot explain its applicability without mentioning an instance ID, mark it as a case or phrase it as “when ...”.

---

## Global git apply: `N.diff` Relative to Intermediate State — astropy__astropy-pull-19199

**PR intent:** add `result_type`; change `common_dtype` to delegate to it.

**Counterexample:** step 3 in `metadata/utils.py` was hand-written relative to `original`, with a hunk header containing `def common_dtype`. After steps 1-2 had already appended `result_type` at file end, strict validation failed at step 3 with `git apply`; per-file `apply_patch_batch` could still pass.

**Revision:** `3.diff = generate_diff(apply(original, 1+2), final)`, with tail context from `def result_type` and a standard hunk header.

**Principle:** after splitting, use `python scripts/pr_subedit_instance.py validate` by default for global `git apply`. `--per-file` is only a coarse check.

---

## Bare `if` Shell, No pass — django__django-pull-20614, 3 Steps

**PR intent:** add `if self.elidable: kwargs["elidable"] = self.elidable` in two `deconstruct` methods.

**Counterexample:** step 1 added `if self.elidable: pass`, then step 2 changed it to assignment. Do not add `pass`; it is unnecessary and not in the final code.

**Correct split:**

| Step | Diff | Note |
|------|------|------|
| 1 | `RunSQL`: only `+        if self.elidable:` | body comes next; intermediate state may be SyntaxError, no pass |
| 2 | same location adds `+            kwargs["elidable"] = self.elidable` | step 1 already typed the if header |
| 3 | `RunPython`: full same `if` + assignment | mirror |

**Principle:** shell steps should not add `pass` or `...` to make syntax valid. The only requirement is that the full applied chain equals final.

---

## Docs and Comments Must Not Stand Alone — astropy__astropy-pull-19064, 2 Steps

**PR intent:** relax `low_level_wcs` in two utility functions by reading `serialized_classes` via `getattr`, and update docstrings.

**Initial counterexample, 4 steps:**

| Step | Content | Problem |
|------|---------|---------|
| 1 | `high_level_objects_to_values`: `getattr` | has code |
| 2 | `values_to_high_level_objects`: `getattr` | has code |
| 3 | function 1 docstring | comment-only, no prediction/test value |
| 4 | function 2 docstring | comment-only |

**Revision, 2 steps:** split by **function**, with docs + implementation in the same diff.

| Step | Content |
|------|---------|
| 1 | `high_level_objects_to_values`: doc + `getattr` |
| 2 | `values_to_high_level_objects`: doc + `getattr` |

**Principle:** doc hunks must not occupy a standalone `N.diff`; merge them into the same function or local code cluster. A small PR with two symmetric logic edits can be **2 steps**; do not force four steps just for docs.

---

## Do Not Merge Multiple Phases Into One Step — astropy__astropy-pull-19055, 4 Steps

**PR intent:** add an `@function_helper` for `np.average`; use `_as_quantity` to split value/unit.

**Why multiple steps:** this is not a universal “setup/return” rule. It avoids putting **two phases** into one step: (A) prepare the value/unit passed to numpy, and (B) rewrite the helper's `return` tuple. After seeing A, predicting B is reasonable.

| Global Step | Phase | Diff |
|-------------|-------|------|
| 1 | remove dispatch + stub | +6 -1 |
| 2 | write a simple return first, as an exit placeholder | +12 -1 |
| 3 | preparation phase: comments + value extraction, return still simple | +10 |
| 4 | exit phase: only change `return (...)`; blank line is attached here | +8 -4 |

**Counterexample:** merge steps 3+4 into one step, creating a two-phase edit; or make step 4 only a blank line, which has no semantic value.
