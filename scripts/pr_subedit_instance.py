#!/usr/bin/env python3
"""
PR sub-edit splitting helper: validate apply chains, print agent prompts, optionally run cursor agent.

Usage:
  # Inspect instance status
  python scripts/pr_subedit_instance.py status astropy__astropy-pull-19055

  # Strict validation (default): global step order + git apply on testbed
  python scripts/pr_subedit_instance.py validate astropy__astropy-pull-19055

  # Per-file apply_patch_batch only (fast, but no cross-file step conflicts)
  python scripts/pr_subedit_instance.py validate astropy__astropy-pull-19055 --per-file

  # Point to a specific jsonl (default: scan crawled_data/activity_execution/*-task-instances.jsonl)
  python scripts/pr_subedit_instance.py validate INSTANCE --dataset path/to.jsonl

  # Print the standard Cursor Agent prompt
  python scripts/pr_subedit_instance.py prompt astropy__astropy-pull-19055

  # Run cursor agent for one instance
  python scripts/pr_subedit_instance.py agent astropy__astropy-pull-19055

  # Batch agent (sequential; one instance at a time)
  python scripts/pr_subedit_instance.py agent-batch \
    --instance-ids astropy__astropy-pull-19064 astropy__astropy-pull-19199

  # Read instance IDs from a file (one per line; lines starting with # are ignored)
  python scripts/pr_subedit_instance.py agent-batch --from-file ids.txt

Note: do not repeatedly click IDE Run on this script; agent-batch launches heavyweight cursor agent processes.
Only one batch should run at a time (the script uses a flock lock).
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from editbench.editing_split.constants import EDITING_SPLIT_DIR
from editbench.editing_split.diff_utils import apply_patch_batch
from editbench.editing_split.validation import (
    find_activity_instance,
    validate_patch_history_strict,
)

DIFF_STEP_RE = re.compile(r"^(\d+)\.diff$")
BATCH_LOCK_PATH = REPO_ROOT / "tmp" / "pr_subedit_agent_batch.lock"


@contextmanager
def batch_run_lock(*, force: bool = False):
    """Ensure only one agent-batch runs at a time in this repo."""
    BATCH_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(BATCH_LOCK_PATH, "a+", encoding="utf-8")
    try:
        flags = fcntl.LOCK_EX
        if not force:
            flags |= fcntl.LOCK_NB
        fcntl.flock(lock_file.fileno(), flags)
    except BlockingIOError:
        lock_file.seek(0)
        holder = lock_file.read().strip()
        lock_file.close()
        msg = (
            f"Another agent-batch is already running (lock: {BATCH_LOCK_PATH}"
            + (f", {holder}" if holder else "")
            + "). Do not start a second batch; remove the lock after confirming no process is running, or use --force"
        )
        raise SystemExit(msg)

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()} cwd={REPO_ROOT}\n")
    lock_file.flush()
    try:
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def list_running_cursor_agents(workspace: Path | None = None) -> list[tuple[int, str]]:
    """Return (pid, cmdline) for cursor agent CLI processes."""
    ws = str(workspace or REPO_ROOT)
    found: list[tuple[int, str]] = []
    try:
        out = subprocess.run(
            ["pgrep", "-af", "cursor agent"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return found
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, _, cmd = line.partition(" ")
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if ws in cmd or "patch_histories/" in cmd:
            found.append((pid, cmd))
    return found


def warn_if_cursor_agents_running(*, force: bool, context: str) -> None:
    agents = list_running_cursor_agents()
    if not agents:
        return
    print(f"Warning: {len(agents)} cursor agent process(es) already running ({context}):", file=sys.stderr)
    for pid, cmd in agents[:5]:
        snippet = cmd if len(cmd) < 120 else cmd[:117] + "..."
        print(f"   pid {pid}: {snippet}", file=sys.stderr)
    if len(agents) > 5:
        print(f"   ... and {len(agents) - 5} more", file=sys.stderr)
    if not force:
        raise SystemExit(
            "Refusing to start another cursor agent (avoid CPU/memory/token pile-up). "
            "Stop the processes above, or pass --force to continue anyway."
        )
    print("Warning: --force set; continuing anyway (confirm you are not running parallel batches)", file=sys.stderr)


def instance_dir(instance_id: str) -> Path:
    return EDITING_SPLIT_DIR / instance_id


def list_file_dirs(instance_id: str) -> list[Path]:
    root = instance_dir(instance_id)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and list(p.glob("original.*"))
    )


def find_step_diffs(file_dir: Path) -> list[Path]:
    steps: list[tuple[int, Path]] = []
    for f in file_dir.iterdir():
        if not f.is_file():
            continue
        m = DIFF_STEP_RE.match(f.name)
        if m:
            steps.append((int(m.group(1)), f))
    return [p for _, p in sorted(steps)]


def find_original_final(file_dir: Path) -> tuple[Path, Path] | None:
    orig = sorted(file_dir.glob("original.*"))
    final = sorted(file_dir.glob("final.*"))
    if not orig or not final:
        return None
    return orig[0], final[0]


def validate_file_dir_per_file(file_dir: Path) -> tuple[bool, str]:
    """Per-file apply_patch_batch chain (does not test global step ordering)."""
    pair = find_original_final(file_dir)
    if pair is None:
        return False, "missing original/final"
    orig, final = pair
    diffs = find_step_diffs(file_dir)
    if not diffs:
        return False, "no numbered diffs"
    try:
        result = apply_patch_batch(orig, diffs)
    except Exception as exc:
        return False, f"apply error: {exc}"
    if result != final.read_text():
        return False, f"apply({len(diffs)} steps) != final"
    return True, f"{len(diffs)} steps (per-file only)"


def cmd_status(instance_id: str, *, per_file: bool) -> int:
    root = instance_dir(instance_id)
    if not root.is_dir():
        print(f"Directory not found: {root}")
        print("   Run run_split first, or check the instance_id.")
        return 1

    print(f"Instance: {instance_id}")
    print(f"Path:     {root}")
    whole = root / "whole.diff"
    print(f"whole.diff: {'yes' if whole.exists() else 'no'}")

    file_dirs = list_file_dirs(instance_id)
    if not file_dirs:
        print("No file subdirectories (need original.* / whole.diff)")
        return 1

    if not per_file:
        activity = find_activity_instance(instance_id)
        if activity is None:
            print("Instance not found in activity_execution jsonl; skipping strict validation")
            print("   Use status --per-file or pass --dataset")
            per_file = True
        else:
            ok, msg = validate_patch_history_strict(activity, quiet=True)
            mark = "OK" if ok else "FAIL"
            print(f"  strict validate: {mark} {msg}")
            return 0 if ok else 1

    for fd in file_dirs:
        pair = find_original_final(fd)
        steps = find_step_diffs(fd)
        step_str = ", ".join(s.name for s in steps) if steps else "(no 1.diff ...)"
        print(f"  {fd.name}: steps=[{step_str}]")
        if pair and steps:
            ok, msg = validate_file_dir_per_file(fd)
            print(f"    per-file validate: {'OK' if ok else 'FAIL'} {msg}")
    return 0


def cmd_validate(
    instance_id: str,
    *,
    per_file: bool,
    dataset: str | None,
) -> int:
    file_dirs = list_file_dirs(instance_id)
    if not file_dirs:
        print(f"No file directories under {instance_dir(instance_id)}")
        return 1

    if per_file:
        all_ok = True
        for fd in file_dirs:
            ok, msg = validate_file_dir_per_file(fd)
            mark = "OK" if ok else "FAIL"
            print(f"{mark} {fd.name}: {msg}")
            all_ok &= ok
        if all_ok:
            print(f"\n{instance_id}: all files pass per-file apply chain")
            print("   Warning: global git apply was not checked; rerun without --per-file")
            return 0
        print(f"\n{instance_id}: validation failed")
        return 1

    dataset_paths = [dataset] if dataset else None
    activity = find_activity_instance(instance_id, dataset_paths=dataset_paths)
    if activity is None:
        print(f"Instance not found in jsonl: {instance_id}")
        print("   Pass --dataset crawled_data/activity_execution/{repo}-task-instances.jsonl")
        print("   Or use --per-file for a coarse per-file check")
        return 1

    ok, msg = validate_patch_history_strict(activity)
    if ok:
        print(f"\n{instance_id}: strict validation passed (global steps + git apply)")
        return 0
    print(f"\n{instance_id}: strict validation failed: {msg}")
    return 1


def build_agent_prompt(instance_id: str) -> str:
    return f"""Split the PR into an ordered sub-edit sequence:

@patch_histories/{instance_id}

Requirements:
1. Read patch_histories/{instance_id}/whole.diff first to understand PR intent and narrative lines.
2. Follow skills/pr-subedit-principles/SKILL.md (hard preferences in sections 3b/3c; section 3d only when applicable).
3. Follow the workflow in skills/pr-subedit-workflow/SKILL.md.
4. Produce 1.diff, 2.diff, ... per modified file. Step count is descriptive, not a fixed target:
   - PR-level whole.diff is only a few lines: 2-3 steps (use 4 only if there are distinct semantic phases).
   - Very small symmetric change: 2-3 steps; do not pad mechanically.
   - Normal single- or two-file PR: commonly 3-5 steps; 4 is a normal center point.
   - Moderate multi-file change: commonly 4-6 steps, depending on dependencies and repeated patterns.
   - Large refactor or long verified commit chain: 6+ steps only when edit phases truly warrant it.
5. Prefer finer over coarser: one predictable editing phase per step. If a single hunk is large (>~15 net lines) or mixes multiple phases (new structure + implementation + registration/call-site wiring), split further. No doc-only steps; no pass/... placeholders (intermediate syntax may be incomplete).
6. Structure before body: for new class/function/method/solver/handler/adapter, add the minimal shell or signature first, then fill validation, body logic, returns, and call-site wiring in later steps.
7. The final step must be testable: do not end with docs/comments/formatting-only changes. Merge those into earlier code/import steps; keep substantive behavior, API, control flow, or type constraints for the last step.
8. Section 3d / 3b (when used): shell steps contain only the minimal syntax starter; the next step fills the body. Do not add pass just to make intermediate states valid Python.
9. Every N.diff with N>=2 must be generated relative to the repository state after global steps 1..N-1 (use quick_diff or apply 1..N-1 first).
10. After editing, run:
    python scripts/pr_subedit_instance.py validate {instance_id}
11. Remove temporary files created while writing diffs (_old.py / _new.py / *.t.diff).
12. Do not update skill documents unless explicitly asked.
"""


def cmd_prompt(instance_id: str) -> int:
    if not instance_dir(instance_id).is_dir():
        print(f"Directory not found: {instance_dir(instance_id)}", file=sys.stderr)
        return 1
    print(build_agent_prompt(instance_id))
    return 0


@dataclass
class AgentRunResult:
    instance_id: str
    agent_rc: int = 0
    validate_rc: int | None = None
    skipped: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        if self.skipped:
            return False
        if self.agent_rc != 0:
            return False
        return self.validate_rc in (None, 0)


def run_agent(
    instance_id: str,
    *,
    dry_run: bool = False,
    dataset: str | None = None,
    per_file: bool = False,
    skip_validate: bool = False,
    force: bool = False,
    model: str | None = None,
) -> AgentRunResult:
    """Run cursor agent for one instance, then strict validate unless skipped."""
    result = AgentRunResult(instance_id=instance_id)
    if not instance_dir(instance_id).is_dir():
        result.skipped = True
        result.error = f"directory not found: {instance_dir(instance_id)}"
        return result

    prompt = build_agent_prompt(instance_id)
    cmd = [
        "cursor", "agent", "-p", "--force",
        "--workspace", str(REPO_ROOT),
    ]
    if model is not None:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    if dry_run:
        print(f"[dry-run] {instance_id}")
        print("Would run:", " ".join(cmd[:6]), "...")
        return result

    warn_if_cursor_agents_running(force=force, context=f"starting {instance_id}")

    print(f"\n{'=' * 60}")
    print(f"Starting cursor agent: {instance_id}")
    print(f"{'=' * 60}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    result.agent_rc = proc.returncode
    if proc.returncode != 0:
        result.error = f"agent exit {proc.returncode}"
        return result

    if skip_validate:
        return result

    print(f"\n--- post-validate: {instance_id} ---")
    result.validate_rc = cmd_validate(instance_id, per_file=per_file, dataset=dataset)
    if result.validate_rc != 0:
        result.error = "validate failed"
    return result


def load_instance_ids_from_file(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line.split()[0])
    return ids


def run_agent_batch(
    instance_ids: list[str],
    *,
    dry_run: bool = False,
    dataset: str | None = None,
    per_file: bool = False,
    skip_validate: bool = False,
    continue_on_error: bool = False,
    force: bool = False,
    model: str | None = None,
) -> list[AgentRunResult]:
    """Sequentially run cursor agent (+ validate) for each instance_id."""
    seen: set[str] = set()
    ordered: list[str] = []
    for iid in instance_ids:
        if iid not in seen:
            seen.add(iid)
            ordered.append(iid)

    if not ordered:
        return []

    def _run() -> list[AgentRunResult]:
        if not dry_run:
            warn_if_cursor_agents_running(force=force, context="agent-batch startup")

        results: list[AgentRunResult] = []
        total = len(ordered)
        for idx, instance_id in enumerate(ordered, start=1):
            print(f"\n>>> [{idx}/{total}] {instance_id}")
            result = run_agent(
                instance_id,
                dry_run=dry_run,
                dataset=dataset,
                per_file=per_file,
                skip_validate=skip_validate,
                force=force,
                model=model,
            )
            results.append(result)
            if not result.ok and not dry_run and not continue_on_error:
                print(f"\nStopped: {instance_id} failed (use --continue-on-error to keep going)")
                break
        return results

    if dry_run:
        results = _run()
    else:
        with batch_run_lock(force=force):
            results = _run()

    ok_count = sum(1 for r in results if r.ok)
    fail_count = sum(1 for r in results if not r.ok and not r.skipped)
    skip_count = sum(1 for r in results if r.skipped)
    total = len(ordered)
    pending = total - len(results)

    print(f"\n{'=' * 60}")
    print(f"Batch summary: {ok_count} ok, {fail_count} failed, {skip_count} skipped", end="")
    if pending:
        print(f", {pending} not run", end="")
    print()
    for r in results:
        if r.ok:
            mark = "OK"
        elif r.skipped:
            mark = "SKIP"
        else:
            mark = "FAIL"
        detail = f" — {r.error}" if r.error else ""
        print(f"  {mark} {r.instance_id}{detail}")
    print(f"{'=' * 60}")

    return results


def cmd_agent(
    instance_id: str,
    dry_run: bool,
    dataset: str | None,
    per_file: bool,
    force: bool,
    model: str | None = None,
) -> int:
    result = run_agent(
        instance_id,
        dry_run=dry_run,
        dataset=dataset,
        per_file=per_file,
        force=force,
        model=model,
    )
    if dry_run:
        print("\n--- prompt ---\n")
        print(build_agent_prompt(instance_id))
        return 0
    if result.skipped:
        print(result.error, file=sys.stderr)
        return 1
    if result.agent_rc != 0:
        return result.agent_rc
    return result.validate_rc or 0


def cmd_agent_batch(
    instance_ids: list[str],
    *,
    from_file: Path | None,
    dry_run: bool,
    dataset: str | None,
    per_file: bool,
    skip_validate: bool,
    continue_on_error: bool,
    force: bool,
    model: str | None = None,
) -> int:
    ids = list(instance_ids)
    if from_file is not None:
        if not from_file.is_file():
            print(f"File not found: {from_file}", file=sys.stderr)
            return 1
        ids.extend(load_instance_ids_from_file(from_file))

    if not ids:
        print("No instance_id provided (--instance-ids or --from-file)", file=sys.stderr)
        return 1

    results = run_agent_batch(
        ids,
        dry_run=dry_run,
        dataset=dataset,
        per_file=per_file,
        skip_validate=skip_validate,
        continue_on_error=continue_on_error,
        force=force,
        model=model,
    )
    if all(r.ok for r in results):
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="PR sub-edit split helper")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--per-file",
        action="store_true",
        help="validate: per-file apply_patch_batch only (not global git apply)",
    )
    common.add_argument(
        "--dataset",
        metavar="JSONL",
        help="validate: jsonl path containing instance (default: scan activity_execution)",
    )

    agent_common = argparse.ArgumentParser(add_help=False)
    agent_common.add_argument("--dry-run", action="store_true", help="print command only, do not run agent")
    agent_common.add_argument(
        "--skip-validate",
        action="store_true",
        help="skip validate after agent completes",
    )
    agent_common.add_argument(
        "--force",
        action="store_true",
        help="ignore batch lock / existing cursor agent processes (use with care)",
    )
    agent_common.add_argument(
        "--model",
        default="composer-2.5",
        help="cursor agent model (default: composer-2.5)",
    )

    p_status = sub.add_parser("status", parents=[common], help="show instance status")
    p_status.add_argument("instance_id")

    p_validate = sub.add_parser("validate", parents=[common], help="validate apply chain")
    p_validate.add_argument("instance_id")

    p_prompt = sub.add_parser("prompt", help="print agent prompt")
    p_prompt.add_argument("instance_id")

    p_agent = sub.add_parser("agent", parents=[common, agent_common], help="run agent for one instance")
    p_agent.add_argument("instance_id")

    p_batch = sub.add_parser(
        "agent-batch",
        parents=[common, agent_common],
        help="run agent sequentially (--instance-ids or --from-file)",
    )
    p_batch.add_argument(
        "--instance-ids",
        nargs="+",
        default=[],
        metavar="ID",
        help="one or more instance_id values",
    )
    p_batch.add_argument(
        "--from-file",
        type=Path,
        metavar="FILE",
        help="one instance_id per line (# comments ignored)",
    )
    p_batch.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue with remaining instances after a failure",
    )

    args = parser.parse_args()

    if args.command == "status":
        return cmd_status(args.instance_id, per_file=args.per_file)
    if args.command == "validate":
        return cmd_validate(args.instance_id, per_file=args.per_file, dataset=args.dataset)
    if args.command == "prompt":
        return cmd_prompt(args.instance_id)
    if args.command == "agent":
        return cmd_agent(
            args.instance_id,
            args.dry_run,
            args.dataset,
            args.per_file,
            args.force,
            args.model,
        )
    return cmd_agent_batch(
        args.instance_ids,
        from_file=args.from_file,
        dry_run=args.dry_run,
        dataset=args.dataset,
        per_file=args.per_file,
        skip_validate=args.skip_validate,
        continue_on_error=args.continue_on_error,
        force=args.force,
        model=args.model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
