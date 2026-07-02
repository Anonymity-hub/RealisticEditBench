"""Detect Git LFS pointer files and optionally materialize real dataset content."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def is_git_lfs_pointer(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.readline().strip() == LFS_POINTER_PREFIX
    except OSError:
        return False


def parse_git_lfs_pointer(path: str | Path) -> Optional[dict[str, str]]:
    p = Path(path)
    if not is_git_lfs_pointer(p):
        return None
    meta: dict[str, str] = {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line == LFS_POINTER_PREFIX:
                    continue
                key, _, value = line.partition(" ")
                if key and value:
                    meta[key] = value
    except OSError:
        return None
    return meta or None


def find_git_repo_root(path: str | Path) -> Optional[Path]:
    p = Path(path).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def git_lfs_pull(path: str | Path, repo_root: Optional[Path] = None) -> bool:
    """Pull one LFS-tracked file. Returns True if the file is no longer a pointer."""
    p = Path(path).resolve()
    root = repo_root or find_git_repo_root(p)
    if root is None:
        print(f"Warning: cannot locate git repo root for {p}")
        return False

    try:
        rel = p.relative_to(root)
    except ValueError:
        rel = p

    try:
        result = subprocess.run(
            ["git", "lfs", "pull", "--include", str(rel)],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
        )
    except FileNotFoundError:
        print("Warning: `git` not found; cannot run git lfs pull.")
        return False
    except subprocess.TimeoutExpired:
        print(f"Warning: git lfs pull timed out for {p}")
        return False

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        if err:
            print(f"git lfs pull failed ({result.returncode}): {err}")
        return False

    if is_git_lfs_pointer(p):
        print(f"Warning: git lfs pull finished but {p} is still an LFS pointer.")
        return False
    return True


def lfs_pull_help(path: str | Path) -> str:
    p = Path(path).resolve()
    root = find_git_repo_root(p)
    try:
        rel = p.relative_to(root) if root else p
    except ValueError:
        rel = p
    parent_glob = f"{rel.parent}/**/*.jsonl" if rel.parent != Path(".") else "**/*.jsonl"
    return (
        f"Dataset not available: {p} is a Git LFS pointer (real data not downloaded).\n"
        f"Run:\n"
        f"  git lfs install\n"
        f"  git lfs pull --include=\"{rel}\"\n"
        f"Or pull all dataset JSONL files:\n"
        f"  git lfs pull --include=\"{parent_glob}\""
    )


def ensure_jsonl_materialized(path: str | Path, *, auto_pull: bool = True) -> Path:
    """
    If `path` is a Git LFS pointer, try `git lfs pull` and verify the real file exists.
    Raises FileNotFoundError with actionable instructions when materialization fails.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    if not is_git_lfs_pointer(p):
        return p

    meta = parse_git_lfs_pointer(p)
    expected_size = meta.get("size", "unknown") if meta else "unknown"
    print(
        f"Dataset file is a Git LFS pointer (not the real JSONL): {p}\n"
        f"  expected size: {expected_size} bytes"
    )

    if auto_pull:
        print("Attempting git lfs pull for this file...")
        if git_lfs_pull(p):
            print(f"Git LFS pull succeeded: {p}")
            return p

    raise FileNotFoundError(lfs_pull_help(p))
