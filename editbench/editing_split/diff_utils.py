import difflib
import re
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Union


from editbench.editing_split.constants import EDITING_SPLIT_DIR


import tempfile
import os
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class PatchApplyError(Exception):
    pass


def _is_unified_diff_hunk_line(line: str) -> bool:
    """True if the line is an addition/deletion row inside a unified diff hunk."""
    return (
        (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )


_GIT_DIFF_FILE_HEADER = re.compile(
    r"^diff --git a/(?P<old>.+?) b/(?P<new>.+?)\s*$",
    re.MULTILINE,
)


def split_patch_by_file(whole_patch: str) -> dict[str, str]:
    """
    Split a multi-file git diff into per-file diff strings keyed by path.

    Keys use the ``b/`` path from each ``diff --git`` header. When a patch
    renames a file, the ``a/`` path is also keyed to the same diff block.
    """
    if not whole_patch or not whole_patch.strip():
        return {}

    parts = re.split(r"(?=^diff --git )", whole_patch, flags=re.MULTILINE)
    by_file: dict[str, str] = {}
    for part in parts:
        part = part.strip("\n")
        if not part.startswith("diff --git"):
            continue
        match = _GIT_DIFF_FILE_HEADER.match(part)
        if not match:
            continue
        if not part.endswith("\n"):
            part += "\n"
        new_path = match.group("new")
        old_path = match.group("old")
        by_file[new_path] = part
        if old_path != new_path:
            by_file[old_path] = part
    return by_file


def extract_file_diff_from_patch(whole_patch: str, file_path: str) -> str:
    """Return the single-file diff for ``file_path`` from a multi-file patch."""
    by_file = split_patch_by_file(whole_patch)
    if file_path in by_file:
        return by_file[file_path]
    for path, diff in by_file.items():
        if file_path in diff:
            return diff
    return ""


def ensure_diff_trailing_padding(diff: str, trailing_lines: int = 3) -> str:
    """
    Normalize diff text for downstream parsers.

    If the last ``trailing_lines`` lines contain hunk +/- rows, append extra
    newlines so the patch ends with redundant blank lines (some tools require
    this to detect patch boundaries).
    """
    if not diff:
        return diff

    if not diff.endswith("\n"):
        diff += "\n"

    lines = diff.splitlines()
    tail = lines[-trailing_lines:] if len(lines) >= trailing_lines else lines
    if any(_is_unified_diff_hunk_line(line) for line in tail):
        diff += "\n" * trailing_lines

    return diff


def trim_diff_hunks(diff_content: str, context_lines: int = 3) -> str:
    """
    Trims each hunk in a diff to keep only 3 lines of irrelevant context
    before and after the modified content.
    :param diff_content: Original diff content as a string.
    :param context_lines: Number of context lines to keep (default: 3).
    :return: Trimmed diff content.
    """
    # Regex to match diff hunks (starts with @@, ends at next diff --git or EOF)
    hunk_pattern = re.compile(
        r'(diff --git .*?\n.*?\n)(@@ -(\d+),(\d+) \+(\d+),(\d+) @@)(.*?)(?=diff --git|$)',
        re.DOTALL | re.MULTILINE
    )

    trimmed_diff = []
    for match in hunk_pattern.finditer(diff_content):
        file_header = match.group(1)  # diff --git lines and --- +++ lines
        hunk_header = match.group(2)  # @@ -x,y +x,y @@
        hunk_body = match.group(6).strip()  # Content inside the hunk

        # Split hunk into lines (remove empty lines)
        lines = [line for line in hunk_body.split('\n') if line.strip() != '']

        # Locate modified lines (starting with - or +)
        modified_indices = [i for i, line in enumerate(lines) if line.startswith(('-', '+'))]
        if not modified_indices:
            trimmed_diff.append(file_header + hunk_header + '\n' + hunk_body + '\n')
            continue

        # Determine context range to keep (context_lines before/after modifications)
        start = max(0, modified_indices[0] - context_lines)
        end = min(len(lines), modified_indices[-1] + context_lines + 1)  # +1 to include end index

        # Extract trimmed lines and update hunk line counts
        trimmed_lines = lines[start:end]
        original_lines = len([line for line in trimmed_lines if not line.startswith('+')])
        modified_lines = len([line for line in trimmed_lines if not line.startswith('-')])

        # Regenerate hunk header with updated line numbers
        new_hunk_header = re.sub(
            r'@@ -\d+,\d+ \+\d+,\d+ @@',
            f'@@ -{match.group(3)},{original_lines} +{match.group(4)},{modified_lines} @@',
            hunk_header
        )

        # Assemble the trimmed hunk
        trimmed_hunk = (
                file_header
                + new_hunk_header + '\n'
                + '\n'.join(trimmed_lines) + '\n'
        )
        trimmed_diff.append(trimmed_hunk)

    return ''.join(trimmed_diff)


def generate_diff_with_file(old_file: Union[Path, str], new_file: Union[Path, str], filename: str, save_path=None) -> str:
    """
    Generate a git-style diff string between two file versions

    :param old_file: Path to the old version of the file
    :param new_file: Path to the new version of the file
    :param filename: Target filename to display in diff headers (with a/ and b/ prefixes)
    :param save_path: Location to save
    :return: Git-style formatted diff string
    """
    # Read file contents (split by lines while preserving line endings)
    old_content = Path(old_file).read_text()
    new_content = Path(new_file).read_text()
    diff_content = generate_diff(old_content, new_content, filename)
    if save_path:
        save_path = Path(save_path)
        save_path.write_text(diff_content)
    return diff_content


def generate_diff(old_content: str, new_content: str, filename: str) -> str:
    # Read file contents (split by lines while preserving line endings)
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    # Generate base unified diff (difflib uses -/+ for old/new by default)
    raw_diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}",  # Corresponds to git's a/ prefix
        tofile=f"b/{filename}",  # Corresponds to git's b/ prefix
        lineterm="\n"  # Prevent difflib from adding extra newlines
    ))

    # Add git-specific diff header (not generated by difflib)
    git_header = f"diff --git a/{filename} b/{filename}\n"
    return git_header + "".join(raw_diff)


def apply_diff(original_content: str, diff_content: str, fuzz: int = 1, strip: int = 1) -> str:
    """
    Apply a diff patch to original content using the system 'patch' command, supporting fuzzy matching.

    Args:
        original_content: Text of the original file
        diff_content: Patch content in unified diff format
        fuzz: Maximum allowed line offset for fuzzy matching (default: 5)
        strip: Number of path prefixes to remove (default: 1, like -p1)

    Returns:
        (New content, Success status, Logs)
    """
    logs = []
    # try:
    # Create temporary directory to avoid modifying real files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write original content to a temporary file
        original_path = os.path.join(tmpdir, "original.txt")
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(original_content)

        # Write diff content to a temporary patch file
        patch_path = os.path.join(tmpdir, "patch.diff")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(diff_content)

        # Run 'patch' command with parameters: --batch (non-interactive), --fuzz, -p
        cmd = [
            "patch",
            # "--forward",
            "--batch",  # Non-interactive mode (auto-accept defaults)
            f"--fuzz={fuzz}",  # Allow up to 'fuzz' line offset
            f"-p{strip}",  # Remove 'strip' path prefixes
            "-i", patch_path,  # Input patch file
            original_path  # Target file to patch
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception; handle errors manually
        )

        # Collect command output for logs
        logs = [
            "Command executed: " + " ".join(cmd),
            "Stdout: " + result.stdout,
            "Stderr: " + result.stderr
        ]

        # Check if patch succeeded (exit code 0 or 1 for warnings)
        success = result.returncode == 0

        if not success or 'Reversed (or previously applied) patch detected!' in result.stdout:
            error_msg = (
                "Apply failed！\n"
                f"return code: {result.returncode}\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout: {result.stdout.strip()}\n"
                f"stderror: {result.stderr.strip()}"
            )
            raise PatchApplyError(error_msg)

        # Read the patched content
        with open(original_path, "r", encoding="utf-8") as f:
            new_content = f.read()

        # return new_content, success, logs
        return new_content

    # except Exception as e:
    #     return "", False, [f"Error during patching: {str(e)}"]


def apply_diff_with_file(original_file: Union[Path, str], diff_path: Union[Path, str], fuzz: int = 1,
                         strip: int = 1, save_path: str = None) -> str:
    # Read file contents (split by lines while preserving line endings)
    original_content = Path(original_file).read_text()
    diff_content = Path(diff_path).read_text()

    result = apply_diff(original_content, diff_content, fuzz, strip)

    if save_path:
        save_path = Path(save_path)
        save_path.write_text(result)

    return result


def diff_minus(filename: str, original_file: Union[Path, str], sub_diff: Union[Path, str],
               original_diff: Union[Path, str],
               save_path: Union[Path, str] = None):
    mid_file_content = apply_diff_with_file(original_file, sub_diff)

    final_file_content = apply_diff_with_file(original_file, original_diff)

    diff_content = generate_diff(mid_file_content, final_file_content, filename)

    if save_path:
        save_path = Path(save_path)
        save_path.write_text(diff_content)

    return diff_content


def apply_patch_batch(original_file: Union[Path, str], diff_path_list: Union[list[Path], list[str]], fuzz: int = 1,
                      strip: int = 0, save_path: str = None):
    origin_content = Path(original_file).read_text()

    for diff_path in diff_path_list:
        diff_content = Path(diff_path).read_text()
        origin_content = apply_diff(origin_content, diff_content, fuzz, strip)

    if save_path:
        Path(save_path).write_text(origin_content)

    return origin_content


def _filename_from_diff(diff_path: Union[Path, str]) -> str:
    text = Path(diff_path).read_text()
    match = re.search(r"diff --git a/(\S+) b/\S+", text)
    return match.group(1) if match else ""


def find_original_file(sub_dir: Union[Path, str]) -> Path:
    """Return ``original.{ext}`` under a per-file patch history directory."""
    sub_dir = Path(sub_dir)
    matches = sorted(sub_dir.glob("original.*"))
    if not matches:
        raise FileNotFoundError(f"No original.* snapshot found under: {sub_dir}")
    if len(matches) > 1:
        logger.warning(
            "Multiple original.* files found under %s, using %s",
            sub_dir,
            matches[0],
        )
    return matches[0]


def resolve_base_file(sub_dir: Union[Path, str], step: int) -> Path:
    """
    Return the file content to patch when generating ``step.diff``.

    For step 2 this is ``original.*``. For step N>2 this is the file after
    applying ``1.diff`` through ``(N-2).diff``, cached as ``(N-2).py`` when needed.
    """
    if step < 2:
        raise ValueError(f"step must be >= 2, got {step}")

    sub_dir = Path(sub_dir)
    original = find_original_file(sub_dir)
    if step == 2:
        return original

    base_path = sub_dir / f"{step - 2}.py"
    if base_path.exists():
        return base_path

    diff_paths = [sub_dir / f"{i}.diff" for i in range(1, step - 1)]
    missing = [path for path in diff_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot build base file for step {step} under {sub_dir}: "
            f"missing {missing[0].name}"
        )

    apply_patch_batch(original, diff_paths, save_path=base_path)
    return base_path


def _resolve_full_diff_path(sub_dir: Path, prev: int, step: int) -> tuple[Path, bool]:
    """
    Locate the full (pre-trim) diff for step ``prev``.

    Returns ``(path, is_anchor)`` where ``is_anchor`` is True when the path is a
    read-only fallback such as ``whole.diff`` (must not be deleted).
    """
    full_path = sub_dir / f"{prev}.diff"
    if full_path.exists():
        return full_path, False

    if step == 2:
        candidate = sub_dir / "whole.diff"
        if candidate.exists():
            return candidate, True

    return full_path, False


def _promote_trimmed_diff(sub_dir: Path, prev: int, trimmed_path: Path, full_path: Path, is_anchor: bool) -> None:
    """Rename ``{prev}.t.diff`` to ``{prev}.diff`` without removing anchor fallbacks."""
    target = sub_dir / f"{prev}.diff"
    if is_anchor:
        if target.exists():
            target.unlink()
        trimmed_path.rename(target)
        return
    replace_file(full_path, trimmed_path)


def quick_generate_sub_diff(instance_id: str, step_index: Union[str, int]):
    """
    Split step ``(N-1)`` into a trimmed first part and a new ``N.diff``.

    Workflow (same for every ``step_index`` >= 2):

    1. Keep the full previous step at ``{N-1}.diff`` (or, for step 2 only,
       fall back to ``whole.diff`` when ``1.diff`` is missing).
    2. Manually trim hunks and save as ``{N-1}.t.diff``.
    3. Run this command with ``step_index=N``.
    4. It writes ``N.diff = diff_minus(base, trimmed, full)`` and renames
       ``{N-1}.t.diff`` to ``{N-1}.diff``.

    ``base`` is ``original.*`` for N=2, otherwise the file after applying
    ``1.diff`` .. ``(N-2).diff``.
    """
    src_path = EDITING_SPLIT_DIR / instance_id

    if not src_path.exists():
        raise FileNotFoundError(f"Instance root directory does not exist: {src_path}")

    try:
        step = int(step_index)
    except (TypeError, ValueError) as e:
        raise ValueError(f"step_index must be an integer, current value: {step_index}") from e
    if step < 2:
        raise ValueError(f"step_index must be >= 2, current value: {step}")

    prev = step - 1
    for sub_dir in sorted(src_path.iterdir()):
        if not sub_dir.is_dir():
            continue

        trimmed_path = sub_dir / f"{prev}.t.diff"
        result_path = sub_dir / f"{step}.diff"

        if not trimmed_path.exists():
            logger.warning("Trimmed diff does not exist, skipped: %s", trimmed_path)
            continue

        full_path, is_anchor = _resolve_full_diff_path(sub_dir, prev, step)
        if not full_path.exists():
            logger.warning("Full previous diff does not exist, skipped: %s", full_path)
            continue
        if is_anchor:
            logger.info("Using %s as full diff for step %s under %s", full_path.name, step, sub_dir.name)

        filename = _filename_from_diff(full_path) or _filename_from_diff(trimmed_path)
        if not filename:
            logger.warning("Could not parse filename from diff headers under %s", sub_dir)
            continue

        try:
            base_file = resolve_base_file(sub_dir, step)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping %s: %s", sub_dir.name, exc)
            continue

        diff_minus(filename, base_file, trimmed_path, full_path, result_path)
        _promote_trimmed_diff(sub_dir, prev, trimmed_path, full_path, is_anchor)
        logger.info("Sub directory %s processed, wrote %s", sub_dir.name, result_path.name)


def quick_split_diff(instance_id: str, start_index: int):
    src_path = EDITING_SPLIT_DIR / instance_id
    if not src_path.exists():
        raise FileNotFoundError(f"Instance root directory does not exist: {src_path}")
    ground_truth = []
    file_context = {}
    for sub_dir in src_path.iterdir():
        if not sub_dir.is_dir():
            continue
        batch_list = []
        for i in range(1, start_index + 1):
            path_ = sub_dir / f"{str(i)}.diff"
            if path_.exists():
                batch_list.append(path_)
        match = re.search(r'diff --git a/(\S+) b/\S+', (sub_dir/"whole-1.diff").read_text())
        filename = match.group(1) if match else ""
        before_content = apply_patch_batch(sub_dir/"original.py", batch_list)
        file_context[filename] = before_content
        after_content = (sub_dir/"final.py").read_text()

        diff_content = generate_diff(before_content, after_content, filename)
        if len(diff_content.strip().split("\n")) > 2:
            ground_truth.append(diff_content)
    return "\n".join(ground_truth), file_context


def replace_file(file_to_delete: str, source_file: str) -> None:
    """
    Delete file_to_delete, then rename source_file to the name of file_to_delete
    :param file_to_delete: The file to be deleted (target file name)
    :param source_file: The file to be renamed (source file name)
    """

    target = Path(file_to_delete)
    source = Path(source_file)

    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"Source path is not a file: {source}")

    try:
        target.unlink(missing_ok=True)

        source.rename(target)
        print(f"{source} renamed to {target}")

    except PermissionError:
        raise PermissionError(f"No permission to operate file: {target} or {source} is occupied")
    except Exception as e:
        raise RuntimeError(f"Operation failed: {str(e)}")





if __name__ == "__main__":
    # Example:
    #   python -m editbench.editing_split.diff_utils gene --filename foo.py --file1 old.py --file2 new.py [--res out.diff]
    #   python -m editbench.editing_split.diff_utils apply --file base.txt --diff patch.diff [--res patched.txt] [--fuzz 1 --strip 1]
    #   python -m editbench.editing_split.diff_utils batch_apply --file base.txt --diffs 1.diff 2.diff [--res out.txt]
    #   python -m editbench.editing_split.diff_utils diff_minus --filename x.py --file orig.py --diff1 sub.diff --diff2 full.diff [--res result.diff]
    #   python -m editbench.editing_split.diff_utils quick_diff --instance_id astropy__astropy-pull-123 --step_index 2
    #   python -m editbench.editing_split.diff_utils trim --input in.diff [--output out.diff] [--context 3]
    parser = ArgumentParser(
        description="Diff utilities: generate git-style diff, apply patch, batch apply, diff_minus, quick_split, trim hunks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    gene_parser = subparsers.add_parser("gene", help="Generate git-style diff between two files")
    gene_parser.add_argument("--filename", required=True, type=str, help="Filename for diff header (e.g. path/to/file.py)")
    gene_parser.add_argument("--file1", required=True, type=str, help="Path to old file")
    gene_parser.add_argument("--file2", required=True, type=str, help="Path to new file")
    gene_parser.add_argument("--res", type=str, default=None, help="Optional: save diff to this path")

    apply_parser = subparsers.add_parser("apply", help="Apply a diff/patch to a file")
    apply_parser.add_argument("--file", required=True, type=str, help="Path to file to patch")
    apply_parser.add_argument("--diff", required=True, type=str, help="Path to patch/diff file")
    apply_parser.add_argument("--fuzz", type=int, default=1, help="Patch fuzz (default: 1)")
    apply_parser.add_argument("--strip", type=int, default=1, help="Strip path components -p (default: 1)")
    apply_parser.add_argument("--res", type=str, default=None, help="Optional: save patched content to this path")

    apply_batch_parser = subparsers.add_parser("batch_apply", help="Apply multiple diffs in order to a file")
    apply_batch_parser.add_argument("--file", required=True, type=str, help="Path to base file")
    apply_batch_parser.add_argument("--diffs", nargs="+", required=True, type=str, help="Paths to diff files in order")
    apply_batch_parser.add_argument("--fuzz", type=int, default=1, help="Patch fuzz (default: 1)")
    apply_batch_parser.add_argument("--strip", type=int, default=0, help="Strip path components -p (default: 0)")
    apply_batch_parser.add_argument("--res", type=str, default=None, help="Optional: save result to this path")

    diff_minus_parser = subparsers.add_parser("diff_minus", help="Compute diff between two patch results (mid vs final)")
    diff_minus_parser.add_argument("--filename", required=True, type=str, help="Filename for output diff header")
    diff_minus_parser.add_argument("--file", required=True, type=str, help="Path to original file")
    diff_minus_parser.add_argument("--diff1", required=True, type=str, help="Path to first (sub) diff")
    diff_minus_parser.add_argument("--diff2", required=True, type=str, help="Path to second (full) diff")
    diff_minus_parser.add_argument("--res", type=str, default=None, help="Optional: save result diff to this path")

    quick_diff_parser = subparsers.add_parser(
        "quick_diff",
        help=(
            "Split step (N-1) into trimmed (N-1).diff + N.diff for an instance "
            "(requires {N-1}.t.diff to exist)"
        ),
    )
    quick_diff_parser.add_argument("--instance_id", required=True, type=str, help="Instance id (e.g. repo__repo-pull-123)")
    quick_diff_parser.add_argument("--step_index", required=True, type=str, help="Step index to generate (>= 2)")

    trim_parser = subparsers.add_parser("trim", help="Trim diff hunks to limited context lines")
    trim_parser.add_argument("--input", required=True, type=str, help="Path to input diff file")
    trim_parser.add_argument("--output", type=str, default=None, help="Optional: output path; if omitted, print to stdout")
    trim_parser.add_argument("--context", type=int, default=3, help="Context lines to keep (default: 3)")

    args = parser.parse_args()

    if args.command == "gene":
        res = generate_diff_with_file(args.file1, args.file2, args.filename, args.res)
        if args.res:
            print(f"Saved diff to {args.res}")
        else:
            print(res[:2000] + ("..." if len(res) > 2000 else ""))
    elif args.command == "apply":
        res = apply_diff_with_file(args.file, args.diff, args.fuzz, args.strip, args.res)
        if args.res:
            print(f"Saved patched content to {args.res}")
        else:
            print(res[:2000] + ("..." if len(res) > 2000 else ""))
    elif args.command == "batch_apply":
        res = apply_patch_batch(args.file, args.diffs, args.fuzz, args.strip, args.res)
        if args.res:
            print(f"Saved result to {args.res}")
        else:
            print(res[:2000] + ("..." if len(res) > 2000 else ""))
    elif args.command == "diff_minus":
        res = diff_minus(args.filename, args.file, args.diff1, args.diff2, args.res)
        if args.res:
            print(f"Saved diff to {args.res}")
        else:
            print(res[:2000] + ("..." if len(res) > 2000 else ""))
    elif args.command == "quick_diff":
        quick_generate_sub_diff(args.instance_id, args.step_index)
        print(f"Done. Instance {args.instance_id} step_index={args.step_index}")
    elif args.command == "trim":
        content = Path(args.input).read_text()
        out = trim_diff_hunks(content, context_lines=args.context)
        if args.output:
            Path(args.output).write_text(out)
            print(f"Saved trimmed diff to {args.output}")
        else:
            print(out)
