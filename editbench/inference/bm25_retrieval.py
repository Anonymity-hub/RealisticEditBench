import re
import subprocess
from pathlib import Path
from typing import List, Dict, Set
import logging

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank_bm25: pip install rank-bm25")

from editbench.evaluation.constants import MAP_INSTALLED_REPO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_files_from_diff(diff_content: str) -> Set[str]:
    """
    Extract the files involved from the diff content
    
    :param diff_content: diff content string
    :return: file path set
    """
    files = set()
    # Match diff --git a/path b/path format
    pattern = r'diff --git a/(\S+) b/\S+'
    matches = re.findall(pattern, diff_content)
    files.update(matches)
    return files


def extract_query_text_from_pre_edits(pre_edits: List[str]) -> str:
    """
    Extract the query text from pre_edits
    
    :param pre_edits: history edit steps list
    :return: query text
    """
    # Merge all pre_edits as query text
    query_text = "\n".join(pre_edits)
    return query_text


def get_repo_path(repo: str) -> Path:
    """
    Get the repository path from MAP_INSTALLED_REPO
    
    :param repo: repository name, format as "owner/repo"
    :return: repository path
    """
    repo_path = MAP_INSTALLED_REPO.get(repo)
    if not repo_path:
        raise ValueError(f"Repository {repo} is not in MAP_INSTALLED_REPO")
    return Path(repo_path)


def reset_repo_to_commit(repo_path: Path, commit_sha: str, pre_edits: List[str] = None) -> None:
    """
    Reset the repository to the specified commit, and optionally apply pre_edits
    
    :param repo_path: repository path
    :param commit_sha: commit SHA
    :param pre_edits: optional pre_edits list, if provided, these patches will be applied after reset and added to the staging area
    """
    try:
        # Switch to the repository directory and reset
        subprocess.run(
            ["git", "reset", "--hard", commit_sha],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Repository {repo_path} reset to commit {commit_sha}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Reset repository failed: {e.stderr}")
        raise
    
    # Apply pre_edits (if provided)
    if pre_edits:
        logger.info(f"Starting to apply {len(pre_edits)} pre_edits...")
        
        # Apply each patch directly
        for idx, patch in enumerate(pre_edits):
            try:
                patch = patch + "\n"
                # Extract file name for logging
                file_match = re.findall(r"--- a/(.*)", patch)
                file_info = file_match[0] if file_match else "unknown"
                logger.info(f"  Applying patch-{idx + 1} (file: {file_info})...")
                
                # Use subprocess to execute git apply, passing patch through stdin
                result = subprocess.run(
                    ["git", "apply", "-v"],
                    cwd=repo_path,
                    input=patch,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"  ✓ Applied patch-{idx + 1}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to apply pre_edit patch-{idx + 1}: {e.stderr}")
                raise RuntimeError(f"Failed to apply pre_edit patch-{idx + 1}: {e.stderr}")
        
        logger.info(f"Applied all {len(pre_edits)} pre_edits")
        
        # Add the changes applied by pre_edits to the staging area
        try:
            # Get all modified files
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            modified_files = result.stdout.strip().split('\n')
            modified_files = [f for f in modified_files if f.strip()]
            
            if modified_files:
                # Add all modified files to the staging area
                subprocess.run(
                    ["git", "add"] + modified_files,
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(f"Added {len(modified_files)} modified files to the staging area: {', '.join(modified_files[:5])}{'...' if len(modified_files) > 5 else ''}")
            else:
                logger.warning("No file modifications detected")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add files to the staging area: {e.stderr}")
            raise RuntimeError(f"Failed to add files to the staging area: {e.stderr}")


def get_all_python_files(repo_path: Path) -> List[Path]:
    """
    Get all Python files in the repository
    
    :param repo_path: repository path
    :return: Python file path list
    """
    python_files = []
    for py_file in repo_path.rglob("*.py"):
        # Exclude some common directories that do not need to be searched
        if any(excluded in str(py_file) for excluded in [".git", "__pycache__", ".venv", "venv", "env"]):
            continue
        python_files.append(py_file)
    return python_files


def tokenize_code(text: str) -> List[str]:
    """
    Tokenize the code text
    
    :param text: code text
    :return: token list
    """
    # Simple code tokenization: keep identifiers, keywords, etc.
    # Use regex to extract words, numbers, operators, etc.
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens


def bm25_retrieve_files(
    repo: str,
    base_commit: str,
    pre_edits: List[str],
    topn: int = 10,
    exclude_files: Set[str] = None
) -> Dict[str, str]:
    """
    Use BM25 to retrieve related files from the repository
    
    :param repo: repository name, format as "owner/repo"
    :param base_commit: target commit SHA
    :param pre_edits: history edit steps list
    :param topn: return top-n most related files
    :param exclude_files: file paths to exclude (files involved in pre_edits will be automatically included)
    :return: file path to file content dictionary
    """
    exclude_files = exclude_files or set()
    
    # Get the repository path
    repo_path = get_repo_path(repo)
    
    # Reset the repository to the target commit, and apply pre_edits
    reset_repo_to_commit(repo_path, base_commit, pre_edits=pre_edits)
    
    # Extract the files involved from pre_edits
    pre_edit_files = set()
    for pre_edit in pre_edits:
        files = extract_files_from_diff(pre_edit)
        pre_edit_files.update(files)
    
    # Extract the query text
    query_text = extract_query_text_from_pre_edits(pre_edits)
    query_tokens = tokenize_code(query_text)
    
    # Get all Python files
    all_python_files = get_all_python_files(repo_path)
    
    # Filter out excluded files and files involved in pre_edits (these will be added separately)
    candidate_files = []
    for py_file in all_python_files:
        # Convert to relative path to the repository root directory
        rel_path = str(py_file.relative_to(repo_path))
        if rel_path not in exclude_files and rel_path not in pre_edit_files:
            candidate_files.append((rel_path, py_file))
    
    # First add files involved in pre_edits (we know these files were edited, so they should be included first)
    file_context = {}
    for rel_path in pre_edit_files:
        file_path = repo_path / rel_path
        if file_path.exists():
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_context[rel_path] = content
                logger.info(f"Files involved in pre_edit: {rel_path}")
            except Exception as e:
                logger.warning(f"Failed to read pre_edit file {rel_path}: {e}")
    
    # Then add the retrieved files (exclude files already in file_context to avoid duplicates)
    if not candidate_files:
        logger.warning("No candidate files found for retrieval")
    else:
        # Read all candidate file contents and build BM25 index
        file_contents = []
        file_paths = []
        
        for rel_path, file_path in candidate_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                file_contents.append(content)
                file_paths.append(rel_path)
            except Exception as e:
                logger.warning(f"Failed to read file {rel_path}: {e}")
                continue
        
        if not file_contents:
            logger.warning("No file contents successfully read")
        else:
            # Build BM25 index
            tokenized_corpus = [tokenize_code(content) for content in file_contents]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Retrieve
            scores = bm25.get_scores(query_tokens)
            
            # Get all file scores and sort (by descending score)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            # Add top-n files, ensure these files are not in pre_edit_files
            # (candidate file list has already excluded pre_edit_files, but to be safe, we check again here)
            added_count = 0
            for idx in sorted_indices:
                if added_count >= topn:
                    break
                
                rel_path = file_paths[idx]
                # Ensure not in pre_edit_files (although candidate list has excluded pre_edit_files, we check again for safety)
                if rel_path not in pre_edit_files:
                    file_context[rel_path] = file_contents[idx]
                    added_count += 1
                    logger.info(f"Retrieved file: {rel_path} (score: {scores[idx]:.4f})")
            
            # record the final result
            logger.info(
                f"Retrieval completed: {len(pre_edit_files)} pre_edit files, {added_count} retrieved files, "
                f"final file total {len(file_context)} (target: pre_edit + {topn} retrieved files)"
            )
            
            # If the number of added files is less than topn, record a warning (maybe candidate files are not enough)
            if added_count < topn:
                logger.warning(
                    f"Expected to retrieve {topn} files, actually retrieved {added_count} files. "
                    f"Candidate file total: {len(candidate_files)}"
                )
    
    return file_context

