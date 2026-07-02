import json
import logging
import os
import re
import ssl
import subprocess
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Iterator, Union
from http.client import IncompleteRead, RemoteDisconnected
import requests
from bs4 import BeautifulSoup
from ghapi.core import GhApi
from tenacity import (
    retry,
    wait_fixed,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    retry_if_exception,
)
from fastcore.net import HTTP404NotFoundError, HTTP403ForbiddenError
from unidiff import PatchSet
from editbench.collection.instance.activity import write_json_line
# from editbench.config.constants import GITHUB_TOKEN
from editbench.config.constants import TAG_VERSION
# from editbench.utils.dataset_utils import get_inf_datasets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests
PR_KEYWORDS = {
    "close", "closes", "closed",
    "fix", "fixes", "fixed",
    "resolve", "resolves", "resolved", "related"
}

_MERGE_COMMIT_MESSAGE_PATTERNS = (
    re.compile(r"^Merge pull request #\d+", re.IGNORECASE),
    re.compile(r"^Merge branch ", re.IGNORECASE),
    re.compile(r"^Merge remote-tracking branch ", re.IGNORECASE),
)

_COMBINE_HUNK_HEADER = re.compile(
    r"^@{3} (?P<p1>[+-]\d+(?:,\d+)?) (?P<p2>[+-]\d+(?:,\d+)?) (?P<result>[+-]\d+(?:,\d+)?) @@@(?P<context>.*)$"
)

_GIT_CACHE_ROOT = Path(os.getenv("EDITBENCH_GIT_CACHE", "/tmp/editbench-git-cache"))


def _activity_text(val) -> str:
    """Coerce ghapi activity fields to str (body/title may be AttrDict when null)."""
    return val if isinstance(val, str) else ""


def _is_valid_issue_number(issue_num: str) -> bool:
    """Ignore template placeholders like closes #0000."""
    try:
        return int(issue_num) > 0
    except (TypeError, ValueError):
        return False


def _is_retryable_network_error(exc: BaseException) -> bool:
    """Retry true network failures, not HTTP 4xx/5xx from the GitHub API."""
    if isinstance(exc, urllib.error.HTTPError):
        return False
    return isinstance(
        exc,
        (
            ssl.SSLError,
            IncompleteRead,
            RemoteDisconnected,
            urllib.error.URLError,
        ),
    )


class Repo:
    def __init__(self,
                 owner: str,
                 name: str,
                 token: Optional[str] = None
                 ):
        self.owner = owner
        self.name = name
        self.token = token
        self.api = GhApi(token=token)
        self.repo = self.call_api(self.api.repos.get, owner=owner, repo=name)
        # self.releases = self.get_all_release_commits()

    @classmethod
    def from_full_name(cls, full_name: str, token: Optional[str] = None):
        return cls(full_name.split("/")[0], full_name.split("/")[1], token)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(15),
        retry=retry_if_exception(_is_retryable_network_error),
    )
    def _call_api_with_network_retry(self, func: Callable, **kwargs):
        return func(**kwargs)

    @retry(
        wait=wait_fixed(5 * 60),
        retry=retry_if_exception_type((HTTP403ForbiddenError,))
    )
    def call_api(self, func: Callable, **kwargs):
        """
        API call wrapper with rate limit handling (checks every 5 minutes if rate limit is reset)
        """
        try:
            return self._call_api_with_network_retry(func, **kwargs)
        except HTTP403ForbiddenError as e:
            rl = self.api.rate_limit.get()
            print(
                f"[{self.owner}/{self.name}] Rate limit exceeded for token {self.token[:10]}, "
                f"waiting for 5 minutes, remaining calls: {rl.resources.core.remaining}"
            )
            raise
        except HTTP404NotFoundError:
            logger.warning(
                "[%s/%s] Resource not found (skipping): %s",
                self.owner,
                self.name,
                kwargs,
            )
            return None

    def get_all_pulls(
            self,
            per_page: int = 100,
            num_pages: Optional[int] = None,
            direction: str = "desc",
            sort: str = "created",
            state: str = "closed",
            quiet: bool = False,
    ) -> Iterator:
        """
        Wrapper for API call to get all PRs from repo

        Args:
            per_page (int): number of PRs to return per page
            num_pages (int): number of pages to return
            direction (str): direction to sort PRs
            sort (str): field to sort PRs by
            state (str): state of PRs to look for
            quiet (bool): whether to print progress
        """
        pulls = self.get_all_loop(
            self.api.pulls.list,
            num_pages=num_pages,
            direction=direction,
            per_page=per_page,
            sort=sort,
            state=state,
            quiet=quiet,
        )
        return pulls

    def get_all_commits(self,
                        per_page: int = 100,
                        num_pages: Optional[int] = None,
                        sha: str = "main",
                        since: Optional[str] = None,
                        quiet: bool = False
                        ) -> Iterator:
        api_kwargs = {}
        if since:
            api_kwargs["since"] = since

        def try_get_branch(branch_name: str) -> bool:
            """Helper function to check if a branch exists"""
            try:
                # Try to get the branch reference
                self.api.repos.get_branch(self.owner, self.name, branch=branch_name)
                return True
            except Exception as e:
                if not quiet:
                    print(f"Branch {branch_name} not found: {e}")
                return False

        def get_default_branch() -> Union[str, None]:
            try:
                repo_info = self.api.repos.get(self.owner, self.name)
                default_branch_name = repo_info.default_branch
                self.api.repos.get_branch(self.owner, self.name, branch=default_branch_name)
                return default_branch_name
            except Exception as e:
                if not quiet:
                    error_msg = f"Default branch not found or repo error: {e}" \
                        if 'default_branch_name' in locals() else f"Failed to get repo info: {e}"
                    print(error_msg)
                return None

        # Determine which branch to use
        target_branch = sha
        if try_get_branch(target_branch):
            pass
        elif sha != "main" and not try_get_branch("main"):
            target_branch = "main"
        elif sha != "master" and not try_get_branch("master"):
            target_branch = "master"
        elif get_default_branch():
            target_branch = get_default_branch()
        else:
            raise ValueError(f"Neither {sha}, main or master or default branch branch exists in the repository")
        if not quiet:
            print(f"Using {target_branch} branch as fallback")
        # Get commits from the determined branch
        return self.get_all_loop(
            self.api.repos.list_commits,
            per_page=per_page,
            num_pages=num_pages,
            sha=target_branch,
            quiet=quiet,
            **api_kwargs
        )

    def get_commit_prs(self, commit_sha: str):
        try:
            prs = self.api.repos.list_pull_requests_associated_with_commit(
                owner=self.owner,
                repo=self.name,
                commit_sha=commit_sha,
            )
            if not prs:
                return None
            return prs[0]
        except Exception as e:
            logger.error(f"[{self.owner}/{self.name}] Error fetching PRs for commit {commit_sha}: {e}")
            return None

    def get_all_loop(
            self,
            func: Callable,
            per_page: int = 100,
            num_pages: Optional[int] = None,
            quiet: bool = False,
            **kwargs,
    ) -> Iterator:
        """
        Return all values from a paginated API endpoint.

        Args:
            func (callable): API function to call
            per_page (int): number of values to return per page
            num_pages (int): number of pages to return
            quiet (bool): whether to print progress
            **kwargs: keyword arguments to pass to API function
        """
        page = 1
        errors_on_page = 0
        max_errors_on_page = 10
        args = {
            "owner": self.owner,
            "repo": self.name,
            "per_page": per_page,
            **kwargs,
        }
        values = []
        while True:
            try:
                values = self.call_api(func, **args, page=page)
                if values is None:
                    break
                errors_on_page = 0
                yield from values
                if len(values) == 0:
                    break
                if not quiet:
                    rl = self.call_api(self.api.rate_limit.get)
                    logger.info(
                        f"[{self.owner}/{self.name}] Processed page {page} ({per_page} values per page). "
                        f"Remaining calls: {rl.resources.core.remaining}"
                    )
                if num_pages is not None and page >= num_pages:
                    break
                page += 1
            except Exception as e:
                errors_on_page += 1
                if errors_on_page >= max_errors_on_page:
                    logger.error(
                        f"[{self.owner}/{self.name}] Giving up on page {page} "
                        f"after {max_errors_on_page} errors: {e}"
                    )
                    raise
                logger.warning(
                    f"[{self.owner}/{self.name}] Error processing page {page} "
                    f"({errors_on_page}/{max_errors_on_page}): {e}. Retrying..."
                )
                time.sleep(min(30 * errors_on_page, 120))
        if not quiet:
            logger.info(
                f"[{self.owner}/{self.name}] Processed {(page - 1) * per_page + len(values)} values"
            )

    def extract_resolved_issues(self, activity: dict) -> list[str]:
        """
        Extract list of issues referenced by a commit or PR

        """
        # Define 1. issue number regex pattern 2. comment regex pattern 3. keywords
        issues_pat = re.compile(r"(\w+)\s+\#(\d+)")
        comments_pat = re.compile(r"(?s)<!--.*?-->")

        # Construct text to search over for issue numbers from PR body and commit messages
        text = ""
        if activity["src_type"] == "commit":
            text = _activity_text(activity["commit"]["message"])
        elif activity["src_type"] == "pull":
            text = _activity_text(activity.get("title_mes") or activity.get("title"))
            text += "\n" + _activity_text(activity.get("body_mes") or activity.get("body"))
            commits = self.get_all_loop(
                self.api.pulls.list_commits, pull_number=activity["number"], quiet=True
            )
            commit_messages = [commit.commit.message for commit in commits]
            commit_text = "\n".join(commit_messages) if commit_messages else ""
            text += "\n" + commit_text
        # Remove comments from text
        text = comments_pat.sub("", text)
        # Look for issue numbers in text via scraping <keyword, number> patterns
        references = issues_pat.findall(text)
        resolved_issues_set = set()
        if references:
            for word, issue_num in references:
                if not _is_valid_issue_number(issue_num):
                    continue
                if any(keyword in word.lower() for keyword in PR_KEYWORDS):
                    resolved_issues_set.add(issue_num)
        return list(resolved_issues_set)

    def get_all_release_commits(self):
        """
        get all release
        """
        releases = self.get_all_loop(self.api.repos.list_releases)
        release_commits = []
        for rel in releases:
            tag_name = rel.tag_name
            try:
                ref = self.api.git.get_ref(owner=self.owner, repo=self.name, ref=f"tags/{tag_name}")
                tag_target_sha = ref.object.sha

                if ref.object.type == "tag":
                    tag_obj = self.api.git.get_tag(owner=self.owner, repo=self.name, tag_sha=tag_target_sha)
                    commit_sha = tag_obj.object.sha
                else:
                    commit_sha = tag_target_sha

                commit_info = self.api.git.get_commit(owner=self.owner, repo=self.name, commit_sha=commit_sha)
                commit_time = datetime.strptime(
                    commit_info.author["date"],
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                commit_time = commit_time.strftime("%Y-%m-%d %H:%M:%S")

                release_commits.append({
                    "tag": tag_name,
                    "commit_sha": commit_sha,
                    "created_at": commit_time,
                    "release_url": rel.html_url
                })
            except requests.exceptions.HTTPError as e:
                print(f"Warning：process tag {tag_name} - error: {str(e)}")
                continue

        return sorted(release_commits, key=lambda x: x["created_at"], reverse=True)


def get_version_and_commit(repo, commit):
    release_path = Path(TAG_VERSION) / f"{repo.owner}-{repo.name}-version.jsonl"
    all_releases = []
    if os.path.exists(release_path):
        with open(release_path, "r") as fr:
            for rel in fr:
                release = json.loads(rel)
                all_releases.append(release)
    else:
        print(f"No release information found for {repo.owner}/{repo.name}, fetching from API...")
        all_releases = repo.get_all_release_commits()
        print(f"Fetched {len(all_releases)} release commits for {repo.owner}/{repo.name}")
        with open(release_path, "w") as f:
            for release in all_releases:
                json_line = json.dumps(release)
                print(json_line, end="\n", file=f, flush=True)
    release = find_nearest_release(repo, all_releases, commit)
    tag = ""
    commit_version = ""
    if release:
        tag = release["tag"]
        match = re.search(r'v(\d+\.\d+)', tag)
        if match:
            tag = match.group(1)
        commit_version = release["commit_sha"]

    return tag, commit_version


def resolve_version_for_commit(repo: "Repo", base_commit: str) -> tuple[str, str]:
    """Return (version, version_commit) for a repository commit."""
    return get_version_and_commit(repo, base_commit)


def find_nearest_release(
        repo,
        releases,
        target_commit_sha: str,
) -> Union[dict, None]:
    """
    Find the nearest released version for the target
    commit (where the release commit is an ancestor of the target commit)
    :param target_commit_sha: target_commi
    :return: Dictionary of the nearest release information (or None)
    """
    try:
        target_commit = repo.api.git.get_commit(repo.owner, repo.name, target_commit_sha)
        commit_time = datetime.strptime(
            target_commit.author["date"],
            "%Y-%m-%dT%H:%M:%SZ"
        )
        commit_time = commit_time.strftime("%Y-%m-%d %H:%M:%S")
        # iterate through release list from new to old
        for release in releases:
            if commit_time >= release["created_at"]:
                return release
    except Exception as e:
        print("target version not found")

    return None


def extract_problem_statement_and_hints(pull: dict, repo: Repo) -> tuple[str, list]:
    """
    Extract problem statement from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    if repo.name == "django":
        return extract_problem_statement_and_hints_django(pull, repo)
    text = ""
    all_hint_texts = list()
    for issue_number in pull.get("resolved_issues", []):
        if not _is_valid_issue_number(str(issue_number)):
            continue
        issue = repo.call_api(
            repo.api.issues.get,
            owner=repo.owner,
            repo=repo.name,
            issue_number=int(issue_number),
        )
        if issue is None:
            continue
        title = issue.title if issue.title else ""
        body = issue.body if issue.body else ""
        text += f"{title}\n{body}\n"
        issue_number = issue.number
        hint_texts = _extract_hints(pull, repo, issue_number)
        hint_text = "\n".join(hint_texts)
        all_hint_texts.append(hint_text)
    return text, all_hint_texts


def _extract_hints(pull: dict, repo: Repo, issue_number: int) -> list[str]:
    """
    Extract hints from comments associated with a pull request (before first commit)

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
        issue_number (int): issue number
    Return:
        hints (list): list of hints
    """
    src_type = pull["src_type"]
    commits = []
    if src_type == "pull":
        # Get all commits in PR
        commits = repo.get_all_loop(
            repo.api.pulls.list_commits, pull_number=pull["number"], quiet=True
        )
        commits = list(commits)
    else:
        commits.append(pull)

    if len(commits) == 0:
        # If there are no comments, return no hints
        return []
    # Get time of first commit in PR
    commit_time = commits[0]["commit"]["author"]["date"]  # str
    commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))
    # Get all comments in PR
    all_comments = repo.get_all_loop(
        repo.api.issues.list_comments, issue_number=issue_number, quiet=True
    )
    all_comments = list(all_comments)
    # Iterate through all comments, only keep comments created before first commit
    comments = list()
    for comment in all_comments:
        comment_time = time.mktime(
            time.strptime(comment.updated_at, "%Y-%m-%dT%H:%M:%SZ")
        )  # use updated_at instead of created_at
        if comment_time < commit_time:
            comments.append(comment)
        else:
            break
        # only include information available before the first commit was created
    # Keep text from comments
    comments = [comment.body for comment in comments]
    return comments


def extract_patches(pull: dict, repo: Repo) -> tuple[str, str]:
    """
    Get patch and test patch from PR

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        patch_change_str (str): gold patch
        patch_test_str (str): test patch
    """
    patch = requests.get(pull["diff_url"]).text
    patch_test = ""
    patch_fix = ""
    for hunk in PatchSet(patch):
        if any(
                test_word in hunk.path for test_word in ["test", "tests", "e2e", "testing"]
        ):
            patch_test += str(hunk)
        else:
            patch_fix += str(hunk)
    return patch_fix, patch_test


### MARK: Repo Specific Parsing Functions ###
def extract_problem_statement_and_hints_django(
        pull: dict, repo: Repo
) -> tuple[str, list]:
    """
    Get problem statement and hints from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    text = ""
    all_hints_text = list()
    for issue_number in pull["resolved_issues"]:
        url = f"https://code.djangoproject.com/ticket/{issue_number}"
        resp = requests.get(url)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get problem statement (title + body)
        issue_desc = soup.find("div", {"id": "ticket"})
        title = issue_desc.find("h1", class_="searchable").get_text()
        title = re.sub(r"\s+", " ", title).strip()
        body = issue_desc.find("div", class_="description").get_text()
        body = re.sub(r"\n+", "\n", body)
        body = re.sub(r"    ", "\t", body)
        body = re.sub(r"[ ]{2,}", " ", body).strip()
        text += f"{title}\n{body}\n"

        # Get time of first commit in PR
        commits = repo.get_all_loop(
            repo.api.pulls.list_commits, pull_number=pull["number"], quiet=True
        )
        commits = list(commits)
        if len(commits) == 0:
            continue
        commit_time = commits[0].commit.author.date
        commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))

        # Get all comments before first commit
        comments_html = soup.find("div", {"id": "changelog"})
        div_blocks = comments_html.find_all("div", class_="change")
        # Loop through each div block
        for div_block in div_blocks:
            # Find the comment text and timestamp
            comment_resp = div_block.find("div", class_="comment")
            timestamp_resp = div_block.find("a", class_="timeline")
            if comment_resp is None or timestamp_resp is None:
                continue

            comment_text = re.sub(r"\s+", " ", comment_resp.text).strip()
            timestamp = timestamp_resp["title"]
            if timestamp.startswith("See timeline at "):
                timestamp = timestamp[len("See timeline at "):]
            if "/" in timestamp:
                timestamp = time.mktime(time.strptime(timestamp, "%m/%d/%y %H:%M:%S"))
            elif "," in timestamp:
                timestamp = time.mktime(
                    time.strptime(timestamp, "%b %d, %Y, %I:%M:%S %p")
                )
            else:
                raise ValueError(f"Timestamp format not recognized: {timestamp}")

            # Append the comment and timestamp as a tuple to the comments list
            if timestamp < commit_time:
                all_hints_text.append((comment_text, timestamp))

    return text, all_hints_text


def classify_files(file_list: list[dict]):
    """
    Classify file based on files_dict from GitHub API.
    """
    test_keywords = {'test', 'tests', 'e2e', 'testing'}
    config_keywords = {'config', 'settings'}
    config_extensions = {'.yaml', '.json', '.toml', '.ini', '.xml'}
    source_extensions = {'.py', '.java', '.js', '.cpp', '.c', '.go', '.ts', '.php', '.vue'}

    files, files_work, files_test, files_no_edit, files_gt, files_config, files_other = [], [], [], [], [], [], []

    for file in file_list:
        file_path = file["filename"]
        status = file.get('status', '')

        files.append(file_path)

        if status != "modified":
            files_no_edit.append(file_path)
            continue

        is_test = (any(kw in file_path.lower() for kw in test_keywords) and
                   any(file_path.lower().endswith(ext) for ext in source_extensions))
        is_config = (any(kw in file_path.lower() for kw in config_keywords) or
                     any(file_path.lower().endswith(ext) for ext in config_extensions))
        is_source = any(file_path.lower().endswith(ext) for ext in source_extensions)

        if is_test or is_source or is_config:
            files_gt.append(file_path)

        if is_test:
            files_test.append(file_path)
        elif is_source:
            files_work.append(file_path)
        elif is_config:
            files_config.append(file_path)
        else:
            files_other.append(file_path)

    return files, files_work, files_test, files_no_edit, files_gt, files_config, files_other


def _commit_info_message(commit_info) -> str:
    try:
        if isinstance(commit_info, dict):
            return (commit_info.get("commit") or {}).get("message", "") or ""
        return commit_info.commit.message or ""
    except Exception:
        return ""


def _commit_has_nonempty_file_patch(commit_info) -> bool:
    for file in _commit_info_files(commit_info):
        if _file_patch_text(file):
            return True
    return False


def is_redundant_merge_commit(commit_info) -> bool:
    """
    Return True for merge commits that should not become work_patch_list steps,
    e.g. "Merge pull request #10 from user/branch".

    Merge commits with non-empty file patches (e.g. conflict resolution) are kept.
    """
    message = _commit_info_message(commit_info).strip()
    if not message:
        return False
    first_line = message.splitlines()[0].strip()
    if not any(pattern.match(first_line) for pattern in _MERGE_COMMIT_MESSAGE_PATTERNS):
        return False
    return not _commit_has_nonempty_file_patch(commit_info)


def _file_patch_text(file) -> str:
    if isinstance(file, dict):
        return file.get("patch") or ""
    return getattr(file, "patch", None) or ""


def _format_file_patch(filename: str, patch_body: str) -> str:
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"--- a/{filename}\n+++ b/{filename}\n{patch_body}"
    )


def _commit_info_files(commit_info) -> list:
    try:
        if isinstance(commit_info, dict):
            return commit_info.get("files") or []
        return getattr(commit_info, "files", None) or []
    except Exception:
        return []


def _file_entry_filename(file) -> str:
    if isinstance(file, dict):
        return file["filename"]
    return file.filename


def _file_entry_status(file) -> str:
    if isinstance(file, dict):
        return file.get("status", "")
    return getattr(file, "status", "") or ""


def _commit_sha(commit_info) -> str:
    if isinstance(commit_info, dict):
        return commit_info.get("sha") or ""
    return getattr(commit_info, "sha", "") or ""


def _commit_parents(commit_info) -> list:
    if isinstance(commit_info, dict):
        parents = commit_info.get("parents") or []
        return [p.get("sha", "") for p in parents if p.get("sha")]
    parents = getattr(commit_info, "parents", None) or []
    return [getattr(p, "sha", "") for p in parents if getattr(p, "sha", None)]


def _is_merge_commit(commit_info) -> bool:
    return len(_commit_parents(commit_info)) >= 2


def _git_cache_dir(repo: "Repo") -> Path:
    return _GIT_CACHE_ROOT / f"{repo.owner}__{repo.name}.git"


def _ensure_bare_git_repo(repo: "Repo") -> Path:
    git_dir = _git_cache_dir(repo)
    git_dir.parent.mkdir(parents=True, exist_ok=True)
    if not git_dir.exists():
        clone_url = f"https://github.com/{repo.owner}/{repo.name}.git"
        subprocess.run(
            ["git", "clone", "--bare", "--quiet", clone_url, str(git_dir)],
            check=True,
        )
    else:
        subprocess.run(["git", "fetch", "--quiet"], cwd=str(git_dir), check=True)
    return git_dir


def _combine_hunk_line_to_unified(hline: str) -> Optional[str]:
    """Map one ``diff --cc`` hunk line to unified-diff form (first parent → merge)."""
    if len(hline) < 2:
        return None
    col1, col2 = hline[0], hline[1]
    content = hline[2:]
    if col1 == "+" and col2 == "+":
        return "+" + content
    if col1 == "-" and col2 == " ":
        return "-" + content
    if col1 == " " and col2 == "-":
        return None
    if col1 == " " and col2 == " ":
        return " " + content
    return None


def _combine_diff_to_unified(filename: str, combine_patch: str) -> str:
    """
    Convert ``git show`` combine diff (``diff --cc``) to a unified diff against
    the first parent, matching the per-commit view on GitHub.
    """
    if not combine_patch.strip().startswith("diff --cc"):
        return ""

    lines = combine_patch.splitlines()
    body_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git") or line.startswith("index ") or line.startswith("---") or line.startswith("+++"):
            i += 1
            continue
        match = _COMBINE_HUNK_HEADER.match(line)
        if not match:
            i += 1
            continue
        body_lines.append(
            f"@@ {match.group('p1')} {match.group('result')} @@{match.group('context')}"
        )
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            unified_line = _combine_hunk_line_to_unified(lines[i])
            if unified_line is not None:
                body_lines.append(unified_line)
            i += 1
    if not body_lines:
        return ""
    return _format_file_patch(filename, "\n".join(body_lines) + "\n")


def _get_merge_file_patch(repo: "Repo", commit_sha: str, filename: str) -> str:
    if not commit_sha:
        return ""
    try:
        git_dir = _ensure_bare_git_repo(repo)
        combine_patch = subprocess.check_output(
            [
                "git",
                "--git-dir",
                str(git_dir),
                "show",
                "--format=",
                "--no-renames",
                "-p",
                commit_sha,
                "--",
                filename,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning(
            "Failed to read combine diff for %s/%s@%s (%s): %s",
            repo.owner,
            repo.name,
            commit_sha[:7],
            filename,
            exc,
        )
        return ""
    return _combine_diff_to_unified(filename, combine_patch)


def get_patches(file_list, related_commits_info, files_work, files_test, repo: Optional["Repo"] = None):
    whole_patch = []
    work_patch_list = [[] for _ in range(len(files_work))]
    test_patch = []
    for file in file_list:
        if _file_entry_status(file) == "modified":
            filename = _file_entry_filename(file)
            patch_body = _file_patch_text(file)
            if not patch_body:
                continue
            patch = _format_file_patch(filename, patch_body)
            if filename in files_work:
                whole_patch.append(patch)
            if filename in files_test:
                test_patch.append(patch)
    whole_patch = "\n".join(whole_patch)
    test_patch = "\n".join(test_patch)

    for commit_info in related_commits_info:
        if is_redundant_merge_commit(commit_info):
            continue
        commit_sha = _commit_sha(commit_info)
        for file in _commit_info_files(commit_info):
            filename = _file_entry_filename(file)
            if filename not in files_work:
                continue
            if _is_merge_commit(commit_info) and repo is not None:
                patch = _get_merge_file_patch(repo, commit_sha, filename)
                if not patch:
                    continue
                index = files_work.index(filename)
                work_patch_list[index].append(patch)
                continue
            patch_body = _file_patch_text(file)
            if not patch_body:
                continue
            index = files_work.index(filename)
            work_patch_list[index].append(_format_file_patch(filename, patch_body))

    return whole_patch, work_patch_list, test_patch


def rebuild_work_patch_list(activity, repo) -> list:
    """
    Re-fetch PR/commit patches and rebuild work_patch_list with current rules.
    """
    files_work = activity.files_work
    files_test = activity.files_test

    if activity.src_type == "pull":
        pull_number = int(activity.instance_num)
        related_files = list(
            repo.get_all_loop(repo.api.pulls.list_files, pull_number=pull_number)
        )
        related_commits = list(
            repo.get_all_loop(repo.api.pulls.list_commits, pull_number=pull_number)
        )
        related_commits_info = [
            repo.call_api(
                repo.api.repos.get_commit,
                owner=repo.owner,
                repo=repo.name,
                ref=related_commit["sha"],
            )
            for related_commit in related_commits
        ]
    elif activity.src_type == "commit":
        commit_info = repo.call_api(
            repo.api.repos.get_commit,
            owner=repo.owner,
            repo=repo.name,
            ref=activity.instance_num,
        )
        related_files = _commit_info_files(commit_info)
        related_commits_info = [commit_info]
    else:
        raise ValueError(f"Unsupported src_type: {activity.src_type}")

    _, work_patch_list, _ = get_patches(
        related_files, related_commits_info, files_work, files_test, repo=repo
    )
    return work_patch_list
