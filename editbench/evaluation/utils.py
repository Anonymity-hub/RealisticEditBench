import json
import time
import os
import os.path
import re
from functools import cache
from pathlib import Path
from typing import Set, List

import requests
from requests.exceptions import SSLError, ConnectionError, Timeout, RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from editbench.collection.instance.activity import Activity
from editbench.config import GITHUB_RAW_PROXY, GITHUB_RAW, SRC_INF_BENCHMARK_DATA
from editbench.evaluation.constants import MAP_REPO_TO_REQS_PATHS, MAP_REPO_TO_ENV_YML_PATHS, NON_TEST_EXTS

def _get_proxy_config():
    """
    Optional HTTP proxy for fetching GitHub files during evaluation image builds.

    Priority: HTTP_PROXY/HTTPS_PROXY (or lowercase variants), then GITHUB_RAW_PROXY
    from .env. If none are set, returns None and requests go directly to GitHub.
    """
    http_proxy = (
        os.getenv('HTTP_PROXY')
        or os.getenv('http_proxy')
        or GITHUB_RAW_PROXY
    )
    https_proxy = (
        os.getenv('HTTPS_PROXY')
        or os.getenv('https_proxy')
        or GITHUB_RAW_PROXY
    )

    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy

    return proxies if proxies else None


def load_inf_results(inf_results: str):
    with open(inf_results, 'r', encoding='utf-8') as file:
        inf_results = []
        for line in file:
            data = json.loads(line.strip())
            inf_results.append(data)

        return inf_results


def find_instance(inf_instances, task_id):
    for instance in inf_instances:
        if instance.task_id == task_id:
            return instance


def get_requirements(instance: Activity) -> str:
    """
    Get requirements.txt for given task instance

    Args:
        instance (dict): task instance
    Returns:
        requirements.txt (str): Returns requirements.txt as string
    """
    # Attempt to find requirements.txt at each path based on task instance's repo
    commit = (
        instance.version_commit
        if instance.version_commit != ""
        else instance.base_commit
    )

    return get_requirements_by_commit(f"{instance.repo}", commit)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((SSLError, ConnectionError, Timeout, RequestException))
)
def _safe_get(url: str, stream: bool = False, timeout: int = 30):
    """safe HTTP GET request, with retry mechanism and proxy support"""
    proxies = _get_proxy_config()
    return requests.get(url, stream=stream, timeout=timeout, proxies=proxies)


@cache
def get_requirements_by_commit(repo: str, commit: str) -> str:
    for req_path in MAP_REPO_TO_REQS_PATHS[repo]:
        reqs_url = os.path.join(f"https://{GITHUB_RAW}", repo, commit, req_path)
        try:
            reqs = _safe_get(reqs_url, stream=True)
            if reqs.status_code == 200:
                break
        except (SSLError, ConnectionError, Timeout, RequestException) as e:
            # if it is the last path, raise an exception
            if req_path == MAP_REPO_TO_REQS_PATHS[repo][-1]:
                raise ValueError(
                    f"Could not fetch requirements.txt at paths {MAP_REPO_TO_REQS_PATHS[repo]} "
                    f"for repo {repo} at commit {commit}. Error: {e}"
                )
            # otherwise, try the next path
            continue
    else:
        raise ValueError(
            f"Could not find requirements.txt at paths {MAP_REPO_TO_REQS_PATHS[repo]} for repo {repo} at commit {commit}"
        )

    lines = reqs.text
    original_req = []  # from the "requirements.txt"
    additional_reqs = []  # from the -r new file
    req_dir = "/".join(req_path.split("/")[:-1])
    exclude_line = lambda line: any(
        [line.strip().startswith(x) for x in ["-e .", "#", ".[test"]]
    )

    for line in lines.split("\n"):
        if line.strip().startswith("-r"):
            # Handle recursive requirements
            file_name = line[len("-r"):].strip()
            reqs_url = os.path.join(
                f"https://{GITHUB_RAW}",
                repo,
                commit,
                req_dir,
                file_name,
            )
            try:
                reqs = _safe_get(reqs_url, stream=False)
                if reqs.status_code == 200:
                    for line_extra in reqs.text.split("\n"):
                        if not exclude_line(line_extra):
                            additional_reqs.append(line_extra)
            except (SSLError, ConnectionError, Timeout, RequestException) as e:
                # if fetching recursive requirements fails, record warning but continue processing
                print(f"Warning: Failed to fetch recursive requirements from {reqs_url}: {e}")
                continue
        else:
            if not exclude_line(line):
                original_req.append(line)

    # Combine all requirements into single text body
    additional_reqs.append("\n".join(original_req))
    all_reqs = "\n".join(additional_reqs)

    return all_reqs


def get_environment_yml(instance: Activity, env_name: str) -> str:
    """
    Get environment.yml for given task instance

    Args:
        instance (dict): Edit Bench Task instance
        env_name (str): Rename retrieved environment.yml to this name
    Returns:
        environment.yml (str): Returns environment.yml as string
    """
    # Attempt to find environment.yml at each path based on task instance's repo

    commit = (
        instance.version_commit
        if instance.version_commit
        else instance.base_commit
    )

    return get_environment_yml_by_commit(instance.repo, commit, env_name)


@cache
def get_environment_yml_by_commit(repo: str, commit: str, env_name: str) -> str:
    for req_path in MAP_REPO_TO_ENV_YML_PATHS[repo]:
        reqs_url = os.path.join(f"https://{GITHUB_RAW}", repo, commit, req_path)
        try:
            reqs = _safe_get(reqs_url, stream=True)
            if reqs.status_code == 200:
                break
        except (SSLError, ConnectionError, Timeout, RequestException) as e:
            # if it is the last path, raise an exception
            if req_path == MAP_REPO_TO_ENV_YML_PATHS[repo][-1]:
                raise ValueError(
                    f"Could not fetch environment.yml at paths {MAP_REPO_TO_ENV_YML_PATHS[repo]} "
                    f"for repo {repo} at commit {commit}. Error: {e}"
                )
            # otherwise, try the next path
            continue
    else:
        raise ValueError(
            f"Could not find environment.yml at paths {MAP_REPO_TO_ENV_YML_PATHS[repo]} for repo {repo} at commit {commit}"
        )

    lines = reqs.text.split("\n")
    cleaned = []
    for line in lines:
        # rename environment to given name
        if line.startswith("name:"):
            cleaned.append(f"name: {env_name}")
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def get_test_directives(instance: Activity) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # For seq2seq code repos, testing command is fixed
    # if instance["repo"] == "swe-bench/humaneval":
    #     return ["test.py"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance.test_patch
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance.repo == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/"):] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives


def remove_docker_images_by_sampled_ids(
    client,
    sampled_ids_file_1: Path = None,
    sampled_ids_file_2: Path = None,
    sampled_ids_file_3: Path = None,
    logger=None,
) -> dict:
    """
    Delete Docker eval images for instance IDs listed in sampled JSON files.

    Image name format: editb.eval.x86_64.{instance_id}:latest
    """
    if logger:
        log_info = logger.info
        log_error = logger.error
    else:
        log_info = print
        log_error = print

    if sampled_ids_file_1 is None:
        sampled_ids_file_1 = SRC_INF_BENCHMARK_DATA / "sampled_instance_ids_0.2.json"
    if sampled_ids_file_2 is None:
        sampled_ids_file_2 = SRC_INF_BENCHMARK_DATA / "union_gpt-5-codex_claude_0.05_0.05.json"
    if sampled_ids_file_3 is None:
        sampled_ids_file_3 = SRC_INF_BENCHMARK_DATA / "union_gpt-5-codex_claude_0.1_0.1.json"

    all_instance_ids: Set[str] = set()
    file_stats = {}

    for file_path, file_name in [
        (sampled_ids_file_1, sampled_ids_file_1.name),
        (sampled_ids_file_2, sampled_ids_file_2.name),
        (sampled_ids_file_3, sampled_ids_file_3.name),
    ]:
        if file_path and file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    instance_ids = data.get("sampled_instance_ids", [])
                    if instance_ids:
                        all_instance_ids.update(instance_ids)
                        file_stats[file_name] = len(instance_ids)
                        log_info(f"read {len(instance_ids)} instance_ids from {file_name}")
                    else:
                        log_info(f"warning: {file_name} does not have sampled_instance_ids field")
            except Exception as e:
                log_error(f"failed to read file {file_path}: {e}")
        else:
            log_info(f"file not found, skip: {file_path}")

    log_info(f"\ntotal {len(all_instance_ids)} unique instance_ids (union)")

    removed_count = 0
    not_found_count = 0
    error_count = 0

    for instance_id in all_instance_ids:
        image_name = f"editb.eval.x86_64.{instance_id}:latest"
        try:
            from editbench.evaluation.docker_utils import remove_image

            remove_image(client, image_name, logger=logger if logger else "quiet")
            removed_count += 1
            if logger:
                log_info(f"deleted image: {image_name}")
        except Exception as e:
            if "not found" in str(e).lower() or "ImageNotFound" in str(type(e).__name__):
                not_found_count += 1
            else:
                error_count += 1
                log_error(f"failed to delete image {image_name}: {e}")

    result = {
        "total_instance_ids": len(all_instance_ids),
        "removed_count": removed_count,
        "not_found_count": not_found_count,
        "error_count": error_count,
        "file_stats": file_stats,
    }

    log_info("\ndeletion statistics:")
    log_info(f"  - total instance_ids: {result['total_instance_ids']}")
    log_info(f"  - successfully deleted: {result['removed_count']}")
    log_info(f"  - not found: {result['not_found_count']}")
    log_info(f"  - error: {result['error_count']}")

    return result
