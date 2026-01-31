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

# get proxy configuration: prioritize environment variables, otherwise use OrbStack default gateway's proxy port
def _get_proxy_config():
    """
    get proxy configuration
    
    priority:
    1. environment variables HTTP_PROXY/HTTPS_PROXY
    2. environment variables http_proxy/https_proxy
    
    note: if using OrbStack, need to allow LAN connection in Mac proxy software
    """
    # prioritize environment variables
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
    
    # if environment variables are not set, try to use OrbStack to access Mac host's proxy
    if not http_proxy and not https_proxy:
        # in OrbStack, can access Mac host via host.docker.internal or host.orb.internal
        # prioritize host.docker.internal, this is the standard way for Docker/OrbStack
        proxy_hosts = ['host.docker.internal', 'host.orb.internal']
        
        # also try to get gateway IP as a backup
        gateway_ip = None
        try:
            import subprocess
            result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                parts = result.stdout.split()
                if len(parts) >= 3:
                    gateway_ip = parts[2]
        except Exception:
            pass
        
        # build possible proxy address list
        proxy_candidates = []
        for host in proxy_hosts:
            proxy_candidates.append(f"http://{host}:7890")
        if gateway_ip:
            proxy_candidates.append(f"http://{gateway_ip}:7890")
        
        # use first candidate address (usually host.docker.internal is the most reliable)
        if proxy_candidates:
            proxy_url = proxy_candidates[0]
            http_proxy = proxy_url
            https_proxy = proxy_url
    
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
