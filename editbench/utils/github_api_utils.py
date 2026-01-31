"""
GitHub API utility functions for fetching repository data.

This module provides functions to interact with GitHub's API, including:
- Fetching raw file content from repositories
- Retrieving commit information
- Managing repository tags and versions
"""

import base64
import logging
from typing import Optional, Dict, Any, List, Tuple

from dateutil import parser
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from editbench.config import *


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default headers for HTTP requests
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
}


@retry(stop=stop_after_attempt(30), wait=wait_fixed(1))
def get_raw_url(repo_owner: str, repo_name: str, file_path: str, commit: str) -> str:
    """
    Fetch raw file content from a GitHub repository at a specific commit.

    Args:
        repo_owner: Repository owner username
        repo_name: Repository name
        file_path: Path to the file within the repository
        commit: Commit SHA or reference

    Returns:
        File content as a string

    Raises:
        requests.HTTPError: If the request fails
    """
    raw_url = f"https://{GITHUB_RAW}/{repo_owner}/{repo_name}/{commit}/{file_path}"
    response = requests.get(raw_url, stream=True, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    content = ''
    for chunk in response.iter_content(chunk_size=1024):
        content += chunk.decode('utf-8')
    return content


def get_commit_info(repo_owner: str, repo_name: str, access_token: Optional[str], commit_sha: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific commit.

    Args:
        repo_owner: Repository owner username
        repo_name: Repository name
        access_token: GitHub personal access token (optional, improves rate limits)
        commit_sha: Commit SHA hash

    Returns:
        Commit information as a dictionary

    Raises:
        Exception: If the API request fails
    """
    url = f"https://{API_GITHUB}/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
    headers = {"Authorization": f"token {access_token}"} if access_token else {}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch commit info: {response.status_code} - {response.text}")


def get_all_tags(repo_owner: str, repo_name: str, access_token: Optional[str]) -> List[Dict[str, Any]]:
    """
    Fetch all tags for a repository.

    Args:
        repo_owner: Repository owner username
        repo_name: Repository name
        access_token: GitHub personal access token (optional, improves rate limits)

    Returns:
        List of tag information dictionaries

    Raises:
        Exception: If the API request fails
    """
    url = f"https://{API_GITHUB}/repos/{repo_owner}/{repo_name}/tags"
    headers = {"Authorization": f"token {access_token}"} if access_token else {}
    tags = []

    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            tags.extend(response.json())
            if "next" in response.links:
                url = response.links["next"]["url"]
            else:
                url = None
        else:
            raise Exception(f"Failed to fetch tags: {response.status_code} - {response.text}")

    return tags


def find_tag_interval(commit_date: str, tags: List[Dict[str, Any]]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find the tags surrounding a specific commit date.

    Args:
        commit_date: ISO format date string
        tags: List of tag dictionaries

    Returns:
        Tuple of (previous_tag, next_tag) where tags can be None
    """
    previous_tag = None
    next_tag = None

    for tag in tags:
        tag_commit_url = tag["commit"]["url"]
        tag_commit_info = requests.get(tag_commit_url).json()
        tag_date = tag_commit_info["commit"]["committer"]["date"]

        if parser.isoparse(tag_date) <= parser.isoparse(commit_date):
            previous_tag = tag
            break
        else:
            next_tag = tag

    return previous_tag, next_tag


def get_version_tag(repo_owner, repo_name, access_token, commit_sha):
    commit_info = get_commit_info(repo_owner, repo_name, access_token, commit_sha)
    commit_date = commit_info["commit"]["committer"]["date"]

    tags_info = get_all_tags(repo_owner, repo_name, access_token)

    previous_tag, next_tag = find_tag_interval(commit_date, tags_info)

    if previous_tag:
        print(f"Commit {commit_sha} is after tag: {previous_tag['name']}")
    else:
        print(f"Commit {commit_sha} is before any tag.")

    if next_tag:
        print(f"Commit {commit_sha} is before tag: {next_tag['name']}")
    else:
        print(f"Commit {commit_sha} is after all tags.")


def get_github_file_content(
        owner: str,
        repo: str,
        commit_sha: str,
        file_path: str,
        access_token: Optional[str] = None
) -> Optional[str]:
    """
    Retrieve content of a specific file from a given commit in a GitHub repository via the GitHub API.

    Args:
        owner: Username of the GitHub repository owner (e.g., "pytorch")
        repo: Name of the repository (e.g., "pytorch")
        commit_sha: SHA hash of the target commit (e.g., "5281337d64b59f41d8f8f8b3f9d834426c8e6d03")
        file_path: Path to the file within the repository (e.g., "README.md")
        access_token: GitHub personal access token (optional, improves rate limits)

    Returns:
        File content as a string; None if retrieval fails
    """
    # build API request URL
    url = f"https://{API_GITHUB}/repos/{owner}/{repo}/contents/{file_path}"
    params = {"ref": commit_sha}  # specify target commit

    # configure request headers (User-Agent is required by GitHub API)
    headers = {"User-Agent": "GitHub File Fetcher"}
    if access_token:
        headers["Authorization"] = f"token {access_token}"

    try:
        response = requests.get(url, headers=headers, params=params)

        # check response status code (200 means success)
        if response.status_code == 200:
            # GitHub returns content is Base64 encoded content
            content_b64 = response.json().get("content", "")
            content_bytes = base64.b64decode(content_b64)
            return content_bytes.decode("utf-8")

        # non 200 status code (e.g. 404 file not found, 403 permission denied, etc.)
        return None

    # handle network request exception (e.g. connection timeout, DNS error, etc.)
    except requests.exceptions.RequestException:
        return None

