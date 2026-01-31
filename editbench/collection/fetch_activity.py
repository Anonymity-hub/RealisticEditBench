import json
import os.path
from datetime import datetime
from typing import Optional

from fastcore.xtras import obj2dict

from editbench.collection.utils import Repo


def fetch_repo_activity(
        repo_name: str,
        pull_output: str,
        token: Optional[str] = None,
        max_tasks: Optional[int] = None,
        cutoff_date: Optional[str] = None
):
    """
    Fetch repository activity data including commits and pull requests.

    Retrieves GitHub repository activity data and saves commits and pull requests
    to specified output files.

    Filtering Criteria:
    1. Commit content from the default branch, or Pull Requests (PRs)
    associated with these commits (with PR deduplication).
    2. All Pull Requests (with deduplication applied consistently with Step 1).

    Args:
        repo_name: Repository name in 'owner/repo' format (e.g., 'facebook/react')
        pull_output: File path to save pull requests data (JSON format recommended)
        token: GitHub API token for authenticated requests (optional)
        max_tasks: Maximum number of tasks/items to fetch (optional)
        cutoff_date: Only fetch activity after this date (YYYY-MM-DD format, optional)

    Returns:
        None: Outputs are written to the specified files
    """
    owner, repo = repo_name.split('/')
    repo = Repo(owner, repo, token=token)
    get_all_activity(repo, pull_output, max_tasks=max_tasks, cutoff_date=cutoff_date)


def get_all_activity(
        repo: Repo,
        pull_output: str,
        max_tasks: Optional[int] = None,
        cutoff_date: Optional[str] = None
):
    """
    Concurrently fetch commits and pull requests from a repository and save to files.

    Args:
        repo: Repository object
        pull_output: Path to save PRs JSON
        max_tasks: Max total activities to fetch
        cutoff_date: Only fetch activity after this date (YYYYMMDD)
    """
    cutoff_date = datetime.strptime(cutoff_date, "%Y%m%d") \
        .strftime("%Y-%m-%dT%H:%M:%SZ") \
        if cutoff_date is not None else None

    seen_pulls = set()

    def process_pulls():
        write_mode = "a" if os.path.exists(pull_output) else "w"
        if write_mode == "a":
            with open(pull_output, encoding="utf-8", mode="r") as fr:
                for line in fr:
                    try:
                        pull_data = json.loads(line)
                        seen_pulls.add(pull_data.get('url'))
                    except json.JSONDecodeError:
                        continue
        with open(pull_output, mode=write_mode, encoding="utf-8") as f_pulls:
            for i_activity, activity in enumerate(repo.get_all_pulls()):
                if activity['url'] in seen_pulls:
                    print(activity['html_url'])
                    continue
                print(activity['html_url'])
                setattr(activity, "src_type", "pull")
                setattr(activity, "resolved_issues", repo.extract_resolved_issues(activity))
                print(json.dumps(obj2dict(activity)), end="\n", flush=True, file=f_pulls)
                print(f"Pull request {len(seen_pulls) + 1} - fetch pull {activity['number']} successfully!")
                seen_pulls.add(activity['url'])

                if max_tasks is not None and i_activity >= max_tasks:
                    break
    process_pulls()
    