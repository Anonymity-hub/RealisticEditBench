import argparse
import logging
import os.path
import traceback
from multiprocessing import Pool
from dotenv import load_dotenv

from editbench.collection.build_datasets import run_build_datasets
from editbench.collection.fetch_activity import fetch_repo_activity
from editbench.collection.instance.activity import filter_by_ft_valid, filter_by_test_valid

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_instances(input_list: list, n: int) -> list:
    """
    Split a list into n approximately equal length sublists

    Args:
        input_list (list): List to split
        n (int): Number of sublists to split into
    Returns:
        result (list): List of sublists
    """
    avg_length = len(input_list) // n
    remainder = len(input_list) % n
    result, start = [], 0

    for i in range(n):
        length = avg_length + 1 if i < remainder else avg_length
        sublist = input_list[start: start + length]
        result.append(sublist)
        start += length

    return result


def construct_data_files(data: dict):
    """
    Logic for combining multiple .all PR or commits files into a single dataset

    Args:
        data (dict): Dictionary containing the following keys:
            repos (list): List of repositories to retrieve instruction data for
            path_prs (str): Path to save PR data files to
            path_tasks (str): Path to save task instance data files to
            token (str): GitHub token to use for API requests
    """
    repos, path_prs_, path_tasks_, max_tasks, cutoff_date, token, is_override = (
        data["repos"],
        data["path_prs"],
        data["path_tasks"],
        data["max_tasks"],
        data["cutoff_date"],
        data["token"],
        data["is_override"],
    )
    for repo in repos:
        repo_name = repo.replace("/", "-")
        try:
            path_prs = os.path.join(path_prs_, f"{repo_name}-prs.jsonl")
            path_tasks = os.path.join(path_tasks_, f"{repo_name}-task-instances.jsonl")
            if cutoff_date:
                path_prs = path_prs.replace('.jsonl', f'-{cutoff_date}.jsonl')
                path_tasks = path_tasks.replace('.jsonl', f'-{cutoff_date}.jsonl')
            if not os.path.exists(path_prs) or is_override:
                print(f"Pull request data for {repo} not found, creating...")
                fetch_repo_activity(
                    repo,
                    path_prs,
                    token,
                    max_tasks=max_tasks,
                    cutoff_date=cutoff_date
                )
                print(f"✅ Successfully saved Activity data for {repo} to {path_prs}")
            else:
                print(f"📁 Activity data for {repo} already exists at {path_prs}, skipping...")

            if not os.path.exists(path_tasks) or is_override:
                print(f"Task instance data for {repo} not found, creating...")
                run_build_datasets(path_prs, path_tasks, ft_filters=[filter_by_ft_valid()],
                                   active_filters=[filter_by_test_valid()], token=token)
                print(f"✅ Successfully saved task instance data for {repo} to {path_tasks}")
            else:
                print(f"📁 Task instance data for {repo} already exists at {path_tasks}, skipping...")
        except Exception as e:
            logger.info("-" * 80)
            logger.info(f"Something went wrong for {repo}, skipping: {e}")
            logger.info("Here is the full traceback:")
            traceback.print_exc()
            logger.info("-" * 80)


def main(
        repos: list[str],
        path_prs: str,
        path_tasks: str,
        max_tasks: int = None,
        cutoff_date: str = None,
        is_override: bool = True,
):
    """
    Collect the pull requests to create the task.
    """
    path_prs, path_tasks =  os.path.abspath(path_prs), os.path.abspath(path_tasks)
    print(f"Will save PR data to {path_prs}")
    print(f"Will save activity instance data to {path_tasks}")
    print(f"Received following repos to create task instances for: {repos}")

    tokens = os.getenv("GITHUB_TOKENS")
    if not tokens:
        raise Exception(
            "Missing GITHUB_TOKENS, consider rerunning with GITHUB_TOKENS=$(gh auth token)"
        )
    tokens = tokens.split(",")

    data_task_lists = split_instances(repos, len(tokens))

    data_pooled = [
        {
            "repos": repos,
            "path_prs": path_prs,
            "path_tasks": path_tasks,
            "max_tasks": max_tasks,
            "cutoff_date": cutoff_date,
            "token": token,
            "is_override": is_override,
        }
        for repos, token in zip(data_task_lists, tokens)
    ]

    # construct_data_files(data_pooled[0])
    with Pool(len(tokens)) as p:
        p.map(construct_data_files, data_pooled)


if __name__ == "__main__":
    # Example:
    #   python -m editbench.collection.run_collection \
    #     --repos django/django scikit-learn/scikit-learn \
    #     --path-prs ./crawled_data/raw \
    #     --path-tasks ./crawled_data/activity
    #   python -m editbench.collection.run_collection \
    #     --repos astropy/astropy --path-prs ./crawled_data/raw --path-tasks ./crawled_data/activity \
    #     --max-tasks 100 --cutoff-date 2024-06-01 --no-override
    parser = argparse.ArgumentParser(description="Collect PRs and build task instance datasets.")
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        required=True,
        help="Repositories (e.g. owner/repo), space-separated or pass multiple --repos.",
    )
    parser.add_argument(
        "--path-prs",
        type=str,
        required=True,
        help="Directory to save PR activity files.",
    )
    parser.add_argument(
        "--path-tasks",
        type=str,
        required=True,
        help="Directory to save task instance files.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Max number of tasks per repo (default: no limit).",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Cutoff date suffix for output filenames (e.g. 2024-01-01).",
    )
    parser.add_argument(
        "--no-override",
        action="store_true",
        help="Do not overwrite existing PR/task files (default: override).",
    )
    args = parser.parse_args()
    main(
        repos=args.repos,
        path_prs=args.path_prs,
        path_tasks=args.path_tasks,
        max_tasks=args.max_tasks,
        cutoff_date=args.cutoff_date,
        is_override=not args.no_override,
    )

