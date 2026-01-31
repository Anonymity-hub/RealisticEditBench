import json
import logging
import os
import re
from typing import Optional, Callable, List, Dict, TextIO

from dotenv import load_dotenv
from tqdm import tqdm

from editbench.collection.instance.activity import Activity, filter_by_ft_valid, filter_by_test_valid, \
    write_json_line
from editbench.collection.utils import Repo, extract_problem_statement_and_hints, classify_files, get_patches, \
    get_version_and_commit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


def _restore_processed_state(
        all_output_path: str,
        ft_filters: List[Callable[[Activity], bool]],
        active_filters: List[Callable[[Activity], bool]],
        seen_activity: set[str],
        stats: dict
) -> None:
    """Restore processed instances state (resume from breakpoint)"""
    try:
        with open(all_output_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Restoring processed state", unit="line"):
                try:
                    act_dict = json.loads(line)
                    act = Activity(**act_dict)
                    instance_id = act.instance_id
                    seen_activity.add(instance_id)

                    print(f"Skip {act.instance_id}...")
                    # Count valid instances
                    if all(filter_func(act) for filter_func in ft_filters):
                        if act.src_type == "commit":
                            stats["valid_commits"] += 1
                        elif act.src_type == "pull":
                            stats["valid_pulls"] += 1

                        # Count filtered instances
                        if all(filter_func(act) for filter_func in active_filters):
                            if act.src_type == "commit":
                                stats["filtered_commits"] += 1
                            elif act.src_type == "pull":
                                stats["filtered_pulls"] += 1
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping invalid state record（{line[:50]}...）: {str(e)}")
    except FileNotFoundError:
        logger.info("State file not found, starting from initial state.")


def _process_activities(
        data_path: str,
        activity_type: str,
        start_idx: int,
        repos: Dict[str, "Repo"],
        ft_filters: List[Callable[[Activity], bool]],
        active_filters: List[Callable[[Activity], bool]],
        seen_activity: set[str],
        f_all: TextIO,
        f_output: TextIO,
        stats: dict
) -> dict:
    """
    Generic activity processing function (supports commits/PRs)
    :return: Updated statistics dictionary
    """
    # Precompute number of lines for progress bar
    with open(data_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    banned_id_path = data_path.replace("jsonl", "banned_id.json")
    # set up invalid_id
    banned_id = load_set_from_file(banned_id_path)
    all_data = []

    with open(data_path, "r", encoding="utf-8") as f_data:
        for line in f_data:
            try:
                data = json.loads(line)
                all_data.append(data)
            except json.JSONDecodeError:
                continue

    def sort_key(item):
        return item.get("created_at", "")

    all_data.sort(key=sort_key, reverse=True)

    for i, data in tqdm(
            enumerate(all_data),
            desc=f"Processing {activity_type}s",
            total=total_lines,
            unit="instance"
    ):
        stats["total_instances"] += 1
        try:
            # Resume and deduplication check
            instance_id = Activity.parse_instance_id(data)
            if instance_id in seen_activity:
                print(f"{instance_id} already exists, skipping...")
                if i % 20 == 0:
                    logger.debug(f"Skipping processed {activity_type} instance: {instance_id}")
                continue

            # check if less than start index
            if i < start_idx:
                print(f"{instance_id} index less than start index {start_idx}, skipping...")
                if i % 20 == 0:
                    logger.debug(f"Skipping {activity_type} instance before start index: {instance_id}")
                continue

            # check if in banned list
            if instance_id in banned_id:
                print(f"{instance_id} banned, skipping...")
                if i % 20 == 0:
                    logger.debug(f"Skipping banned {activity_type} instance: {instance_id}")
                continue
            print(f"{instance_id} not found, initializing...")

            act = activity_from_json(data)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping invalid {activity_type} data (line {i}): {str(e)}")
            banned_id.add(instance_id)
            continue

        # Create/get repository instance
        if act.repo not in repos:
            repos[act.repo] = Repo.from_full_name(act.repo)

        # Apply base filters
        if not all(filter_func(act) for filter_func in ft_filters):
            banned_id.add(instance_id)
            save_set_to_file(banned_id, banned_id_path)
            print(f"{instance_id} is banned...")
            continue

        # Record valid instance
        seen_activity.add(act.instance_id)
        write_json_line(act, f_all)
        stats[f"valid_{activity_type}s"] += 1

        # Apply custom filters
        if all(filter_func(act) for filter_func in active_filters):
            write_json_line(act, f_output)
            stats[f"filtered_{activity_type}s"] += 1
        print(f"{instance_id} processed, {i}/{total_lines} entries...")
        # Periodic logging
        if i % 20 == 0:
            logger.info(
                f"[{activity_type}: {act.repo}] Processed {i}/{total_lines} entries, "
                f"Valid {activity_type}s: {stats[f'valid_{activity_type}s']}, "
                f"Filtered {activity_type}s: {stats[f'filtered_{activity_type}s']}"
            )
    return stats


def run_build_datasets(
        path_prs: str,
        output: str,
        ft_filters: Optional[List[Callable[[Activity], bool]]] = None,
        active_filters: Optional[List[Callable[[Activity], bool]]] = None,
        token: Optional[str] = None,
        idx_pull: int = 0
):
    """
    Filter Criteria:
    1. Has base commit *√
    2. only involved modification *√
    3. timestamp window
    4. pr is closed √
    5. 0 √ < file_gt <= 5 √

    :param path_prs: path of crawled pr data
    :param output: final saving path
    :param ft_filters: basic filters func
    :param active_filters: self-designed filters func
    :param token: GitHub API token
    :param idx_pull: start index of prs data
    """
    if ft_filters is None:
        ft_filters = [filter_by_ft_valid()]
    if active_filters is None:
        active_filters = [filter_by_test_valid()]

    token = token or os.getenv("GITHUB_TOKEN_BUILD_DATASET")
    if not token:
        raise ValueError("GitHub token not configured (please set GITHUB_TOKENS environment variable)")

    ft_output = f"{output.replace('.jsonl', '.all.jsonl')}"

    seen_activity: set[str] = set()
    stats = {
        "valid_commits": 0,
        "filtered_commits": 0,
        "valid_pulls": 0,
        "filtered_pulls": 0,
        "total_instances": 0
    }
    repos: Dict[str, Repo] = {}

    # Continue where we left off if output file already exists
    if os.path.exists(ft_output):
        _restore_processed_state(ft_output, ft_filters, active_filters, seen_activity, stats)

    # Write to .all and output file for all activities
    write_mode_all = "w" if not os.path.exists(ft_output) else "a"
    write_mode = "w" if not os.path.exists(output) else "a"

    with (open(ft_output, write_mode_all, encoding="utf-8") as f_all,
          open(output, write_mode, encoding="utf-8") as f_output):

        stats = _process_activities(
            data_path=path_prs,
            activity_type="pull",
            start_idx=idx_pull,
            repos=repos,
            ft_filters=ft_filters,
            active_filters=active_filters,
            seen_activity=seen_activity,
            f_all=f_all,
            f_output=f_output,
            stats=stats
        )

        repo_names = ", ".join(repos.keys()) or "None"
        logger.info(
            f"[{repo_names}] Processing completed: Total instances {stats['total_instances']}, "
            f"Valid instances {stats['valid_commits'] + stats['valid_pulls']} (Commits: {stats['valid_commits']}, PRs: {stats['valid_pulls']}), "
            f"Filtered instances {stats['filtered_commits'] + stats['filtered_pulls']} (Commits: {stats['filtered_commits']}, PRs: {stats['filtered_pulls']})"
        )
        logger.info(f"[{repo_names}] Skipped processed instances: {len(seen_activity)}")


def activity_from_json(data: dict) -> Activity:
    # Get src type
    src_type = "pull" if data["src_type"] == "pr" else data["src_type"]
    # Get repo name
    repo_full_name = re.search(r'https://api\.github\.com/repos/([^/]+/[^/]+)',
                               data['url']).group(1)
    repo = Repo(repo_full_name.split('/')[0], repo_full_name.split('/')[1],
                os.getenv("GITHUB_TOKENS").split(",")[0])
    # Get instance number
    instance_number = data["sha"] if src_type == "commit" else data["number"]
    # Get instance id
    instance_id = f"{repo_full_name.replace('/', '__')}-{src_type}-{instance_number}"

    # Get html_url
    html_url = data["html_url"]

    created_at = ""
    base_commit = ""
    state = "merged"
    resolved_issues = data.get("resolved_issues", [])

    title_mes = ""
    body_mes = data.get("body", "") if data.get("body", "") else ""
    issues_mes, other_mes = extract_problem_statement_and_hints(data, repo)

    related_files = []
    related_commits_info = []
    (files, files_work, files_test, files_no_edit,
     files_config, files_other) = [], [], [], [], [], []

    if src_type == "commit" and data.get("parents", None):
        base_commit = data["parents"][0]["sha"]
        title_mes = data["commit"]["message"]
        created_at = data['commit']['author']['date']

        related_files = data['files']
        related_commits_info = [data]
        (files, files_work, files_test, files_no_edit,
         _, files_config, files_other) = classify_files(related_files)
    elif src_type == "pull":
        if data.get("state", "") == "open":
            state = "open"
        elif (not data.get("merged_at", None) and
              data.get("state", "") == "closed"):
            state = "closed"
        title_mes = data["title"]
        created_at = data["created_at"]

        related_files = list(repo.get_all_loop(repo.api.pulls.list_files, pull_number=instance_number))
        related_commits = list(repo.get_all_loop(repo.api.pulls.list_commits, pull_number=instance_number))
        related_commits_info = []

        # Add commits title and get commit info
        for related_commit in related_commits:
            body_mes += related_commit["commit"]["message"]
            related_commits_info.append(repo.call_api(repo.api.repos.get_commit, owner=repo.owner,
                                                      repo=repo.name, ref=related_commit["sha"]))
        # Get base commit
        base_commit = data["base"]["sha"]

        # Classify files
        (files, files_work, files_test, files_no_edit,
         _, files_config, files_other) = classify_files(related_files)

    # Get version
    version, version_commit = get_version_and_commit(repo, base_commit)
    # Get patches
    work_patch, work_patch_list, test_patch = get_patches(related_files, related_commits_info, files_work, files_test)

    return Activity(instance_id=instance_id, instance_num=instance_number, src_type=src_type, repo=repo_full_name,
                    base_commit=base_commit, html_url=html_url, created_at=created_at, title_mes=title_mes,
                    body_mes=body_mes, issues_mes=issues_mes, other_mes=other_mes, state=state,
                    resolved_issues=resolved_issues, files=files, files_work=files_work,
                    files_test=files_test, files_other=files_other, files_no_edit=files_no_edit,
                    work_patch=work_patch, work_patch_list=work_patch_list, test_patch=test_patch, version=version,
                    version_commit=version_commit)


def activity_from_swe_or_url(repo_full_name, pull_num, swe_dict=None, token=None):
    token = token or os.getenv("GITHUB_TOKEN_BUILD_DATASET")
    repo = Repo.from_full_name(repo_full_name, token)
    activity = repo.api.pulls.get(repo.owner, repo.name, pull_num)
    setattr(activity, "src_type", "pull")
    setattr(activity, "resolved_issues", repo.extract_resolved_issues(activity))
    activity = activity_from_json(activity)

    if swe_dict:
        activity.version = swe_dict["version"]
        activity.version_commit = swe_dict["environment_setup_commit"]
        activity.fail_to_pass = swe_dict["FAIL_TO_PASS"]
        activity.pass_to_pass = swe_dict["PASS_TO_PASS"]

    return activity


def save_set_to_file(data_set: set, file_path: str = "saved_set.json"):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(list(data_set), f, ensure_ascii=False, indent=4)
        print(f"set has been saved to {file_path}")
    except Exception as e:
        print(f"save failed: {str(e)}")


def load_set_from_file(file_path: str = "saved_set.json") -> set:
    if not os.path.exists(file_path):
        print(f"{file_path} not found")
        return set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    except Exception as e:
        print(f"load failed: {str(e)}")
        return set()
