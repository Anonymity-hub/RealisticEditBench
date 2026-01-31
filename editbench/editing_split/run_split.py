import os
from argparse import ArgumentParser
from datetime import datetime

from tqdm import tqdm

from editbench.collection.instance.activity import load_datasets_from_jsonl, Activity
from editbench.config import SRC_EXECUTION_FILTER_DATA
from editbench.editing_split.constants import EDITING_SPLIT_DIR, REPO_AND_LOG_DIR
from editbench.utils.github_api_utils import get_github_file_content
from editbench.editing_split.validation import execute_script, validation_instance


def get_repo(repo: str):
    """
    Download a repo
    :param repo: e.g. astropy/astropy
    :return:
    """
    repo_path = REPO_AND_LOG_DIR / repo.replace("/", "__") / "testbed"
    repo_path.mkdir(exist_ok=True, parents=True)

    if not os.path.exists(repo_path / ".git"):
        print(f"Donwload repo: {repo}, git clone --progress -o origin https://github.com/{repo} {repo_path}")
        # execute command
        output = execute_script(f"git clone --progress -o origin https://github.com/{repo} {repo_path}")
        if output["returncode"] != 0:
            print(f"Failed: git clone {repo}")
            return False
    return True


def get_original_file(repo: str, commit: str, file_path: str, instance_id: str):
    """
    Retrieve and save the original content of a specific file from a given Git commit.
    Handles both remote (via GitHub API) and local repository scenarios.

    Args:
        repo: Full repository name in "owner/repo" format (e.g., "pytorch/pytorch")
        commit: SHA hash of the target Git commit (e.g., "5281337d64b59f41d8f8f8b3f9d834426c8e6d03")
        file_path: Relative path to the target file within the repository (e.g., "src/main.py")
        instance_id: Unique identifier for the current editing instance (used for file organization)

    Workflow:
        1. Constructs the target repository path based on the input repo name
        2. If no local Git repository exists:
            - Fetches file content via GitHub API
            - Saves content to a standardized path using instance_id
        3. If local Git repository exists:
            - Uses `git checkout` to restore the file to the target commit state
            - Reads the local file and saves it to the standardized path
    """
    repo_path = REPO_AND_LOG_DIR / repo.replace("/", "__") / "testbed"
    commands = []
    if not os.path.exists(repo_path / ".git"):
        # Split "owner/repo" into owner and repo name components
        owner, repo_name = repo.split("/")
        # Retrieve file content using GitHub API helper function
        read_content = get_github_file_content(owner, repo_name, commit, file_path)
        if read_content:
            # Construct save path with instance-specific organization:
            # Replace "/" and "." in file_path to avoid file system issues, append original extension
            save_file_path = (
                    EDITING_SPLIT_DIR
                    / instance_id
                    / file_path.replace("/", "__").replace(".", "__")
                    / f"original.{file_path.split('.')[-1]}"
            )
            # Ensure parent directories exist and write content to file
            save_file_path.parent.mkdir(parents=True, exist_ok=True)
            save_file_path.write_text(read_content)
        else:
            get_repo(repo)
    else:
        # Add command to navigate to the repository directory
        commands.append(f"cd {repo_path}")
        # Add command to reset the target file to its state in the specified commit
        commands.append(f"git checkout {commit} {file_path}")

        # Execute the constructed shell commands
        execute_script("\n".join(commands))

        # Construct path to the locally restored file
        read_file_path = REPO_AND_LOG_DIR / repo.replace("/", "__") / "testbed" / file_path
        # Construct save path (same structure as remote case)
        save_file_path = (
                EDITING_SPLIT_DIR
                / instance_id
                / file_path.replace("/", "__").replace(".", "__")
                / f"original.{file_path.split('.')[-1]}"
        )
        # Ensure parent directories exist, read local file content, and save
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        save_file_path.write_text(read_file_path.read_text())


def split_instance(instance: Activity):
    """
    Splits an original Activity instance's patch history into individual files and subdirectories.

    This function processes the patch data from an Activity instance, creating a structured
    directory hierarchy to store:
    - A complete combined diff file ("whole.diff")
    - Individual sub-diff files organized by modified work files

    Args:
        instance (Activity): The Activity instance containing patch data to be split.
            Should have the following attributes:
            - instance_id (str): Unique identifier for the activity instance
            - work_patch (str): Full combined diff content
            - files_work (list[str]): List of file paths that were modified
            - work_patch_list (list[list[str]]): Nested list containing individual
              diff segments, where:
              - First level: corresponds to files in files_work
              - Second level: contains diff segments for that specific file

    Returns:
        None
    """
    print(f"\n[{datetime.now()}]: Start instance {instance.instance_id}")
    instance_id = instance.instance_id
    patch_histories_path = EDITING_SPLIT_DIR / instance_id
    patch_histories_path.mkdir(exist_ok=True, parents=True)

    whole_diff_path = patch_histories_path / "whole.diff"
    # Write to work_file_path / whole.diff
    whole_diff_path.write_text(instance.work_patch)

    for idx, work_file in enumerate(instance.files_work):
        work_file_str = work_file.replace("/", "__").replace("\\", "__").replace(".", "__")
        work_file_path = patch_histories_path / work_file_str
        work_file_path.mkdir(exist_ok=True, parents=True)

        # Write to sub diff
        diff_list = instance.work_patch_list[idx]
        for i_sub, diff in enumerate(diff_list):
            sub_diff_path = work_file_path / f"whole-{i_sub + 1}.diff"
            sub_diff_path.write_text(diff)
        # Get Original file
        get_original_file(instance.repo, instance.base_commit, work_file, instance_id)

    validation_instance(instance, instance.work_patch_list, is_output_changed_file=True)
    print(f"[{datetime.now()}]: End instance {instance.instance_id}")


def main(dataset_name: str,
         instance_ids: list = None,
         time_window: str = "20241201"):
    """
    Execution Initial instance splitting.
    :param dataset_name: dataset_name | path
    :param instance_ids: target ids
    """
    print("\n>>>>>>>>>>>>>>>>>>>>>> Start: init instances. >>>>>>>>>>>>>>>>>>>>>>")
    dataset = list(load_datasets_from_jsonl(dataset_name))

    try:
        start_date = datetime.strptime(time_window, "%Y%m%d").date()
    except ValueError:
        raise ValueError(f"time_window format error! please input YYYYMMDD format (e.g. 20241201), current value: {time_window}")
    dataset = [ins for ins in dataset if datetime.fromisoformat(ins.created_at.rstrip('Z')).date() >= start_date]

    for instance in tqdm(dataset, desc="Splitting instances"):
        # if instance.instance_id != "scikit-learn__scikit-learn-pull-31556":
        #     continue
        # if instance.instance_id != "matplotlib__matplotlib-pull-29879":
        #     continue
        if instance_ids and instance.instance_id not in instance_ids:
            # print(f"Skipp instance: {instance.instance_id}")
            continue

        split_instance(instance)
    print(">>>>>>>>>>>>>>>>>>>>>> End: init instances. >>>>>>>>>>>>>>>>>>>>>>>>>")


if __name__ == "__main__":
    """
    extract merged commits
    """
    # dataset_names = ["matplotlib/matplotlib"]
    # dataset_names = ["scikit-learn/scikit-learn"]
    # dataset_names = ["sphinx-doc/sphinx"]
    # dataset_names = ["pydata/xarray"]
    dataset_names = ["pylint-dev/pylint"]
    for name in dataset_names:
        path = f"{SRC_EXECUTION_FILTER_DATA}/{name.replace('/', '-')}-task-instances.jsonl"
        main(path)

