import logging
import os
import re
import subprocess
import sys
import traceback
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, List, Optional
import tempfile
from tqdm import tqdm

from editbench.collection.instance.activity import Activity, load_datasets_from_jsonl
from editbench.editing_split.constants import REPO_AND_LOG_DIR, EDITING_SPLIT_DIR
from editbench.editing_split.diff_utils import generate_diff_with_file
from editbench.evaluation.docker_build import setup_logger, close_logger


def make_valid_script(instance: Activity, patch_lists: list[list[str]]):
    """
    Generate a sequence of valid shell commands to:
    1. Set up a Git repository (clone if missing)
    2. Reset files to a base commit state
    3. Apply multiple patches in sequential steps

    Args:
        instance: Activity object containing repository metadata (e.g., repo name, base commit, target files)
        patch_lists: Nested list of patch strings. Each sublist represents a "step" of patches to apply together

    Returns:
        Tuple containing:
            - List of individual shell commands
            - Single string with all commands joined by newlines
    """
    valid_commands = []

    # Delimiter for here-document syntax in git apply (prevents content injection issues)
    HEREDOC_DELIMITER = "EOF_114329324912"

    repo_path = REPO_AND_LOG_DIR / instance.repo.replace("/", "__") / "testbed"
    repo_path.mkdir(exist_ok=True, parents=True)

    if not os.path.exists(repo_path / ".git"):
        print(f"No repo {instance.repo}, will download: git clone --progress -o origin https://github.com/{instance.repo} {repo_path}")
        valid_commands.append(f"git clone --progress -o origin https://github.com/{instance.repo} {repo_path}")
    valid_commands.append(f"cd {repo_path}")

    work_files = instance.files_work
    # Reset work files to the state they should be in before the patch.
    reset_works_command = f"git checkout {instance.base_commit} {' '.join(work_files)}"

    valid_commands.append(reset_works_command)

    for idx, patch_list in enumerate(patch_lists):
        valid_commands.append(f'echo "Start apply step patch-{idx + 1}..."')
        for patch in patch_list:
            valid_commands.append(f'echo "Start apply patch-{idx + 1} of file: {re.findall(r"--- a/(.*)", patch)}..."')
            apply_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{patch}\n{HEREDOC_DELIMITER}"
            valid_commands.append(apply_patch_command)

    return valid_commands, "\n".join(valid_commands)



def execute_script(script: str, logger: Optional[logging.Logger] = None) -> Dict[str, str]:
    """
    Execute a shell script safely using a temporary file to avoid "Argument list too long" errors.
    """
    temp_script_path = None
    try:
        # 1. write script to a temporary file (Write script to a temporary file)
        # delete=False because we need to close the file before subprocess can read it
        # Windows usually needs .sh suffix to be recognized by Git Bash, or .bat to be recognized by CMD
        suffix = ".sh" if sys.platform != "win32" or "bash" in script.lower() else ".bat"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix, encoding='utf-8') as tmp_file:
            tmp_file.write(script)
            temp_script_path = tmp_file.name

        # 2. grant execution permission (Linux/macOS needs)
        if sys.platform != "win32":
            os.chmod(temp_script_path, 0o755)

        # 3. build execution command (Build execution command)
        # we no longer use shell=True to pass content, but directly call Shell program to run file
        cmd = []

        if sys.platform == "win32":
            git_bash_path = r"C:\Program Files\Git\bin\bash.exe"
            if Path(git_bash_path).exists():
                # use Git Bash to run script file
                cmd = [git_bash_path, temp_script_path]
            else:
                # use default CMD or PowerShell (depending on file suffix association)
                # note: Windows directly running .bat file usually needs shell=True
                cmd = [temp_script_path]
        else:
            # Linux/macOS: use /bin/sh or /bin/bash to execute file
            cmd = ["/bin/sh", temp_script_path]

        # 4. execute process (Execute process)
        # note: here usually does not need shell=True, because we directly specify the executable program (bash/sh) and parameters (file path)
        # Windows without git bash and directly running file, may need shell=True
        use_shell = (sys.platform == "win32" and not cmd[0].endswith("bash.exe"))

        process = subprocess.Popen(
            cmd,
            shell=use_shell,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            bufsize=1,  # Line-buffered
            universal_newlines=True,
            encoding='utf-8',  # explicitly specify encoding to prevent encoding errors
            errors='replace'  # prevent decoding errors from causing errors
        )

        # 5. capture streaming output (Capture streaming output)
        output_lines = []
        if process.stdout:
            for line in process.stdout:
                line = line.rstrip()
                print(f"Script execution message: {line}")
                output_lines.append(line)
                if logger:
                    logger.info(line)

        process.wait()
        return_code = process.returncode

        output = {
            "output": "\n".join(output_lines),
            "error": "",
            "returncode": str(return_code)
        }

        if logger:
            logger.info(f"Script execution completed with return code: {return_code}")

        return output

    except Exception as e:
        traceback.print_exc()
        error_msg = f"Exception occurred during execution: {str(e)}"
        print(error_msg)
        if logger:
            logger.error(error_msg)
        return {"output": "", "error": error_msg, "returncode": "-1"}

    finally:
        # 6. clean up temporary file (Cleanup)
        if temp_script_path and os.path.exists(temp_script_path):
            try:
                os.remove(temp_script_path)
            except Exception:
                pass  # ignore errors during cleanup


def check_apply_output(logger_path: str, instance_id: str):
    """
    Check the output result of patch apply.
    """
    # failed_keywords = ["patch failed", "error:", "version error"]
    failed_keywords = ["patch failed", "version error"]
    output = Path(logger_path).read_text()
    if any(keyword in output.lower() for keyword in failed_keywords):
        print(f"❌❌❌Apply error: {instance_id}, the detailed error see {logger_path}.❌❌❌")
        return False
    return True


def gather_all_res(repos: list[str], instance_ids=None):
    """
    Gather and analyze patch application results by checking validation logs for multiple repository instances.

    Args:
        repos: List of repository names (in "owner/repo" format) to process
        instance_ids: Optional list of instance identifiers to exclude from processing.
                      If provided, only instances NOT in this list will be evaluated.

    Returns:
        Dictionary containing two sets:
            - "success": Set of instance identifiers where patch application succeeded
            - "fail": Set of instance identifiers where patch application failed
    """
    res_log_path = REPO_AND_LOG_DIR
    pass_res = {
        "success": set(),
        "fail": set()
    }
    for repo in repos:
        repo = repo.replace("/", "__")
        repo_path = res_log_path / repo
        for dir_path in repo_path.iterdir():
            if dir_path.is_dir() and dir_path.name != "testbed":
                instance_id = f"{repo}-{dir_path.name}"
                if instance_ids and instance_id not in instance_ids:
                    continue
                else:
                    logger_path = dir_path / "validation.log"
                    res = check_apply_output(logger_path, instance_id)
                    if res:
                        pass_res["success"].add(instance_id)
                    else:
                        pass_res["fail"].add(instance_id)

    print(f"✅ [{datetime.now()}] success applied instance: {pass_res['success']}\n"
          f"❌ [{datetime.now()}] fail applied instance: {pass_res['fail']}")

    return pass_res


def load_patch_list(diff_file_list: list):
    patch_list = []
    for file in diff_file_list:
        diff_text = Path(file).read_text()
        patch_list.append(diff_text)
    return patch_list


def load_patch_list_instance(instance: Activity,
                             target_files: list = None,
                             end_idx: Union[int] = None):
    """
    Load patch history files from the filesystem and organize them into sequential patch lists for an activity instance.

    Args:
        instance: Activity object containing instance metadata (e.g., instance_id, files_work)
        target_files: Optional list of target files to filter - only patches for these files will be loaded.
                      If None, all work files are processed.
        end_idx: Optional index limit - only patches with indices ≤ end_idx will be included.
                 If None, all patches are loaded.

    Returns:
        Nested list of patch strings. Each sublist represents a sequential "step" of patches to apply together.
    """
    instance_id = instance.instance_id
    patch_histories_path = EDITING_SPLIT_DIR / instance_id
    # patch_histories_path.mkdir(exist_ok=True, parents=True)

    patch_lists = []

    for idx, work_file in enumerate(instance.files_work):
        if target_files and work_file not in target_files:
            continue
        work_file_str = work_file.replace("/", "__").replace("\\", "__").replace(".", "__")
        work_file_path = patch_histories_path / work_file_str
        diff_files = find_digit_start_diff_files(work_file_path)
        for diff_file in diff_files:
            current_idx = extract_numeric_prefix(diff_file.name)
            if end_idx and idx > end_idx:
                continue

            target_index = current_idx - 1
            if target_index >= len(patch_lists):
                for _ in range(len(patch_lists), target_index + 1):
                    patch_lists.append([])
            patch_lists[target_index].append(diff_file.read_text())

    return patch_lists


def find_digit_start_diff_files(work_file_path: str) -> List[Path]:
    """
    Find files in target directory that:
    - Start with one or more digits
    - End with 'diff' (case-sensitive)

    :param work_file_path: Directory path to search in
    :return: List of Path objects matching the criteria
    """
    target_dir = Path(work_file_path)
    matched_files = []

    # Validate directory existence and type
    if not target_dir.exists():
        # raise FileNotFoundError(f"Directory not found: {work_file_path}")
        print(f"Directory not found: {work_file_path}")
        return []
    if not target_dir.is_dir():
        # raise NotADirectoryError(f"Path is not a directory: {work_file_path}")
        print(f"Path is not a directory: {work_file_path}")
        return []

    # Regular expression pattern:
    # ^\d+ -> starts with 1+ digits
    # .* -> any characters in between
    # diff$ -> ends with 'diff' (case-sensitive)
    file_pattern = re.compile(r'^\d+.*diff$')

    # Iterate through all items in the directory
    for item in target_dir.iterdir():
        if item.is_file():  # Only check files (ignore subdirectories)
            if file_pattern.match(item.name):
                matched_files.append(item)

    return sorted(matched_files, key=lambda file: extract_numeric_prefix(file.name))


def extract_numeric_prefix(filename: str) -> int:
    """
    Extract leading numeric prefix from a filename

    :param filename: Target filename (e.g., "001_changes.diff")
    :return: Numeric value of the leading digits (e.g., 1 for "001_changes.diff")
    :raises ValueError: If no numeric prefix found (shouldn't occur with filtered files)
    """
    # Regex to match 1+ digits at the start of the filename
    match = re.match(r'^(\d+)', filename)
    if not match:
        raise ValueError(f"No numeric prefix found in '{filename}'")
    return int(match.group(1))


def validation_instance(instance: Activity, patch_list: list[list[str]], is_output_changed_file=False):
    """
    Validate the application of patch lists to a repository instance, including:

    Script generation for cloning, resetting, and applying patches
    Execution of validation scripts with logging
    Optional saving of modified files after patch application

    Args:
    instance: Activity object containing instance metadata (repo, instance_id, files_work, etc.)
    patch_list: Nested list of patch strings. Each sublist represents a step of patches to apply.
    is_output_changed_file: If True, save the final state of modified files after validation.

    Workflow:

        1.Constructs validation directories and paths
        2.Sets up logging for the validation process
        3.Generates and executes shell scripts for patch application
        4.Optionally saves modified files to a standardized location
        5.Checks validation results and cleans up logging resources
    """
    test_path = REPO_AND_LOG_DIR / instance.repo.replace("/", "__") / f"{instance.src_type}-{instance.instance_num}"
    logger_path = test_path / "validation.log"
    script_sh = test_path / "apply.sh"
    logger = setup_logger(instance.instance_id, logger_path)

    logger.info(f"=======Start validation: {instance.instance_id}=========")
    print(f"=======Start validation: {instance.instance_id}=========")

    try:
        valid_script_list, valid_script = make_valid_script(instance, patch_list)
        script_sh.write_text(valid_script)
        execute_script(valid_script, logger)

        for file in instance.files_work:
            read_file_path = REPO_AND_LOG_DIR / instance.repo.replace("/", "__") / "testbed" / file
            save_file_path = (EDITING_SPLIT_DIR / instance.instance_id /
                         file.replace("/", "__").replace(".", "__") / f"final.{file.split('.')[-1]}")
            if is_output_changed_file:
                save_file_path.write_text(read_file_path.read_text())
            if save_file_path.exists():
                diff_content = generate_diff_with_file(read_file_path, save_file_path, file)
                if len(diff_content.split("\n")) > 2:
                    logger.error(f"Version Error: Different final file version\n"
                                 f"-------------\n{diff_content}\n--------------")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        output_path = logger.log_file
        check_apply_output(output_path, instance.instance_id)
        logger.info(f"=======End validation: {instance.instance_id}=========")
        print(f"=======End validation: {instance.instance_id}=========")
        close_logger(logger)


def execute_validation_instance(instance: Activity):
    """
    Execute full validation workflow for an activity instance by:
    1. Loading patch history from filesystem
    2. Applying patches and validating the results

    Args:
        instance: Activity object containing metadata required for validation
                  (e.g., instance ID, repository info, target files)

    Workflow:
        1. Loads patch lists from the instance's patch history directory using `load_patch_list_instance`
        2. Triggers validation by passing the loaded patches to `validation_instance`
    """
    print(f"\n[{datetime.now()}]: Start instance {instance.instance_id}")
    # Load patch history from filesystem based on the activity instance metadata
    patch_list = load_patch_list_instance(instance)
    if not patch_list or all(not inner_list for inner_list in patch_list):
        print(f"[{datetime.now()}]: No patch list, skip")
        return
    # Execute validation process with the loaded patches
    validation_instance(instance, patch_list)
    print(f"[{datetime.now()}]: End instance {instance.instance_id}")


def main(dataset_name: str,
         instance_ids: Optional[list] = None):
    """
    Execution validation
    :param dataset_name: dataset_name | path
    :param instance_ids: target ids
    """
    print("\n>>>>>>>>>>>>>>>>>>>>>> Start executing patch feasibility validation. >>>>>>>>>>>>>>>>>>>>>>")
    dataset = list(load_datasets_from_jsonl(dataset_name))
    repos = set()
    for instance in tqdm(dataset, desc="Validating instances"):
        if instance_ids and instance.instance_id not in instance_ids:
            # print(f"Skipp instance: {instance.instance_id}")
            continue
        if instance.repo not in repos:
            repos.add(instance.repo)
        execute_validation_instance(instance)

    gather_all_res(list(repos), instance_ids)
    print(">>>>>>>>>>>>>>>>>>>>>> End executing patch feasibility validation. >>>>>>>>>>>>>>>>>>>>>>>>")


def validation_inf(instance: Activity):
    valid_commands, scripts = make_valid_script_inf(instance)
    sh_dir = REPO_AND_LOG_DIR / "validation" / instance.repo.replace("/", "__")/instance.instance_id
    sh_dir.mkdir(exist_ok=True, parents=True)
    sh_path = sh_dir / "apply.sh"
    logger_path = sh_dir / "validation.log"

    sh_path.write_text(scripts)
    logger = setup_logger(instance.instance_id, logger_path)

    logger.info(f"=======Start validation: {instance.instance_id}=========")
    print(f"=======Start validation: {instance.instance_id}=========")
    res = execute_script(scripts, logger)
    logger.info(f"=======End validation: {instance.instance_id}=========")
    print(f"=======End validation: {instance.instance_id}=========")
    return res


def make_pre_edits_apply_script(pre_edits: List[str], return_string: bool = True) -> Union[List[str], str]:
    """
    generate script to only apply pre_edits
    
    Args:
        pre_edits: patch string list
        return_string: if True, return string format script; if False, return command list
    """
    # Delimiter for here-document syntax in git apply (prevents content injection issues)
    HEREDOC_DELIMITER = "EOF_114329324912"
    
    commands = []
    
    for idx, patch in enumerate(pre_edits):
        # extract file name for logging
        file_match = re.findall(r"--- a/(.*)", patch)
        file_info = file_match[0] if file_match else "unknown"
        commands.append(f'echo "Start apply step patch-{idx + 1} of file: {file_info}..."')
        apply_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{patch}\n{HEREDOC_DELIMITER}"
        commands.append(apply_patch_command)
    
    if return_string:
        return "\n".join(commands)
    else:
        return commands


def make_valid_script_inf(instance:Activity):
    """
    Generate a sequence of valid shell commands to:
    1. Set up a Git repository (clone if missing)
    2. Reset files to a base commit state
    3. Apply multiple patches in sequential steps

    Args:
        instance: Activity object containing repository metadata (e.g., repo name, base commit, target files)
        patch_lists: Nested list of patch strings. Each sublist represents a "step" of patches to apply together

    Returns:
        Tuple containing:
            - List of individual shell commands
            - Single string with all commands joined by newlines
    """
    valid_commands = []

    # Delimiter for here-document syntax in git apply (prevents content injection issues)
    HEREDOC_DELIMITER = "EOF_114329324912"

    repo_path = REPO_AND_LOG_DIR / instance.repo.replace("/", "__") / "testbed"
    repo_path.mkdir(exist_ok=True, parents=True)

    if not os.path.exists(repo_path / ".git"):
        print(f"No repo {instance.repo}, will download: git clone --progress -o origin https://github.com/{instance.repo} {repo_path}")
        valid_commands.append(f"git clone --progress -o origin https://github.com/{instance.repo} {repo_path}")
    valid_commands.append(f"cd {repo_path}")

    work_files = instance.files_work
    # Reset work files to the state they should be in before the patch.
    reset_works_command = f"git checkout {instance.base_commit} {' '.join(work_files)}"

    valid_commands.append(reset_works_command)

    for idx, patch in enumerate(instance.pre_edits):
        valid_commands.append(f'echo "Start apply step patch-{idx + 1} of file: {re.findall(r"--- a/(.*)", patch)}..."')
        apply_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{patch}\n{HEREDOC_DELIMITER}"
        valid_commands.append(apply_patch_command)
    valid_commands.append(f'echo "Start apply ground patch..."')
    apply_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{instance.ground_truth}\n{HEREDOC_DELIMITER}"
    valid_commands.append(apply_patch_command)

    return valid_commands, "\n".join(valid_commands)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name",
                        default="crawled_data/activity_execution/astropy-astropy-task-instances.jsonl", type=str,
                        help="Name of dataset or path to JSON file.")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    args = parser.parse_args()

    main(**vars(args))





