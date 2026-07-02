import csv
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Union, Optional, Set, Tuple

import pandas as pd
from editbench.collection.instance.activity import Activity, load_datasets_from_jsonl
from editbench.collection.utils import Repo, rebuild_work_patch_list, resolve_version_for_commit
from editbench.config.constants import SRC_ACTIVITY_DATA, SRC_INF_BENCHMARK_DATA, TAG_VERSION
from editbench.evaluation.constants import MAP_REPO_VERSION_TO_SPECS


def get_inf_datasets(
        dataset_name: str,
        instance_ids: list[str] = None,
        sort_by_time: bool = True,
        repos: Optional[List[str]] = None,
        sampled_ids_file: Optional[str] = None
) -> List[Activity]:
    """
    load dataset, support group by repository and sort by time
    
    :param dataset_name: dataset file path
    :param instance_ids: list of instance IDs to load
    :param sort_by_time: whether to sort by time (from recent to distant)
    :param repos: list of repositories to load (format: ["owner/repo"]), if None, load all repositories
    :param sampled_ids_file: sampled JSON file path, if provided, read sampled_instance_ids from file and use for filtering
    :return: list of Activity, grouped by repository, sorted by time from recent to distant
    """
    # if sampled file is provided, read instance_ids from file
    if sampled_ids_file:
        sampled_file_path = Path(sampled_ids_file)
        if sampled_file_path.exists():
            try:
                with open(sampled_file_path, 'r', encoding='utf-8') as f:
                    sampled_data = json.load(f)
                    sampled_instance_ids = sampled_data.get('sampled_instance_ids', [])
                    if sampled_instance_ids:
                        # if instance_ids is provided, take intersection; otherwise use IDs from sampled file
                        if instance_ids:
                            instance_ids = list(set(instance_ids) & set(sampled_instance_ids))
                            print(f"📋 use sampled file: {sampled_file_path}")
                            print(f"    number of instances in sampled file: {len(sampled_instance_ids)}")
                            print(f"    intersection with provided instance_ids: {len(instance_ids)} instances")
                        else:
                            instance_ids = sampled_instance_ids
                            print(f"📋 use sampled file: {sampled_file_path}")
                            print(f"    number of sampled instances: {len(sampled_instance_ids)}")
                    else:
                        print(f"⚠️  sampled file does not have sampled_instance_ids field: {sampled_file_path}")
            except Exception as e:
                print(f"⚠️  failed to read sampled file: {e}")
        else:
            print(f"ℹ️  sampled file not found: {sampled_file_path}")

    print(f"Load dataset from path: {dataset_name}")
    try:
        dataset = list(load_datasets_from_jsonl(dataset_name))
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e

    if not dataset:
        from editbench.utils.lfs_utils import is_git_lfs_pointer, lfs_pull_help

        if is_git_lfs_pointer(dataset_name):
            raise ValueError(lfs_pull_help(dataset_name))
        raise ValueError(
            f"No instances loaded from dataset: {dataset_name}\n"
            "Check that the JSONL path is correct and the file is not empty."
        )


    if instance_ids:
        dataset_ids = {i.instance_id for i in dataset}
        instance_ids_set = set(instance_ids)
        missing_ids = instance_ids_set - dataset_ids
        if missing_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(missing_ids)}"
                )
            )
        dataset = [instance for instance in dataset if instance.instance_id in instance_ids_set]

    # if repositories are specified, filter first
    if repos is not None:
        dataset = [instance for instance in dataset if instance.repo in repos]

    # group by repository
    repo_activities: Dict[str, List[Activity]] = defaultdict(list)
    for activity in dataset:
        repo = activity.repo
        if not repo:
            # if repo field is not present, try to extract from instance_id
            # instance_id format: owner__repo-src_type-number
            # ensure repo part is not "pull"
            match = re.match(r'^([^_]+)__([^_]+)-pull', activity.instance_id)
            if match and match.group(2) != 'pull':
                repo = f"{match.group(1)}/{match.group(2)}"
            else:
                repo = "unknown"
        repo_activities[repo].append(activity)

    # sort by time for each repository (from recent to distant)
    if sort_by_time:
        for repo in repo_activities:
            repo_activities[repo].sort(key=lambda x: x.created_at if x.created_at else "", reverse=True)

    # sort by repository name, then merge results
    sorted_repos = sorted(repo_activities.keys())
    result = []
    for repo in sorted_repos:
        result.extend(repo_activities[repo])

    return result


DEFAULT_CSV_EXCLUDE_FIELDS = frozenset({
    "prompt",
    "num_changes",
    "split_idx",
    "pre_edits",
    "ground_truth",
})

_CSV_LIST_FIELDS = frozenset({
    "files", "files_work", "files_test",
    "files_other", "files_no_edit", "resolved_issues", "other_mes",
    "fail_to_pass", "pass_to_fail", "fail_to_fail", "pass_to_pass",
})


def jsonl_to_csv(
    jsonl_path: str,
    csv_path: str,
    exclude_fields: Optional[Iterable[str]] = None,
) -> None:
    """
    Convert an Activity JSONL file to a CSV file (UTF-8 encoded).

    :param jsonl_path: Path to the input JSONL file
    :param csv_path: Path to save the output CSV file
    :param exclude_fields: Field names to omit from CSV columns; defaults to
        prompt, num_changes, split_idx, pre_edits, ground_truth
    """
    excluded = (
        set(exclude_fields)
        if exclude_fields is not None
        else set(DEFAULT_CSV_EXCLUDE_FIELDS)
    )
    all_field_names = [f.name for f in fields(Activity)]
    field_names = [name for name in all_field_names if name not in excluded]

    # Read and parse JSONL data
    instances: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i_line, line in enumerate(f):
            try:
                instance_dict = json.loads(line.strip())
                # Ensure all fields exist (fill missing fields with default values)
                for field in all_field_names:
                    if field not in instance_dict:
                        instance_dict[field] = getattr(Activity, field).default
                instances.append(instance_dict)
            except json.JSONDecodeError as e:
                print(f"Skipping line that failed to parse: {line}\nError: {e}")
                continue

    # Write data to CSV file (UTF-8 encoded)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()

        for i_ins, instance in enumerate(instances):
            for list_field in _CSV_LIST_FIELDS:
                if list_field in field_names:
                    instance[list_field] = json.dumps(instance[list_field])

            writer.writerow({name: instance[name] for name in field_names})


def json_to_csv_pd(jsonl_path: str, csv_path: str) -> None:
    df = pd.read_json(jsonl_path, lines=True)
    df.to_csv(csv_path, index=False)


def modified_jsonl_field(file_path, target_instance_id, field_name, new_value):
    # read all lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    modified = False
    # iterate through each line to find matching instance_id
    for i in range(len(lines)):
        try:
            # parse JSON object
            data = json.loads(lines[i])

            # check if instance_id field exists and matches the target
            if 'instance_id' in data and data['instance_id'] == target_instance_id:
                # modify specified field
                data[field_name] = new_value

                # convert modified data back to JSON string
                lines[i] = json.dumps(data, ensure_ascii=False) + '\n'
                modified = True
                break  # assume instance_id is unique, exit loop after finding

        except json.JSONDecodeError:
            print(f"warning: line {i + 1} is not a valid JSON format, skipped")
        except Exception as e:
            print(f"error: processing line {i + 1}: {str(e)}, skipped")

    # if found and modified the record, write back to file
    if modified:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print(f"successfully modified instance_id '{target_instance_id}' field '{field_name}'")
    else:
        print(f"not found instance_id '{target_instance_id}' record")

    return modified


def print_and_get_file_type(file_path: Union[Path, str], time_end=None):
    """
    1. instance.created_time field in datasets is str time '2025-09-19T21:15:54Z'
    2. instance,files_work is list array
    return number of single file and multi file >= time_end
    :param file_path:
    :param time_end:
    :return:
    """
    datasets = get_inf_datasets(file_path, sort_by_time=True)

    single_file_count = 0  # single file counter
    multi_file_count = 0  # multi file counter

    te_datetime = None
    if time_end is not None:
        try:
            # process UTC time with Z (replace Z with +00:00 to compatible with fromisoformat)
            te_str = time_end.replace('Z', '+00:00')
            te_datetime = datetime.fromisoformat(te_str)
        except ValueError:
            raise ValueError(f"time_end format error, should be 'YYYY-MM-DDTHH:MM:SSZ', current value: {time_end}")

    # iterate through all instances
    for instance in datasets:
        # get instance creation time and convert to UTC time object
        created_time_str = instance.created_at
        if not created_time_str:
            continue  # no creation time, skip

        try:
            ct_str = created_time_str.replace('Z', '+00:00')
            ct_datetime = datetime.fromisoformat(ct_str)
        except ValueError:
            continue  # creation time format error, skip

        # check if satisfies time condition (>= time_end)
        if te_datetime is not None and ct_datetime < te_datetime:
            continue  # does not satisfy time condition, skip

        # count file type
        files_work = instance.files_work
        fw_length = len(files_work)

        if fw_length == 1:
            single_file_count += 1
        elif fw_length > 1:
            multi_file_count += 1
    print(single_file_count, multi_file_count)
    return (single_file_count, multi_file_count)


def get_instance_ids_from_jsonl(file_path: Union[Path, str]) -> Set[str]:
    """
    Collect all instance_id values from a JSONL file.

    :param file_path: JSONL file path
    :return: set of instance_ids
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"jsonl not found: {file_path}")

    instance_ids: Set[str] = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                iid = data.get("instance_id")
                if iid:
                    instance_ids.add(iid)
            except json.JSONDecodeError as e:
                print(f"warning: {file_path} line {line_num} invalid JSON, skipped: {e}")
    return instance_ids


def remove_extra_instance_ids_from_jsonl(
    first_jsonl_path: Union[Path, str],
    second_jsonl_path: Union[Path, str],
    backup: bool = True,
) -> Set[str]:
    """
    Remove records from the second JSONL file whose instance_id does not appear in the first.

    This computes: extra_ids = instance_ids(second) - instance_ids(first)
    then deletes those extra ids from the second file and prints them.

    :param first_jsonl_path: Reference JSONL file path
    :param second_jsonl_path: JSONL file path to be pruned in place
    :param backup: Whether to create a .backup file before modifying the second file
    :return: The set of extra instance_ids removed from the second file
    """
    first_ids = get_instance_ids_from_jsonl(first_jsonl_path)
    second_jsonl_path = Path(second_jsonl_path)
    if not second_jsonl_path.exists():
        raise FileNotFoundError(f"jsonl not found: {second_jsonl_path}")

    second_ids = get_instance_ids_from_jsonl(second_jsonl_path)
    extra_ids = second_ids - first_ids

    print(f"first jsonl instance_ids: {len(first_ids)}")
    print(f"second jsonl instance_ids: {len(second_ids)}")
    print(f"extra instance_ids to remove: {len(extra_ids)}")
    if extra_ids:
        print("extra ids:")
        for iid in sorted(extra_ids):
            print(iid)

    if not extra_ids:
        print("no extra ids found; second jsonl unchanged")
        return extra_ids

    if backup:
        backup_path = second_jsonl_path.with_suffix(second_jsonl_path.suffix + ".backup")
        shutil.copy2(second_jsonl_path, backup_path)
        print(f"backup: {backup_path}")

    kept_lines = []
    removed_count = 0
    extra_ids_set = set(extra_ids)
    with open(second_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line
            line = line.strip()
            if not line:
                kept_lines.append(raw)
                continue
            try:
                data = json.loads(line)
                iid = data.get("instance_id")
                if iid in extra_ids_set:
                    removed_count += 1
                else:
                    kept_lines.append(raw)
            except json.JSONDecodeError as e:
                print(f"warning: {second_jsonl_path} line {line_num} invalid JSON, kept: {e}")
                kept_lines.append(raw)

    with open(second_jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)

    print(f"removed {removed_count} records from {second_jsonl_path}")
    return extra_ids


def remove_instance_ids_from_jsonl(
    dataset_paths: List[Union[Path, str]],
    instance_ids: Optional[Union[Set[str], List[str]]] = None,
    instance_ids_file: Optional[Union[Path, str]] = None,
    backup: bool = True,
) -> Dict[str, int]:
    """
    Remove lines whose instance_id is in the given set from each JSONL file.

    :param dataset_paths: List of JSONL file paths to process
    :param instance_ids: Set or list of instance_ids to remove
    :param instance_ids_file: Optional JSON file path; if given, read ids from key
        "union_filter_instance_ids" or "instance_ids" (used when instance_ids is None)
    :param backup: Whether to create a .backup file before modifying (default True)
    :return: Dict mapping each file path (str) to the number of removed records
    """
    if instance_ids_file is not None:
        path = Path(instance_ids_file)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ids = data.get("union_filter_instance_ids") or data.get("instance_ids") or []
            instance_ids = list(ids)
        else:
            raise FileNotFoundError(f"instance_ids_file not found: {path}")

    if instance_ids is None:
        raise ValueError("Either instance_ids or instance_ids_file must be provided")
    ids_set = set(instance_ids)

    result: Dict[str, int] = {}
    for file_path in dataset_paths:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"skip (not found): {file_path}")
            continue
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + ".backup")
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"backup: {backup_path}")

        kept_lines = []
        removed_count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                raw = line
                line = line.strip()
                if not line:
                    kept_lines.append(raw)
                    continue
                try:
                    data = json.loads(line)
                    iid = data.get("instance_id")
                    if iid in ids_set:
                        removed_count += 1
                    else:
                        kept_lines.append(raw)
                except json.JSONDecodeError as e:
                    print(f"warning: {file_path} line {line_num} invalid JSON, kept: {e}")
                    kept_lines.append(raw)

        if removed_count > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
            print(f"removed {removed_count} records from {file_path}")
        result[str(file_path)] = removed_count
    return result


def remove_failed_retries_from_jsonl(file_path: Union[Path, str], backup: bool = True) -> int:
    """
    remove records with status "failed" and error contains "Max retries" from JSONL file
    
    :param file_path: JSONL file path
    :param backup: whether to create backup file (default True)
    :return: number of removed records
    """
    file_path = Path(file_path)
    
    # create backup file
    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"created backup file: {backup_path}")
    
    # read all lines
    kept_lines = []
    removed_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            original_line = line  # keep original line (including newline)
            line = line.strip()
            if not line:  # empty line keep
                kept_lines.append(original_line)
                continue
            
            try:
                # parse JSON object
                data = json.loads(line)
                
                # check if need to delete: status is "failed" and error contains "Max retries"
                status = data.get('status', '')
                error = data.get('error', '')
                
                should_remove = (
                    status == 'failed' and 
                    error and 
                    'Max retries' in error
                )
                
                if should_remove:
                    removed_count += 1
                else:
                    # keep this line (use original line to keep format)
                    kept_lines.append(original_line)
                    
            except json.JSONDecodeError as e:
                print(f"warning: line {line_num} is not a valid JSON format, skipped: {e}")
                # keep lines with format error, avoid data loss (use original line)
                kept_lines.append(original_line)
            except Exception as e:
                print(f"error: processing line {line_num}: {str(e)}, skipped")
                # keep error records, avoid data loss (use original line)
                kept_lines.append(original_line)
    
    # write back to file
    if removed_count > 0:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(kept_lines)
        print(f"successfully removed {removed_count} records")
    else:
        print("no records to delete")
    
    return removed_count


def modified_jsonl_patch_list(file_path):
    # read all lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    modified = False
    # iterate through each line to find matching instance_id
    for i in range(len(lines)):
        try:
            # parse JSON object
            data = json.loads(lines[i])
            patch_list = data["work_patch_list"]
            file_list = data["files_work"]
            whole_patch = data["work_patch"]

            diff_list = re.split(r'(?=^diff --git )', whole_patch, flags=re.MULTILINE)
            diff_list = [d for d in diff_list if d.strip().startswith("diff --git")]

            # num_patch_list
            commit_gt_2 = any(len(row) > 1 for row in patch_list)

            # check if instance_id field exists and matches the target
            new_patch_list = [[] for _ in file_list]
            if commit_gt_2:
                for j, (file_name, diff_) in enumerate(zip(file_list, diff_list)):
                    if file_name not in diff_:
                        raise Exception("order error!!!")
                    new_patch_list[j].append(diff_)
                # modify specified field
                data["work_patch_list"] = new_patch_list

                # convert modified data back to JSON string
                lines[i] = json.dumps(data, ensure_ascii=False) + '\n'
                modified = True

        except json.JSONDecodeError:
            print(f"warning: line {i + 1} is not a valid JSON format, skipped")
        except Exception as e:
            print(f"error: processing line {i + 1}: {str(e)}, skipped")

    # if found and modified the record, write back to file
    if modified:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        print(f"successfully modified")

    return modified


def _work_patch_list_changed(old: list, new: list) -> bool:
    if len(old) != len(new):
        return True
    for old_patches, new_patches in zip(old, new):
        if old_patches != new_patches:
            return True
    return False


def refresh_work_patch_lists_in_jsonl(
    jsonl_path: Union[Path, str],
    instance_ids: Optional[List[str]] = None,
    min_date: Optional[str] = "2025-10-01",
    backup: bool = True,
    token: Optional[str] = None,
) -> Dict[str, tuple[list, list]]:
    """
    Rebuild work_patch_list for activities in a jsonl file (skipping merge commits).

    :param jsonl_path: Path to the jsonl file to update in place
    :param instance_ids: Optional subset of instance_ids to refresh; refresh all if None
    :param min_date: Only refresh instances with created_at on/after this date (YYYY-MM-DD).
        Rows before this date are kept unchanged. Pass None to refresh all rows.
    :param backup: Whether to write a .backup copy before modifying
    :param token: GitHub token; defaults to GITHUB_TOKENS / GITHUB_TOKEN env
    :return: Dict mapping instance_id -> (old_lens, new_lens) for updated rows
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path}")

    token = token or (os.getenv("GITHUB_TOKENS") or "").split(",")[0] or os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Missing GitHub token: set GITHUB_TOKENS or GITHUB_TOKEN")

    min_date_obj = None
    if min_date:
        min_date_obj = datetime.strptime(min_date, "%Y-%m-%d").date()

    target_ids = set(instance_ids) if instance_ids else None
    repo_cache: Dict[str, Repo] = {}
    updated: Dict[str, tuple[list, list]] = {}
    kept_lines: List[str] = []
    scanned = 0
    skipped_date = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line
            line = line.strip()
            if not line:
                kept_lines.append(raw)
                continue
            data = json.loads(line)
            instance_id = data.get("instance_id")
            if target_ids is not None and instance_id not in target_ids:
                kept_lines.append(raw)
                continue

            activity = Activity(**data)
            scanned += 1
            created = _activity_created_date(activity)
            if min_date_obj is not None and (created is None or created < min_date_obj):
                skipped_date += 1
                kept_lines.append(raw)
                continue

            old_work_patch_list = activity.work_patch_list
            old_lens = [len(x) for x in old_work_patch_list]
            if activity.repo not in repo_cache:
                repo_cache[activity.repo] = Repo.from_full_name(activity.repo, token)
            new_work_patch_list = rebuild_work_patch_list(activity, repo_cache[activity.repo])
            new_lens = [len(x) for x in new_work_patch_list]
            if _work_patch_list_changed(old_work_patch_list, new_work_patch_list):
                activity.work_patch_list = new_work_patch_list
                updated[instance_id] = (old_lens, new_lens)
                if new_lens != old_lens:
                    print(f"updated {instance_id}: {old_lens} -> {new_lens}")
                else:
                    print(f"updated {instance_id}: content changed (lens unchanged {old_lens})")
                kept_lines.append(json.dumps(asdict(activity), ensure_ascii=False) + "\n")
            else:
                kept_lines.append(raw)

    print(
        f"scanned {scanned} row(s), skipped {skipped_date} before {min_date}, "
        f"changed {len(updated)} row(s)"
    )

    if not updated:
        print("No work_patch_list changes detected.")
        return updated

    if backup:
        backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".backup")
        shutil.copy2(jsonl_path, backup_path)
        print(f"backup: {backup_path}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)
    print(f"Refreshed {len(updated)} instance(s) in {jsonl_path}")
    return updated


def _activity_created_date(activity: Activity):
    if not activity.created_at:
        return None
    return datetime.fromisoformat(activity.created_at.rstrip("Z")).date()


def refresh_versions_in_jsonl(
    jsonl_path: Union[Path, str],
    instance_ids: Optional[List[str]] = None,
    min_date: Optional[str] = "2025-10-01",
    backup: bool = True,
    token: Optional[str] = None,
    refresh_tag_cache: bool = False,
    force: bool = False,
    set_version: Optional[str] = None,
    set_version_commit: Optional[str] = None,
) -> Dict[str, tuple[tuple[str, str], tuple[str, str]]]:
    """
    Recompute or set version fields for matching activities in a jsonl file.

    Two modes (mutually exclusive):
    - **Tag mode** (default): resolve from base_commit + tag_version via GitHub API.
    - **Manual mode**: pass ``set_version`` (and optionally ``set_version_commit``);
      no GitHub / release lookup; only rows matching ``min_date`` / ``instance_ids`` are updated.

    :param jsonl_path: Path to the jsonl file to update in place
    :param instance_ids: Optional subset of instance_ids; refresh all matching rows if None
    :param min_date: Only refresh instances with created_at on/after this date (YYYY-MM-DD).
        Set to None to refresh every row in the file.
    :param backup: Whether to write a .backup copy before modifying
    :param token: GitHub token (tag mode only); defaults to GITHUB_TOKENS / GITHUB_TOKEN env
    :param refresh_tag_cache: If True, refetch tag_version jsonl per repo before resolving
    :param force: If True, rewrite rows even when version fields are unchanged
    :param set_version: If set, assign this version string directly (manual mode)
    :param set_version_commit: If set with manual mode, also assign version_commit; if omitted,
        version_commit is left unchanged
    :return: Dict mapping instance_id -> ((old_version, old_commit), (new_version, new_commit))
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path}")

    manual_mode = set_version is not None
    if manual_mode:
        if refresh_tag_cache:
            raise ValueError("--refresh-tag-cache cannot be used with --version (manual mode)")
    else:
        token = token or (os.getenv("GITHUB_TOKENS") or "").split(",")[0] or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("Missing GitHub token: set GITHUB_TOKENS or GITHUB_TOKEN")

    min_date_obj = None
    if min_date:
        min_date_obj = datetime.strptime(min_date, "%Y-%m-%d").date()

    target_ids = set(instance_ids) if instance_ids else None
    repo_cache: Dict[str, Repo] = {}
    tag_cache_refreshed: Set[str] = set()
    updated: Dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
    kept_lines: List[str] = []
    scanned = 0
    skipped_date = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line
            line = line.strip()
            if not line:
                kept_lines.append(raw)
                continue

            data = json.loads(line)
            instance_id = data.get("instance_id")
            if target_ids is not None and instance_id not in target_ids:
                kept_lines.append(raw)
                continue

            activity = Activity(**data)
            scanned += 1

            created = _activity_created_date(activity)
            if min_date_obj is not None and (created is None or created < min_date_obj):
                skipped_date += 1
                kept_lines.append(raw)
                continue

            old_pair = (activity.version or "", activity.version_commit or "")

            if manual_mode:
                version = set_version
                version_commit = (
                    set_version_commit
                    if set_version_commit is not None
                    else activity.version_commit
                )
            else:
                if not activity.base_commit:
                    print(f"skip {instance_id}: missing base_commit")
                    kept_lines.append(raw)
                    continue

                repo_key = activity.repo
                if repo_key not in repo_cache:
                    repo_cache[repo_key] = Repo.from_full_name(repo_key, token)

                if refresh_tag_cache and repo_key not in tag_cache_refreshed:
                    owner, name = repo_key.split("/", 1)
                    tag_path = Path(TAG_VERSION) / f"{owner}-{name}-version.jsonl"
                    if tag_path.exists():
                        tag_path.unlink()
                        print(f"refreshed tag cache: {tag_path}")
                    tag_cache_refreshed.add(repo_key)

                version, version_commit = resolve_version_for_commit(
                    repo_cache[repo_key], activity.base_commit
                )

            new_pair = (version or "", (version_commit or "") if version_commit is not None else "")

            if not force and new_pair == old_pair:
                kept_lines.append(raw)
                continue

            activity.version = version
            activity.version_commit = version_commit or ""
            updated[instance_id] = (old_pair, new_pair)
            print(f"updated {instance_id}: version {old_pair[0]!r} -> {new_pair[0]!r}")
            kept_lines.append(json.dumps(asdict(activity), ensure_ascii=False) + "\n")

    print(
        f"scanned {scanned} row(s), skipped {skipped_date} before {min_date}, "
        f"changed {len(updated)} row(s)"
    )

    if not updated:
        print("No version changes detected.")
        return updated

    if backup:
        backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".backup")
        shutil.copy2(jsonl_path, backup_path)
        print(f"backup: {backup_path}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)
    print(f"Refreshed versions for {len(updated)} instance(s) in {jsonl_path}")
    return updated


def refresh_versions_in_activity_dir(
    activity_dir: Union[Path, str] = SRC_ACTIVITY_DATA,
    pattern: str = "*-task-instances.jsonl",
    **kwargs,
) -> Dict[str, Dict]:
    """
    Run refresh_versions_in_jsonl on each matching jsonl under activity_dir.
    """
    activity_dir = Path(activity_dir)
    results = {}
    for jsonl_path in sorted(activity_dir.glob(pattern)):
        if jsonl_path.name.endswith(".backup"):
            continue
        print(f"\n>>> {jsonl_path}")
        results[str(jsonl_path)] = refresh_versions_in_jsonl(jsonl_path, **kwargs)
    return results


def normalize_dataset_name(dataset_name: str, run_id: str) -> Tuple[str, str]:
    """
    Normalize dataset_name to full path and extract name.

    Args:
        dataset_name: Can be "all", "owner/repo_name", or full path
        run_id: Run ID for constructing path

    Returns:
        (normalized_path, name): Full path and extracted name
    """
    if dataset_name.endswith(".jsonl"):
        path = Path(dataset_name)
        stem = path.stem
        name_match = re.match(r'^(.+?)-task-instances_.+$', stem)
        if name_match:
            name = name_match.group(1)
        else:
            name = stem.split('-task-instances_')[0] if '-task-instances_' in stem else "all"
        return str(path), name

    if dataset_name == "all" or dataset_name in MAP_REPO_VERSION_TO_SPECS.keys():
        name = dataset_name.replace("/", "-")
        normalized_path = f"{SRC_INF_BENCHMARK_DATA}/{name}-task-instances_{run_id}.jsonl"
        return normalized_path, name

    name = dataset_name.replace("/", "-")
    normalized_path = f"{SRC_INF_BENCHMARK_DATA}/{name}-task-instances_{run_id}.jsonl"
    return normalized_path, name


if __name__ == "__main__":
    import argparse

    # Examples:
    # python -m editbench.utils.dataset_utils refresh-work-patch-list \
    #   --jsonl-path ./crawled_data/activity_execution/astropy-astropy-task-instances.jsonl \
    #   --instance-ids astropy__astropy-pull-19267
    parser = argparse.ArgumentParser(description="Dataset jsonl utilities")
    parser.add_argument(
        "command",
        # choices=(
        #     "jsonl-to-csv",
        #     "refresh-work-patch-list",
        #     "refresh-versions",
        #     "remove-extra-instance-ids",
        # ),
        nargs="?",
        default="jsonl-to-csv",
        help="Utility to run (default: jsonl-to-csv)",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="./crawled_data/activity_execution/astropy-astropy-task-instances.jsonl",
        help="Input jsonl path",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Output csv path (jsonl-to-csv only; default: same name with .csv)",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional instance_ids to refresh (refresh-work-patch-list only)",
    )
    parser.add_argument(
        "--remove-instance-ids",
        type=str,
        nargs="*",
        default=None,
        help="Instance IDs to remove from the jsonl file (remove-instance-ids only)",
    )
    parser.add_argument(
        "--instance-ids-file",
        type=str,
        default=None,
        help="JSON file containing instance_ids or union_filter_instance_ids (remove-instance-ids only)",
    )
    parser.add_argument(
        "--first-jsonl-path",
        type=str,
        default=None,
        help="Reference jsonl path (remove-extra-instance-ids only)",
    )
    parser.add_argument(
        "--second-jsonl-path",
        type=str,
        default=None,
        help="Target jsonl path to prune in place (remove-extra-instance-ids only)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .backup before refreshing jsonl",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default="2025-10-01",
        help="Only refresh instances created on/after this date (YYYY-MM-DD). "
        "Use 'none' to refresh all rows.",
    )
    parser.add_argument(
        "--activity-dir",
        type=str,
        default=None,
        help="If set, refresh all *-task-instances.jsonl under this directory "
        "(refresh-versions only; ignores --jsonl-path).",
    )
    parser.add_argument(
        "--refresh-tag-cache",
        action="store_true",
        help="Refetch crawled_data/tag_version/*-version.jsonl from GitHub before resolving.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite rows even when version fields are unchanged.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Manual mode: set version to this value (no release-tag lookup). "
        "Use with --min-date / --instance-ids to filter rows.",
    )
    parser.add_argument(
        "--version-commit",
        type=str,
        default=None,
        help="Manual mode: also set version_commit; if omitted, version_commit is unchanged.",
    )
    args = parser.parse_args()

    if args.command == "jsonl-to-csv":
        csv_path = args.csv_path or str(Path(args.jsonl_path).with_suffix(".csv"))
        jsonl_to_csv(jsonl_path=args.jsonl_path, csv_path=csv_path)
    elif args.command == "refresh-work-patch-list":
        min_date = None if args.min_date.lower() == "none" else args.min_date
        refresh_work_patch_lists_in_jsonl(
            jsonl_path=args.jsonl_path,
            instance_ids=args.instance_ids,
            min_date=min_date,
            backup=not args.no_backup,
        )
    elif args.command == "remove-extra-instance-ids":
        if not args.first_jsonl_path or not args.second_jsonl_path:
            raise ValueError(
                "remove-extra-instance-ids requires --first-jsonl-path and --second-jsonl-path"
            )
        remove_extra_instance_ids_from_jsonl(
            first_jsonl_path=args.first_jsonl_path,
            second_jsonl_path=args.second_jsonl_path,
            backup=not args.no_backup,
        )
    elif args.command == "remove-instance-ids":
        if not args.jsonl_path:
            raise ValueError("remove-instance-ids requires --jsonl-path")
        if not args.remove_instance_ids and not args.instance_ids_file:
            raise ValueError(
                "remove-instance-ids requires --remove-instance-ids or --instance-ids-file"
            )
        remove_instance_ids_from_jsonl(
            dataset_paths=[args.jsonl_path],
            instance_ids=args.remove_instance_ids,
            instance_ids_file=args.instance_ids_file,
            backup=not args.no_backup,
        )
    else:
        min_date = None if args.min_date.lower() == "none" else args.min_date
        version_kwargs = dict(
            instance_ids=args.instance_ids,
            min_date=min_date,
            backup=not args.no_backup,
            refresh_tag_cache=args.refresh_tag_cache,
            force=args.force,
            set_version=args.version,
            set_version_commit=args.version_commit,
        )
        if args.activity_dir:
            refresh_versions_in_activity_dir(
                activity_dir=args.activity_dir,
                **version_kwargs,
            )
        else:
            refresh_versions_in_jsonl(
                jsonl_path=args.jsonl_path,
                **version_kwargs,
            )
