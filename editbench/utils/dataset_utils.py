import csv
import json
import re
from collections import defaultdict
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional

import pandas as pd
from editbench.collection.instance.activity import Activity, load_datasets_from_jsonl


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
    # dataset = InfInstance.load_inf_instances_from_ext(dataset_name)
    dataset = list(load_datasets_from_jsonl(dataset_name))


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


def jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    """
    Convert an Activity JSONL file to a CSV file (UTF-8 encoded).

    :param jsonl_path: Path to the input JSONL file
    :param csv_path: Path to save the output CSV file
    """
    field_names = [f.name for f in fields(Activity)]

    # Read and parse JSONL data
    instances: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i_line, line in enumerate(f):
            try:
                instance_dict = json.loads(line.strip())
                # Ensure all fields exist (fill missing fields with default values)
                for field in field_names:
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
            for list_field in ["files", "files_work", "files_test",
                               "files_other", "files_no_edit", "resolved_issues", "other_mes",
                               "fail_to_pass", "pass_to_fail", "fail_to_fail", "pass_to_pass"]:
                instance[list_field] = json.dumps(instance[list_field])

            writer.writerow(instance)


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
