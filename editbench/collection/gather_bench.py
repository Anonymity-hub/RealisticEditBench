import argparse
import json
import os
from dataclasses import asdict

from editbench.collection.utils import Repo, extract_problem_statement_and_hints

from editbench.collection.instance.activity import write_json_line, Activity
from pathlib import Path

from editbench.editing_split.constants import EDITING_SPLIT_DIR
from typing import Union
from editbench.editing_split.validation import load_patch_list_instance

from editbench.utils.dataset_utils import get_inf_datasets
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()

def collect_bench_instance(ref_path: Union[Path, str], tar_path: Union[Path, str], 
                          instance_ids: Optional[List[str]] = None, overwrite: bool = False):
    """
    collect bench instances from tasks that have been split
    
    :param ref_path: reference execution-filtered task data path
    :param tar_path: target file path
    :param instance_ids: optional, specify the list of instance_ids to collect, if None then collect all
    :param overwrite: whether to overwrite the original file, if True then overwrite, if False then append (but will replace the existing instance_ids)
    """
    ref_instances = get_inf_datasets(ref_path)
    
    # if specified instance_ids, only process these
    if instance_ids is not None:
        instance_ids_set = set(instance_ids)
        ref_instances = [ins for ins in ref_instances if ins.instance_id in instance_ids_set]
    
    tar_path = Path(tar_path)
    
    # process new instances
    new_instances = {}
    target_instance_ids = set(instance_ids) if instance_ids is not None else None
    
    for ref_ins in ref_instances:
        instance_id = ref_ins.instance_id
        split_ins_path = EDITING_SPLIT_DIR / instance_id
        if not split_ins_path.exists():
            continue
        patch_list = load_patch_list_instance(ref_ins)
        if not patch_list:
            continue
        ref_ins.work_patch_list = patch_list
        # ref_ins = check_issues(ref_ins)
        new_instances[instance_id] = ref_ins
    
    # if overwrite=True, overwrite the file directly
    if overwrite:
        mode = "w"
        seen_ids = set()
    else:
        # if overwrite=False, need to read the existing file
        # if specified instance_ids, need to filter out these instance_ids to replace
        mode = "a" if tar_path.exists() else "w"
        seen_ids = set()
        
        if tar_path.exists():
            # if specified instance_ids, need to rewrite the file (filter out the instance_ids to replace)
            if target_instance_ids is not None:
                # read the existing file, filter out the instance_ids to replace
                existing_lines = []
                with open(tar_path, encoding="utf-8", mode="r") as fr:
                    for line in fr:
                        ins = json.loads(line)
                        if ins["instance_id"] not in target_instance_ids:
                            existing_lines.append(line.strip())
                        seen_ids.add(ins["instance_id"])
                
                # rewrite the file (first write the retained data)
                if existing_lines or new_instances:
                    with open(tar_path, encoding="utf-8", mode="w") as fw:
                        for line in existing_lines:
                            fw.write(line + "\n")
                    mode = "a"  # then append new data
            else:
                # if not specified instance_ids, append mode, skip the existing
                with open(tar_path, encoding="utf-8", mode="r") as fr:
                    for line in fr:
                        ins = json.loads(line)
                        seen_ids.add(ins["instance_id"])
    
    # write new data
    with open(tar_path, encoding="utf-8", mode=mode) as fo:
        for ref_ins in new_instances.values():
            instance_id = ref_ins.instance_id
            # if specified instance_ids, always write (replace the existing)
            # if not specified instance_ids and overwrite=True, always write
            # if not specified instance_ids and overwrite=False, skip the existing
            if target_instance_ids is not None or overwrite or instance_id not in seen_ids:
                write_json_line(ref_ins, fo, is_instance=True)


# TODO: check the function of check_issues, improve the issues information
def check_issues(instance: Activity, token=None):
    if not token:
        token = os.getenv("GITHUB_TOKENS").split(',')[0]
    owner, repo = instance.repo.split('/')
    repo = Repo(owner, repo, token=token)
    data = asdict(instance)
    resolved_issues = repo.extract_resolved_issues(data)
    issues_mes, other_mes = extract_problem_statement_and_hints(data, repo)
    if resolved_issues and len(resolved_issues) > len(instance.resolved_issues):
        instance.resolved_issues = resolved_issues
        instance.issues_mes = issues_mes
        instance.other_mes = other_mes
    return instance


if __name__ == "__main__":
    # Example:
    #   python -m editbench.collection.gather_bench \
    #     --ref-path ./crawled_data/execution_filter/all-task-instances.jsonl \
    #     --tar-path ./crawled_data/bench/all-task-instances.jsonl
    parser = argparse.ArgumentParser(description="Gather bench instances from split task data.")
    parser.add_argument(
        "--ref-path",
        type=str,
        required=True,
        help="Reference execution-filtered task data path.",
    )
    parser.add_argument(
        "--tar-path",
        type=str,
        required=True,
        help="Target file path to write bench instances.",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of instance_ids to collect; if omitted, collect all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target file; if False, append (and replace specified instance_ids).",
    )
    args = parser.parse_args()
    collect_bench_instance(
        ref_path=args.ref_path,
        tar_path=args.tar_path,
        instance_ids=args.instance_ids,
        overwrite=args.overwrite,
    )


