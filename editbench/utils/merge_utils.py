"""
merge utils for bench, infbench datasets and experimental results

merge multiple repositories' bench data and different conditions' infbench data and experimental results
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from editbench.config import SRC_BENCHMARK_DATA, SRC_INF_BENCHMARK_DATA
from editbench.inference.constants import EXPERIMENTAL_RESULTS


def _get_default_dirs() -> Tuple[Path, Path]:
    """
    get default bench and infbench directory paths
    """
    bench_dir = SRC_BENCHMARK_DATA
    infbench_dir = SRC_INF_BENCHMARK_DATA
    return bench_dir, infbench_dir


def merge_bench(output_path: Optional[Path] = None,
                repos: Optional[List[str]] = None) -> Path:
    """
    merge all jsonl files in crawled_data/bench directory
    
    :param output_path: output file path, default is bench_dir/all-task-instances.jsonl
    :param repos: list of repositories to merge, if None, merge all repositories
    :return: output file path
    """
    bench_dir, _ = _get_default_dirs()
    
    if output_path is None:
        output_path = bench_dir / "all-task-instances.jsonl"
    else:
        output_path = Path(output_path)
    
    # get all jsonl files
    jsonl_files = list(bench_dir.glob("*.jsonl"))

    # filter self
    jsonl_files = [f for f in jsonl_files if f.name != "all-task-instances.jsonl"]

    if repos is not None:
        # filter specified repositories
        repo_patterns = [repo.replace("/", "-") for repo in repos]
        jsonl_files = [
            f for f in jsonl_files 
            if any(repo in f.name for repo in repo_patterns)
        ]
    
    if not jsonl_files:
        raise ValueError(f"No jsonl files found in {bench_dir}")
    
    print(f"Found {len(jsonl_files)} files, merging...")
    
    total_lines = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for jsonl_file in sorted(jsonl_files):
            print(f"  processing: {jsonl_file.name}")
            file_lines = 0
            with open(jsonl_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:  # skip empty lines
                        outfile.write(line + '\n')
                        file_lines += 1
                        total_lines += 1
            print(f"    wrote {file_lines} records")
    
    print(f"Merge done. Total {total_lines} records, output: {output_path}")
    return output_path


def get_infbench_conditions() -> Dict[str, List[Path]]:
    """
    Get all files under infbench dir and group by condition.

    :return: dict, key = condition (e.g. '0.2', '0.4', 'body_issue'), value = list of file paths
    """
    _, infbench_dir = _get_default_dirs()

    jsonl_files = list(infbench_dir.glob("*.jsonl"))
    jsonl_files = [file for file in jsonl_files if "all-task-instances" not in file.name]

    # group by condition; filename format: {repo}-task-instances_{condition}.jsonl
    conditions = defaultdict(list)

    for file_path in jsonl_files:
        # extract condition, e.g. astropy-astropy-task-instances_0.2.jsonl -> 0.2
        match = re.search(r'-task-instances_(.+?)\.jsonl$', file_path.name)
        if match:
            condition = match.group(1)
            conditions[condition].append(file_path)
        else:
            print(f"Warning: could not parse filename format: {file_path.name}")
    
    return dict(conditions)


def merge_infbench_by_condition(condition: str,
                                 output_path: Optional[Path] = None,
                                 repos: Optional[List[str]] = None) -> Path:
    """
    Merge all infbench files for a given condition.

    :param condition: condition name (e.g. '0.2', '0.4', 'body_issue')
    :param output_path: output path, default infbench_dir/all-task-instances_{condition}.jsonl
    :param repos: repos to merge; if None, merge all
    :return: output file path
    """
    _, infbench_dir = _get_default_dirs()

    if output_path is None:
        output_path = infbench_dir / f"all-task-instances_{condition}.jsonl"
    else:
        output_path = Path(output_path)

    conditions = get_infbench_conditions()

    if condition not in conditions:
        available = ', '.join(conditions.keys())
        raise ValueError(f"Condition '{condition}' not found. Available: {available}")

    jsonl_files = conditions[condition]

    if repos is not None:
        repo_patterns = [repo.replace("/", "-") for repo in repos]
        jsonl_files = [
            f for f in jsonl_files
            if any(repo in f.name for repo in repo_patterns)
        ]

    if not jsonl_files:
        raise ValueError(f"No matching files for condition '{condition}'")

    print(f"Found {len(jsonl_files)} files (condition: {condition}), merging...")

    total_lines = 0
    seen_ids: Set[str] = set()

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for jsonl_file in sorted(jsonl_files):
            print(f"  processing: {jsonl_file.name}")
            file_lines = 0
            with open(jsonl_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            instance_id = data.get('instance_id')

                            if instance_id and instance_id in seen_ids:
                                continue

                            if instance_id:
                                seen_ids.add(instance_id)

                            outfile.write(line + '\n')
                            file_lines += 1
                            total_lines += 1
                        except json.JSONDecodeError as e:
                            print(f"    Warning: skip invalid JSON line: {e}")
                            continue

            print(f"    wrote {file_lines} records")

    print(f"Merge done. Total {total_lines} records (deduped), output: {output_path}")
    return output_path


def merge_all_infbench_conditions(output_dir: Optional[Path] = None,
                                 repos: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Merge all infbench files for all conditions.

    :param output_dir: output directory, default infbench_dir
    :param repos: repos to merge; if None, merge all
    :return: dict, key = condition, value = output file path
    """
    _, infbench_dir = _get_default_dirs()

    if output_dir is None:
        output_dir = infbench_dir
    else:
        output_dir = Path(output_dir)

    conditions = get_infbench_conditions()
    results = {}

    print(f"Found {len(conditions)} conditions, merging...")

    for condition in sorted(conditions.keys()):
        print(f"\nProcessing condition: {condition}")
        output_path = output_dir / f"all-task-instances_{condition}.jsonl"
        results[condition] = merge_infbench_by_condition(
            condition, output_path, repos
        )
    
    return results


def list_available_conditions() -> List[str]:
    """
    List all available infbench conditions.

    :return: list of condition names
    """
    conditions = get_infbench_conditions()
    return sorted(conditions.keys())


def merge_experimental_results(
    model_name: str,
    temperature: Optional[float] = None,
    n: Optional[int] = None,
    output_path: Optional[Path] = None,
    repos: Optional[List[str]] = None,
    run_id: Optional[str] = None
) -> List[Path]:
    """
    Merge all jsonl files under experimental_results for a model.

    Merges {owner-repo}-task-instances_{run_id}.jsonl under
    experimental_results/{model_name}/T={temperature}/n={n}/.
    Produces all-task-instances_{run_id}.jsonl per run_id; data grouped by repo.

    :param model_name: model name (e.g. "gemini-2.5-pro")
    :param temperature: if None, use all T=* dirs
    :param n: if None, use all n=* dirs
    :param output_path: output path prefix; default {model_dir}/T=.../n=.../
    :param repos: repos to merge (e.g. ["owner/repo"]); if None, merge all
    :param run_id: run_id to merge (e.g. "0.2"); if None, merge all run_ids
    :return: list of output file paths
    """
    base_dir = EXPERIMENTAL_RESULTS / model_name

    if not base_dir.exists():
        raise ValueError(f"Model dir does not exist: {base_dir}")

    # find all T=* and n=* dirs
    if temperature is not None:
        t_dirs = [base_dir / f"T={temperature}"]
    else:
        t_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("T=")]
    
    if not t_dirs:
        raise ValueError(f"No T=* dirs found in {base_dir}")
    
    all_jsonl_files = []
    
    for t_dir in t_dirs:
        if n is not None:
            n_dirs = [t_dir / f"n={n}"]
        else:
            n_dirs = [d for d in t_dir.iterdir() if d.is_dir() and d.name.startswith("n=")]
        
        for n_dir in n_dirs:
            jsonl_files = list(n_dir.glob("*.jsonl"))
            jsonl_files = [
                f for f in jsonl_files 
                if not f.name.startswith("all-task-instances") 
                and "-task-instances_" in f.name
            ]
            
            if run_id is not None:
                jsonl_files = [
                    f for f in jsonl_files
                    if f.name.endswith(f"-task-instances_{run_id}.jsonl")
                ]
            
            all_jsonl_files.extend(jsonl_files)
    
    if not all_jsonl_files:
        raise ValueError("No jsonl files to merge")

    if repos is not None:
        repo_patterns = [repo.replace("/", "-") for repo in repos]
        all_jsonl_files = [
            f for f in all_jsonl_files
            if any(repo in f.name for repo in repo_patterns)
        ]
    
    if not all_jsonl_files:
        raise ValueError("No matching jsonl files after filter")

    # group files by run_id
    files_by_run_id: Dict[str, List[Path]] = defaultdict(list)
    
    for jsonl_file in all_jsonl_files:
        file_run_id = None
        if "-task-instances_" in jsonl_file.name:
            parts = jsonl_file.name.replace(".jsonl", "").split("-task-instances_")
            if len(parts) == 2:
                file_run_id = parts[1]
        
        if file_run_id:
            files_by_run_id[file_run_id].append(jsonl_file)
    
    if not files_by_run_id:
        raise ValueError(f"Could not extract run_id from filenames")
    
    # determine output directory
    if output_path is None:
        output_dir = all_jsonl_files[0].parent
    else:
        output_path = Path(output_path)
        if output_path.is_file():
            output_dir = output_path.parent
        else:
            output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(all_jsonl_files)} files in {len(files_by_run_id)} run_id groups, merging...")

    output_files = []

    for run_id_key, jsonl_files in sorted(files_by_run_id.items()):
        print(f"\nProcessing run_id: {run_id_key} ({len(jsonl_files)} files)")

        repo_data: Dict[str, List[Dict]] = defaultdict(list)
        seen_ids: Set[str] = set()

        for jsonl_file in sorted(jsonl_files):
            print(f"  loading: {jsonl_file.name}")
            file_count = 0
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            instance_id = data.get('instance_id', '')
                            
                            if instance_id and instance_id in seen_ids:
                                continue
                            if instance_id:
                                seen_ids.add(instance_id)
                            
                            # extract repo information
                            repo = 'unknown'
                            if instance_id:
                                # instance_id format: owner__repo-src_type-number
                                # ensure repo part is not "pull"
                                # match = re.match(r'^([^_]+)__([^_]+?)-(?!pull)', instance_id)
                                match = re.match(r'^([^_]+)__([^_]+)-pull', instance_id)
                                if match:
                                    repo = f"{match.group(1)}/{match.group(2)}"
                            
                            # if repos is specified, only keep matching repos
                            if repos is None or repo in repos:
                                repo_data[repo].append(data)
                                file_count += 1
                        except json.JSONDecodeError as e:
                            print(f"    Warning: skip invalid JSON line: {e}")
                            continue
                print(f"    loaded {file_count} records")
            except Exception as e:
                print(f"    Warning: error loading file: {e}")
                continue
        
        # generate output file path
        output_file = output_dir / f"all-task-instances_{run_id_key}.jsonl"
        
        # sort by repo and write
        print(f"  sorting and writing to: {output_file.name}")
        total_lines = 0
        
        # sort by repo name
        sorted_repos = sorted(repo_data.keys())
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for repo in sorted_repos:
                data_list = repo_data[repo]
                # if created_at field exists, sort by time from recent to distant
                if data_list and 'created_at' in data_list[0]:
                    data_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                
                print(f"    writing repo {repo}: {len(data_list)} records")
                for data in data_list:
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    total_lines += 1

        print(f"  Done run_id {run_id_key}. Total {total_lines} records (deduped), {len(sorted_repos)} repos")
        output_files.append(output_file)
    
    print(f"\nAll merges done. Generated {len(output_files)} files:")
    for output_file in output_files:
        print(f"  - {output_file}")
    
    return output_files


if __name__ == "__main__":
    # examples:
    #   python -m editbench.utils.merge_utils merge-bench [--output-path PATH] [--repos repo1 repo2]
    #   python -m editbench.utils.merge_utils merge-infbench --condition 0.2 [--output-path PATH] [--repos repo1]
    #   python -m editbench.utils.merge_utils merge-all-infbench [--output-dir DIR] [--repos repo1]
    #   python -m editbench.utils.merge_utils merge-results --model-name qwen3-235b-a22b [--temperature 0] [--n 1] [--run-id 0.2] [--repos repo1]
    parser = argparse.ArgumentParser(
        description="Merge bench, infbench datasets and experimental results.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand")

    # merge-bench
    p_bench = subparsers.add_parser("merge-bench", help="Merge all bench jsonl files into all-task-instances.jsonl")
    p_bench.add_argument("--output-path", type=str, default=None, help="Output path (default: bench_dir/all-task-instances.jsonl)")
    p_bench.add_argument("--repos", type=str, nargs="+", default=None, help="Repos to merge (e.g. owner/repo); if omitted, merge all")

    # merge-infbench
    p_inf = subparsers.add_parser("merge-infbench", help="Merge infbench files for one condition")
    p_inf.add_argument("--condition", type=str, required=True, help="Condition name (e.g. 0.2, 0.2_body_issue)")
    p_inf.add_argument("--output-path", type=str, default=None, help="Output path (default: infbench_dir/all-task-instances_{condition}.jsonl)")
    p_inf.add_argument("--repos", type=str, nargs="+", default=None, help="Repos to merge; if omitted, merge all")

    # merge-all-infbench
    p_all = subparsers.add_parser("merge-all-infbench", help="Merge infbench for all conditions")
    p_all.add_argument("--output-dir", type=str, default=None, help="Output directory (default: infbench_dir)")
    p_all.add_argument("--repos", type=str, nargs="+", default=None, help="Repos to merge; if omitted, merge all")

    # merge-results
    p_res = subparsers.add_parser("merge-results", help="Merge experimental result jsonl files by model")
    p_res.add_argument("--model-name", type=str, required=True, help="Model name (e.g. qwen3-235b-a22b)")
    p_res.add_argument("--temperature", type=float, default=None, help="Temperature (e.g. 0); if omitted, use all T=* dirs")
    p_res.add_argument("--n", type=int, default=None, help="n (e.g. 1); if omitted, use all n=* dirs")
    p_res.add_argument("--output-path", type=str, default=None, help="Output path or dir prefix (default: model run dir)")
    p_res.add_argument("--repos", type=str, nargs="+", default=None, help="Repos to merge; if omitted, merge all")
    p_res.add_argument("--run-id", type=str, default=None, help="Run ID to merge (e.g. 0.2); if omitted, merge all run_ids")

    args = parser.parse_args()

    if args.command == "merge-bench":
        merge_bench(
            output_path=Path(args.output_path) if args.output_path else None,
            repos=args.repos,
        )
    elif args.command == "merge-infbench":
        merge_infbench_by_condition(
            args.condition,
            output_path=Path(args.output_path) if args.output_path else None,
            repos=args.repos,
        )
    elif args.command == "merge-all-infbench":
        merge_all_infbench_conditions(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            repos=args.repos,
        )
    elif args.command == "merge-results":
        merge_experimental_results(
            model_name=args.model_name,
            temperature=args.temperature,
            n=args.n,
            output_path=Path(args.output_path) if args.output_path else None,
            repos=args.repos,
            run_id=args.run_id,
        )