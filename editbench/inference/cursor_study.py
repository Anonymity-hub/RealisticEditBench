"""
Cursor study tools for processing task instances and managing repository states.
"""
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

from editbench.evaluation.constants import MAP_INSTALLED_REPO
from editbench.inference.constants import EXPERIMENTAL_RESULTS
from editbench.utils.dataset_utils import get_inf_datasets
from editbench.config import SRC_INF_BENCHMARK_DATA


def clean_prompt(prompt: str) -> str:
    """
    Clean the prompt: remove the content between <code></code> and the content after "Please respond only with a single patch file"
    
    :param prompt: original prompt
    :return: cleaned prompt
    """
    # Remove the content between <code></code> (including the tags themselves)
    prompt = re.sub(r'<code>.*?</code>', '', prompt, flags=re.DOTALL)
    
    # Remove the sentence " by generating a single patch file that can be directly applied to this repository using git apply"
    prompt = prompt.replace(" by generating a single patch file that can be directly applied to this repository using git apply", "")
    
    # Remove the content after "Please respond only with a single patch file"
    idx = prompt.find("Please respond only with a single patch file")
    if idx != -1:
        prompt = prompt[:idx].rstrip()
    
    return prompt.strip()


def process_task_instances(
    input_file: str,
    run_id: str = "0.2",
    sampled_ids_file: Optional[str] = None
) -> None:
    """
    Process task instances: use get_inf_datasets to read tasks, clean the prompt, create directory structure and save prompt.txt
    
    :param input_file: input JSONL file path
    :param run_id: run ID, used to create directory structure
    :param sampled_ids_file: sampled JSON file path, if provided, only process sampled tasks
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    # Create output directory
    output_base = EXPERIMENTAL_RESULTS / "cursor" / run_id
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Use get_inf_datasets to get tasks (support sampled filtering)
    print(f"Loading dataset: {input_file}")
    if sampled_ids_file:
        print(f"Using sampled file: {sampled_ids_file}")
    
    activities = get_inf_datasets(
        dataset_name=input_file,
        split="",
        sampled_ids_file=sampled_ids_file
    )
    
    print(f"Loaded {len(activities)} task instances")
    
    # Process each task instance
    for activity in activities:
        instance_id = activity.instance_id
        if not instance_id:
            print(f"Warning: skipping task without instance_id")
            continue
        
        # Clean the prompt
        original_prompt = activity.prompt if hasattr(activity, 'prompt') else ''
        cleaned_prompt = clean_prompt(original_prompt)
        
        # Create instance directory
        instance_dir = output_base / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the cleaned prompt
        prompt_file = instance_dir / "prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as pf:
            pf.write(cleaned_prompt)
        
        print(f"Processed task {instance_id}: {prompt_file}")


def reset_repo_for_task(
    instance_id: str,
    run_id: str = "0.2",
    apply_pre_edits: bool = False,
) -> None:
    """
    Reset the repository to base_commit; optionally apply pre_edits and add to staging.

    :param instance_id: task instance ID
    :param run_id: run ID, used to build data file path
    :param apply_pre_edits: if True, apply pre_edits and add to staging; if False (default), only checkout base_commit
    """
    # Build data file path and sampled file path
    dataset_file = SRC_INF_BENCHMARK_DATA / f"all-task-instances_{run_id}.jsonl"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Data file does not exist: {dataset_file}")
    
    # use get_inf_datasets to load tasks (with sampled filtering)
    print(f"Loading dataset: {dataset_file}")

    activities = get_inf_datasets(
        dataset_name=str(dataset_file),
        split="",
    )
    
    # Find the corresponding task
    target_activity = None
    for activity in activities:
        if activity.instance_id == instance_id:
            target_activity = activity
            break
    
    if target_activity is None:
        raise ValueError(f"Instance ID not found: {instance_id}")
    
    # Get repo and base_commit from the task
    repo = target_activity.repo
    base_commit = target_activity.base_commit
    
    if not repo:
        raise ValueError(f"Task {instance_id} has no repo field")
    if not base_commit:
        raise ValueError(f"Task {instance_id} has no base_commit field")
    
    print(f"Found task: {instance_id}")
    print(f"  Repository: {repo}")
    print(f"  Base commit: {base_commit}")
    
    # Get repository path
    repo_path = MAP_INSTALLED_REPO.get(repo)
    if not repo_path:
        raise ValueError(f"Repository {repo} not in MAP_INSTALLED_REPO")
    
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    
    # First clean the working area and staging area
    try:
        # Empty the staging area
        subprocess.run(
            ["git", "reset", "HEAD"],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True
        )
        
        # Discard the changes in the working area
        subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True
        )
    except Exception as e:
        print(f"Warning: error cleaning working area (maybe no changes): {e}")
    
    # Switch to base_commit
    try:
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to commit {base_commit}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Switch commit failed: {e.stderr}")

    if not apply_pre_edits:
        print(f"Repository {repo} reset to base_commit {base_commit} (pre_edits skipped)")
        return

    # Apply pre_edits (if exists)
    pre_edits = getattr(target_activity, 'pre_edits', []) or []
    if pre_edits:
        print(f"Starting to apply {len(pre_edits)} pre_edits...")
        
        # Apply each patch directly
        for idx, patch in enumerate(pre_edits):
            try:
                # Extract file name for logging
                file_match = re.findall(r"--- a/(.*)", patch)
                file_info = file_match[0] if file_match else "unknown"
                print(f"  Applying patch-{idx + 1} (file: {file_info})...")
                
                # Use subprocess to execute git apply, passing patch through stdin
                result = subprocess.run(
                    ["git", "apply", "-v"],
                    cwd=repo_path,
                    input=patch,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"  ✓ Applied patch-{idx + 1}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to apply pre_edit patch-{idx + 1}: {e.stderr}")
        
        print(f"Applied all {len(pre_edits)} pre_edits")
        
        # Add the changes applied by pre_edits to the staging area
        try:
            # Get all modified files
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            modified_files = result.stdout.strip().split('\n')
            modified_files = [f for f in modified_files if f.strip()]
            
            if modified_files:
                # Add all modified files to the staging area
                subprocess.run(
                    ["git", "add"] + modified_files,
                    cwd=repo_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Added {len(modified_files)} modified files to the staging area: {', '.join(modified_files[:5])}{'...' if len(modified_files) > 5 else ''}")
            else:
                print("No file modifications detected")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to add files to the staging area: {e.stderr}")
    else:
        print("No pre_edits to apply")
    
    # Ensure the staging area is correct
    try:
        # Check the status of the staging area
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        staged_files = [f for f in result.stdout.strip().split('\n') if f.strip()]
        
        if staged_files:
            print(f"Staging area contains {len(staged_files)} files")
        else:
            print("Staging area is empty")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to check git status: {e.stderr}")
    
    print(f"Repository {repo} reset to {base_commit}, pre_edits applied and added to the staging area")


def collect_patch_and_save_result(
    run_id: str,
    instance_id: str,
    overwrite: bool = False
) -> Dict:
    """
    Collect manual modified patch and save the result
    
    :param run_id: run ID, used to build data file path and sampled file path
    :param instance_id: task instance ID
    :return: task result dictionary
    """
    # Build data file path and sampled file path
    dataset_file = SRC_INF_BENCHMARK_DATA / f"all-task-instances_{run_id}.jsonl"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Data file does not exist: {dataset_file}")
    
    # use get_inf_datasets to load tasks (with sampled filtering)
    print(f"Loading dataset: {dataset_file}")
    
    activities = get_inf_datasets(
        dataset_name=str(dataset_file),
        split="",
    )
    
    # Find the corresponding task
    target_activity = None
    for activity in activities:
        if activity.instance_id == instance_id:
            target_activity = activity
            break
    
    if target_activity is None:
        raise ValueError(f"Instance ID not found: {instance_id}")
    
    # Get repo from the task
    repo = target_activity.repo
    
    if not repo:
        raise ValueError(f"Task {instance_id} has no repo field")
    
    print(f"Found task: {instance_id}")
    print(f"  Repository: {repo}")
    
    # Get repository path
    repo_path = MAP_INSTALLED_REPO.get(repo)
    if not repo_path:
        raise ValueError(f"Repository {repo} not in MAP_INSTALLED_REPO")
    
    repo_path = Path(repo_path)
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    
    # Get patch
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        patch_content = result.stdout
        
        # Remove index lines (e.g. index 53fd458c1f..b89ca8a19f 100644)
        lines = patch_content.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip lines starting with index
            if line.startswith('index '):
                continue
            filtered_lines.append(line)
        patch_content = '\n'.join(filtered_lines)
        
        # Ensure there is a newline at the end
        if patch_content and not patch_content.endswith('\n'):
            patch_content += "\n"
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get patch: {e.stderr}")
    
    if not patch_content.strip():
        print(f"Warning: task {instance_id} has no changes detected")
        patch_content = ""
        raise ValueError(f"Task {instance_id} has no changes detected")
    
    # Save patch file
    output_base = EXPERIMENTAL_RESULTS / "cursor" / run_id
    instance_dir = output_base / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    patch_file = instance_dir / "prediction.patch"

    if patch_file.exists() and not overwrite:
        print(f"Warning: patch file already exists, skipping save: {patch_file}")
        return None
    
    with open(patch_file, 'w', encoding='utf-8') as pf:
        pf.write(patch_content)
    
    print(f"Saved patch to: {patch_file}")
    
    # Read prompt
    prompt_file = instance_dir / "prompt.txt"
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as pf:
            prompt = pf.read()
    else:
        prompt = ""
        print(f"Warning: prompt.txt does not exist, using empty string")
    
    # Assemble result
    result = {
        "instance_id": instance_id,
        "model_name": "cursor",
        "n": 1,
        "prompt": prompt,
        "model_patch": patch_content
    }
    
    return result


def save_result_to_jsonl(
    result: Dict,
    run_id: str = "0.2",
    output_file: Optional[str] = None
) -> Path:
    """
    Save the result to JSONL file, if instance_id already exists, skip
    
    :param result: task result dictionary
    :param run_id: run ID
    :param output_file: output file path (optional, default is all-task-instances_{run_id}.jsonl)
    :return: saved file path
    """
    output_base = EXPERIMENTAL_RESULTS / "cursor" 
    
    if output_file is None:
        output_file = output_base / f"T=1/n=1/all-task-instances_{run_id}.jsonl"
    else:
        output_file = Path(output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    instance_id = result.get("instance_id")
    if not instance_id:
        raise ValueError("Instance ID is missing in the result")
    
    # Check if the file exists, if exists, read the existing record
    existing_ids = set()
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        existing_result = json.loads(line)
                        existing_id = existing_result.get("instance_id")
                        if existing_id:
                            existing_ids.add(existing_id)
        except Exception as e:
            print(f"Warning: failed to read existing file: {e}, will create a new file")
    
    # Check if instance_id already exists
    if instance_id in existing_ids:
        print(f"Skipping: instance_id {instance_id} already exists in {output_file}")
        return output_file
    
    # Append new record
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Appended result to: {output_file} (instance_id: {instance_id})")
    return output_file


def batch_reset_repos_from_jsonl(
    run_id: str = "0.2"
) -> None:
    """
    Batch reset sampled tasks' repositories based on run_id
    
    :param run_id: run ID, used to build data file path and sampled file path
    """
    # Build data file path and sampled file path
    dataset_file = SRC_INF_BENCHMARK_DATA / f"all-task-instances_{run_id}.jsonl"
    sampled_ids_file = SRC_INF_BENCHMARK_DATA / f"sampled_instance_ids_{run_id}.json"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Data file does not exist: {dataset_file}")
    
    # Use get_inf_datasets to load tasks (with sampled filtering)
    print(f"Loading dataset: {dataset_file}")
    if sampled_ids_file.exists():
        print(f"Using sampled file: {sampled_ids_file}")
    
    activities = get_inf_datasets(
        dataset_name=str(dataset_file),
        split="",
        sampled_ids_file=str(sampled_ids_file) if sampled_ids_file.exists() else None
    )
    
    print(f"Loaded {len(activities)} task instances, starting batch reset...")
    
    # Batch reset each task
    for activity in activities:
        instance_id = activity.instance_id
        try:
            print(f"\nProcessing task: {instance_id}")
            reset_repo_for_task(instance_id, run_id)
        except Exception as e:
            print(f"Error: failed to process task {instance_id}: {e}")
            continue


def batch_collect_patches_from_jsonl(
    run_id: str = "0.2"
) -> None:
    """
    Batch collect sampled tasks' patches based on run_id
    
    :param run_id: run ID, used to build data file path and sampled file path
    """
    # Build data file path and sampled file path
    dataset_file = SRC_INF_BENCHMARK_DATA / f"all-task-instances_{run_id}.jsonl"
    sampled_ids_file = SRC_INF_BENCHMARK_DATA / f"sampled_instance_ids_{run_id}.json"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Data file does not exist: {dataset_file}")
    
    # Use get_inf_datasets to load tasks (with sampled filtering)
    print(f"Loading dataset: {dataset_file}")
    if sampled_ids_file.exists():
        print(f"Using sampled file: {sampled_ids_file}")
    
    activities = get_inf_datasets(
        dataset_name=str(dataset_file),
        split="",
        sampled_ids_file=str(sampled_ids_file) if sampled_ids_file.exists() else None
    )
    
    print(f"Loaded {len(activities)} task instances, starting batch collect patches...")
    
    results = []
    
    # Batch collect each task's patch
    for activity in activities:
        instance_id = activity.instance_id
        try:
            print(f"\nProcessing task: {instance_id}")
            result = collect_patch_and_save_result(run_id, instance_id)
            results.append(result)
            save_result_to_jsonl(result, run_id)
        except Exception as e:
            print(f"Error: failed to process task {instance_id}: {e}")
            continue
    
    # Save summary result
    if results:
        output_base = EXPERIMENTAL_RESULTS / "cursor" / run_id
        summary_file = output_base / "all-results.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary result to: {summary_file}")


if __name__ == "__main__":
    # Example usage
    import sys
    

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Process task instances: python cursor_study.py process <input_file> [run_id] [sampled_ids_file]")
        print("  Reset repository (single): python cursor_study.py reset <run_id> <instance_id> [--pre-edits]")
        print("  Batch reset repositories: python cursor_study.py batch-reset [run_id]")
        print("  Collect patch (single): python cursor_study.py collect <run_id> <instance_id>")
        print("  Batch collect patches: python cursor_study.py batch-collect [run_id]")
        sys.exit(1)
    # python editbench/inference/cursor_study.py reset 0.2 astropy__astropy-pull-17514
    # python editbench/inference/cursor_study.py collect 0.2 astropy__astropy-pull-17514
    
    
    command = sys.argv[1]
    
    if command == "process":
        input_file = sys.argv[2]
        run_id = sys.argv[3] if len(sys.argv) > 3 else "0.2"
        sampled_ids_file = sys.argv[4] if len(sys.argv) > 4 else None
        process_task_instances(input_file, run_id, sampled_ids_file)
    
    elif command == "reset":
        args_reset = [a for a in sys.argv[2:] if a != "--pre-edits"]
        apply_pre_edits = "--pre-edits" in sys.argv
        run_id = args_reset[0] if args_reset else None
        instance_id = args_reset[1] if len(args_reset) > 1 else None
        if not run_id or not instance_id:
            print("Error: reset requires <run_id> <instance_id>")
            sys.exit(1)
        reset_repo_for_task(instance_id, run_id, apply_pre_edits=apply_pre_edits)
    
    elif command == "batch-reset":
        run_id = sys.argv[2] if len(sys.argv) > 2 else "0.2"
        batch_reset_repos_from_jsonl(run_id)
    
    elif command == "collect":
        run_id = sys.argv[2]
        instance_id = sys.argv[3] if len(sys.argv) > 3 else None
        if not instance_id:
            print("Error: instance_id is required")
            sys.exit(1)
        result = collect_patch_and_save_result(run_id, instance_id)
        save_result_to_jsonl(result, run_id)
    
    elif command == "batch-collect":
        run_id = sys.argv[2] if len(sys.argv) > 2 else "0.2"
        batch_collect_patches_from_jsonl(run_id)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
