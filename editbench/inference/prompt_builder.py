import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from editbench.config import SRC_BENCHMARK_DATA, SRC_INF_BENCHMARK_DATA
from editbench.editing_split.validation import validation_inf

from editbench.collection.instance.activity import Activity, write_json_line
from editbench.editing_split.diff_utils import quick_split_diff
from editbench.inference.utils import prompt_replace_regex, fill_line_number
from editbench.inference.bm25_retrieval import bm25_retrieve_files
from editbench.utils.dataset_utils import get_inf_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

patch_exp = """<patch>
diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file2.py
@@ -114,9 +114,8 @@ def __eq__(self, other):


def close_caches(**kwargs):
- # Some caches -- python-memcached in particular -- need to do a cleanup at the
- # end of a request cycle. If not implemented in a particular backend
- # cache.close is a no-op
+ # Some caches need to do a cleanup at the end of a request cycle. If not
+ # implemented in a particular backend cache.close() is a no-op.
for cache in caches.all():
cache.close()

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -3,10 +3,12 @@
import pickle
import re
import time
+import warnings

from django.core.cache.backends.base import (
DEFAULT_TIMEOUT, BaseCache, InvalidCacheKey, memcache_key_warnings,
)
+from django.utils.deprecation import RemovedInDjango41Warning
from django.utils.functional import cached_property


@@ -164,6 +166,11 @@ def validate_key(self, key):
class MemcachedCache(BaseMemcachedCache):
"An implementation of a cache binding using python-memcached"
def __init__(self, server, params):
+ warnings.warn(
+ 'MemcachedCache is deprecated in favor of PyMemcacheCache and '
+ 'PyLibMCCache.',
+ RemovedInDjango41Warning, stacklevel=2,
+ )
# python-memcached ≥ 1.45 returns None for a nonexistent key in
# incr/decr(), python-memcached < 1.45 raises ValueError.
import memcache
</patch>"""

PROMPT_STYLE = """You will be provided with 1) a partial code base, 2) a code editing requirement describing the target modification, and 3) an ordered history of prior code edits (typically Git diffs) made to progress toward solving the task.
<requirement>
${problem_statement}$
</requirement>
<code>
${code_context}$
</code>
<change_history>
${historical_code_changes}$
</change_history>
I need you to comprehensively understand the content and purpose of the above multi-step historical code changes in relation to the given editing task requirement. Based on this, recommend all the remaining edits to address the editing objective the developer intends to accomplish by generating a single patch file that can be directly applied to this repository using git apply. Please note that you do not need to consider editing test files. Please respond only with a single patch file in the following format.
Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.
""" + f"{patch_exp}"

PROMPT_STYLE_ONLY_CHANGES = """You will be provided with 1) a partial code base and 2) an ordered history of prior code edits (typically Git diffs) made to progress toward solving the task.
<code>
${code_context}$
</code>
<change_history>
${historical_code_changes}$
</change_history>
I need you to comprehensively understand the content and purpose of the above multi-step historical code changes. Based on this, recommend all the remaining edits to address the editing objective the developer intends to accomplish by generating a single patch file that can be directly applied to this repository using git apply. Please note that you do not need to consider editing test files. Please respond only with a single patch file in the following format.
Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.
""" + f"{patch_exp}"

PROMPT_STYLE_ONLY_ISSUE = """You will be provided with 1) a partial code base and 2) a code editing requirement describing the target modification.
<requirement>
${problem_statement}$
</requirement>
<code>
${code_context}$
</code>
I need you to solve the given editing task requirement by generating a single patch file that can be directly applied to this repository using git apply. Please note that you do not need to consider editing test files. Please respond only with a single patch file in the following format.
Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.
""" + f"{patch_exp}"


def with_prompt(instance: Activity, info_pct: Optional[float] = None, include_pr_mes: bool = False,
                include_issue_mes: bool = False, use_bm25: bool = False, topn: int = 10):
    """

    :param instance:
    :param info_pct:
    :param include_pr_mes:
    :param include_issue_mes:
    :param use_bm25: whether to use BM25 retrieval, if True, use BM25 retrieval result to replace oracle content
    :param topn: number of top-n files returned by BM25 retrieval
    :return:
    """
    # editing steps
    patch_list = instance.work_patch_list

    # number of editing steps
    patch_len = len(patch_list)

    # split ratio
    info_pct = info_pct if info_pct else 0

    # split position
    split_idx, ok = 0, True

    if patch_len >= 2 and info_pct > 0:
        # calculate original split position (rounding)
        raw_idx = round(patch_len * info_pct)

        if patch_len == 2:
            # when there are 2 steps, the non-0 ratio must be split at 1 (guarantee 1 prediction content)
            split_idx = 1
            ok = True if raw_idx <= 1 else False
        else:
            # when the number of steps ≥ 3, it is necessary to ensure that at least 1 previous step and at least 1 prediction
            if 1 <= raw_idx <= (patch_len - 1):
                split_idx = raw_idx
            else:
                split_idx = 1 if raw_idx < 1 else patch_len - 1
                ok = False

    pre_edits = [(joined := "\n".join(p_list)) + ("\n" if len(p_list) == 1 else "") for p_list in patch_list[:split_idx]]
    ground_truth, file_context = quick_split_diff(instance.instance_id, split_idx)

    # if using BM25 retrieval, use retrieval result to replace oracle content
    if use_bm25:
        if not instance.base_commit:
            logger.warning(f"Instance {instance.instance_id} has no base_commit, cannot use BM25 retrieval, revert to oracle content")
        else:
            # check if the repository is in MAP_INSTALLED_REPO
            from editbench.evaluation.constants import MAP_INSTALLED_REPO
            if instance.repo not in MAP_INSTALLED_REPO:
                logger.warning(f"Repository {instance.repo} is not in MAP_INSTALLED_REPO, cannot use BM25 retrieval, revert to oracle content")
            else:
                try:
                    # use BM25 retrieval to get file context
                    bm25_file_context = bm25_retrieve_files(
                        repo=instance.repo,
                        base_commit=instance.base_commit,
                        pre_edits=pre_edits,
                        topn=topn
                    )
                    if bm25_file_context:
                        file_context = bm25_file_context
                        logger.info(f"Using BM25 retrieval to get {len(file_context)} files for instance {instance.instance_id}")
                    else:
                        logger.warning(f"BM25 retrieval did not return any files, revert to oracle content")
                except Exception as e:
                    logger.error(f"BM25 retrieval failed: {e}, revert to oracle content")
                    raise RuntimeError(f"Failed to apply pre_edit patch: ")

    instance.split_idx = split_idx
    instance.pre_edits = pre_edits
    instance.ground_truth = ground_truth
    instance.num_changes = len(instance.work_patch_list)

    requirement = ""
    if include_pr_mes:
        requirement += instance.title_mes + "\n" + instance.body_mes + "\n"
        # filter unuseful mes
        match_pattern = r"<!--.*?-->"
        requirement = re.sub(match_pattern, '', requirement, flags=re.DOTALL)
        requirement = re.sub(r'\r\n?', '\n', requirement)
        requirement = re.sub(r'\n+', '\n', requirement).strip()
    if include_issue_mes:
        requirement += instance.issues_mes

    prompt_style = ""
    if requirement.strip() and split_idx != 0:
        prompt_style = PROMPT_STYLE
    elif requirement.strip() and split_idx == 0:
        prompt_style = PROMPT_STYLE_ONLY_ISSUE
    elif not requirement.strip() and split_idx != 0:
        prompt_style = PROMPT_STYLE_ONLY_CHANGES

    instance.prompt = get_prompt(prompt_style, requirement, file_context, pre_edits)

    return instance, ok


def remove_last_file_from_prompt(prompt: str) -> str:
    """
    Remove the last file from the code_context part of the prompt
    
    :param prompt: original prompt
    :return: prompt after removing the last file
    """
    # match the content between <code> and </code>
    code_pattern = r'<code>(.*?)</code>'
    match = re.search(code_pattern, prompt, re.DOTALL)
    
    if not match:
        # if there is no <code> tag, return the original prompt
        return prompt
    
    code_context = match.group(1)
    
    # parse the file list, format: [start of {filename}]\n{content}\n[end of {filename}]
    file_pattern = r'\[start of ([^\]]+)\]\n(.*?)\n\[end of \1\]'
    file_matches = list(re.finditer(file_pattern, code_context, re.DOTALL))
    
    if len(file_matches) == 0:
        # if no file is found, return the original prompt
        return prompt
    
    # remove the last file (but at least keep one file)
    if len(file_matches) == 1:
        # if there is only one file, try to half the file content (at least keep 1000 lines)
        file_match = file_matches[0]
        filename = file_match.group(1)
        file_content = file_match.group(2)
        
        # calculate the number of lines in the file
        lines = file_content.split('\n')
        total_lines = len(lines)
        
        # if the number of lines in the file >= 2000, half (at least keep 1000 lines)
        # if the number of lines in the file < 2000, halfing will be less than 1000 lines, so no modification
        print(f"Single file compression")
        if total_lines >= 2000:
            # keep the first half of the lines (because total_lines >= 2000, so halfing will at least keep 1000 lines)
            keep_lines = total_lines // 2
            new_file_content = '\n'.join(lines[:keep_lines])
            
            # rebuild code_context (only include the modified file content)
            new_code_context = f"[start of {filename}]\n{new_file_content}\n[end of {filename}]"
            
            # replace the code_context in the prompt
            new_prompt = prompt[:match.start(1)] + '\n' + new_code_context + '\n' + prompt[match.end(1):]
            return new_prompt
        else:
            # the number of lines in the file < 2000, halfing will be less than 1000 lines, so no modification
            return prompt
    else:
        # remove the last file match result
        last_match = file_matches[-1]
        # rebuild code_context, exclude the last file
        new_code_context = code_context[:last_match.start()].rstrip()
        
        # replace the code_context in the prompt
        new_prompt = prompt[:match.start(1)] + "\n" + new_code_context + "\n" + prompt[match.end(1):]
        
        return new_prompt


def get_prompt(prompt_style, requirement="", file_context=None, patch_list=None):
    patch_list = patch_list if patch_list else []
    file_context = file_context if file_context else {}
    source_code = ""
    for filename, context in file_context.items():
        source_code += f"[start of {filename}]\n"
        context = fill_line_number(context)
        source_code += f"{context}"
        source_code += f"\n[end of {filename}]\n"
    source_code.rstrip()
    if source_code.endswith('\n'):
        source_code = source_code[:-1]
    recent_edit_operations = ""
    for i, patch in enumerate(patch_list):
        recent_edit_operations += f"[start of step {i + 1}]\n"
        recent_edit_operations += f"{patch}"
        recent_edit_operations += f"\n[end of step {i + 1}]\n"
    if recent_edit_operations.endswith('\n'):
        recent_edit_operations = recent_edit_operations[:-1]
    prompt = prompt_replace_regex(prompt_style, problem_statement=requirement, code_context=source_code,
                                  historical_code_changes=recent_edit_operations)
    return prompt


def make_inf_bench(dataset_name, save_path, info_pct: Optional[float] = None, include_pr_mes: bool = True,
                include_issue_mes: bool = True, use_bm25: bool = False, topn: int = 10):
    dataset = get_inf_datasets(dataset_name)

    seen_ids = set()
    save_path = save_path.replace(".jsonl", f"_{str(info_pct)}{'_body' if include_pr_mes else ''}{'_issue' if include_pr_mes else ''}{'_bm25' if use_bm25 else ''}{f'_{topn}' if topn else ''}.jsonl")
    mode = "a" if Path(save_path).exists() else "w"

    # print task information
    print(f"================================================================================")
    print(f"Task information:")
    print(f"   - Input dataset: {dataset_name}")
    print(f"   - Output file: {save_path}")
    print(f"   - Total number of tasks: {len(dataset)}")
    print(f"================================================================================")

    if mode == "a":
        with open(save_path, encoding="utf-8", mode="r") as fr:
            for line in fr:
                seen_ids.add(json.loads(line)["instance_id"])
        print(f"   - Number of existing tasks: {len(seen_ids)}")

    processed_count = 0
    skipped_count = 0
    success_count = 0

    with open(save_path, encoding="utf-8", mode=mode) as fw:
        for ins in tqdm(dataset, desc="Processing tasks", unit="tasks"):
            if ins.instance_id in seen_ids:
                skipped_count += 1
                continue
            processed_count += 1
            ins, ok = with_prompt(ins, info_pct, include_pr_mes, include_issue_mes, use_bm25, topn)
            res = validation_inf(ins)
            # if ok and res["returncode"] == "0":
            if res["returncode"] == "0":
                write_json_line(ins, fw, is_instance=True, is_prompt=True)
                success_count += 1
            else:
                print("inf generation ok:", ok, "script return code:", res["returncode"])
                raise Exception("validation error")
    
    # read the number of tasks in the final file
    final_count = 0
    if Path(save_path).exists():
        with open(save_path, encoding="utf-8", mode="r") as fr:
            final_count = sum(1 for _ in fr)
    
    print(f"================================================================================")
    print(f"Task completed:")
    print(f"   - Number of tasks processed: {processed_count}")
    print(f"   - Number of tasks skipped: {skipped_count}")
    print(f"   - Number of tasks successfully processed: {success_count}")
    print(f"   - Total number of tasks in the final file: {final_count}")
    print(f"================================================================================")
    


if __name__ == "__main__":
    # Example:
    #   python -m editbench.inference.prompt_builder \
    #     --dataset-name ./crawled_data/bench/all-task-instances.jsonl \
    #     --save-path ./crawled_data/infbench/all-task-instances.jsonl \
    #     --info-pct 0.2
    #   python -m editbench.inference.prompt_builder \
    #     --dataset-name ./bench/django-django-task-instances.jsonl \
    #     --save-path ./infbench/django-django-task-instances.jsonl \
    #     --info-pct 0.2 --use-bm25 --topn 5 --no-pr-mes --no-issue-mes
    parser = argparse.ArgumentParser(description="Build infbench prompts from bench task instances.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Path to input bench task-instances jsonl.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to output infbench jsonl.",
    )
    parser.add_argument(
        "--info-pct",
        type=float,
        default=None,
        help="Split ratio for pre-edits (e.g. 0.2 = first 20%% as history). Default: None (no split).",
    )
    parser.add_argument(
        "--no-pr-mes",
        action="store_true",
        help="Do not include PR title/body in requirement (default: include).",
    )
    parser.add_argument(
        "--no-issue-mes",
        action="store_true",
        help="Do not include issue message in requirement (default: include).",
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        help="Use BM25 retrieval for code context instead of oracle.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Top-N files for BM25 retrieval (default: 10).",
    )
    args = parser.parse_args()
    make_inf_bench(
        args.dataset_name,
        args.save_path,
        info_pct=args.info_pct,
        include_pr_mes=not args.no_pr_mes,
        include_issue_mes=not args.no_issue_mes,
        use_bm25=args.use_bm25,
        topn=args.topn,
    )
