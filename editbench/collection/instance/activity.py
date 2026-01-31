import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Iterator

from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class Activity:
    """
    A dataclass that represents metadata information for activity instances
    in a code repository, such as pull requests or commits.
    """
    # (repo_owner)__(repo_name)_(src_type)-(id)
    instance_id: str = ""
    # pr: pr number, commit: commit hash
    instance_num: str = ""
    # pull or commit
    src_type: str = ""
    # repo_owner/repo_name
    repo: str = ""
    # the activity's base commit
    base_commit: str = ""
    # the activity's url
    html_url: str = ""
    # the activity's creation time
    created_at: str = ""
    # prompt_mes
    prompt: str = ""
    # the activity's title message(commit message or pr title)
    title_mes: str = ""
    # the activity's body message(pr body)
    body_mes: str = ""
    # the activity's issues message(activity-related issue body)
    issues_mes: str = ""
    # the activity's other message(activity-related issue's hint message)
    other_mes: list = field(default_factory=list)
    # the activity's state (Merged or Closed)
    state: str = ""
    #  the activity's resolved issues
    resolved_issues: list = field(default_factory=list)
    # the activity's all changed files
    files: list = field(default_factory=list)
    #  the activity's changed edit files
    files_work: list = field(default_factory=list)
    #  the activity's changed test files
    files_test: list = field(default_factory=list)
    #  the activity's changed other files (configuration files, etc.)
    files_other: list = field(default_factory=list)
    #  the activity's changed files that are not edited (added or deleted file)
    files_no_edit: list = field(default_factory=list)

    # the activity's work patch
    work_patch: str = ""
    work_patch_list: list = field(default_factory=list)
    num_changes: int = 0
    # the activity's test patch
    test_patch: str = ""

    split_idx: int = 0
    pre_edits: list = field(default_factory=list)
    ground_truth: str = ""

    fail_to_pass: list = field(default_factory=list)
    pass_to_fail: list = field(default_factory=list)
    fail_to_fail: list = field(default_factory=list)
    pass_to_pass: list = field(default_factory=list)

    # the repo's version
    version: str = ""
    # the repo's version commit
    version_commit: str = ""

    @property
    def version_fmt(self):
        """get processed version number (extract x.xx format)"""
        if not self.version:
            return "default"
        four_two_match = re.search(r'\d{4}\.\d{2}', self.version)
        if four_two_match:
            return four_two_match.group()
        match = re.search(r'\d+\.\d+', self.version)
        return match.group() if match else self.version

    @classmethod
    def parse_instance_id(cls, data: dict):
        repo_full_name = re.search(r'https://api\.github\.com/repos/([^/]+/[^/]+)',
                                   data['url']).group(1)
        instance_number = data["sha"] if data["src_type"] == "commit" else data["number"]
        instance_id = f"{repo_full_name.replace('/', '__')}-{data['src_type']}-{instance_number}"
        return instance_id

    @property
    def no_edit_number(self) -> int:
        source_extensions = {'.py', '.java', '.js', '.cpp', '.c', '.go', '.ts', '.php', '.vue'}
        num = 0
        for file_name in self.files_no_edit:
            if any(file_name.lower().endswith(ext) for ext in source_extensions):
                num += 1
        return num

    @property
    def edit_number(self) -> int:
        return len(self.files_work)

    @property
    def test_number(self) -> int:
        return len(self.files_test)

    def is_merged(self):
        return self.state == "merged"

    def has_base_commit(self) -> bool:
        return self.base_commit != "" and self.base_commit is not None

    def is_ft_valid(self):
        return self.no_edit_number == 0 and self.has_base_commit()

    def is_test_valid(self):
        return ((self.no_edit_number == 0 and self.has_base_commit()
                 and self.is_merged()) and self.test_number > 0
                and self.edit_number > 0)

    def to_instance_dict(self, is_prompt: bool = False) -> dict:
        result = {}
        all_fields = fields(self.__class__)
        banned_name = ["src_type", "state", "files", "files_other", "files_no_edit"] if is_prompt \
            else ["src_type", "state", "files", "files_other", "files_no_edit", "prompt", "pre_edits",
                  "ground_truth", "split_idx"]
        for field_info in all_fields:
            field_name = field_info.name
            if field_name in banned_name:
                continue
            value = getattr(self, field_name)
            if is_dataclass(value) and not isinstance(value, type):
                value = asdict(value)
            result[field_name] = value
        return result

    def __getitem__(self, item):
        return getattr(self, item)

    def __contains__(self, item):
        return hasattr(self, item)


FilterFunc = Callable[[Activity], bool]


def filter_by_ft_valid() -> FilterFunc:
    def _filter(instance: Activity) -> bool:
        return instance.is_ft_valid()

    return _filter


def filter_by_test_valid() -> FilterFunc:
    def _filter(instance: Activity) -> bool:
        return instance.is_test_valid()

    return _filter


def filter_by_edit_number(num: int) -> FilterFunc:
    def _filter(instance: Activity) -> bool:
        return instance.edit_number <= num

    return _filter


def filter_by_timestamp(start_time_str: Optional[str] = None,
                        end_time_str: Optional[str] = None) -> FilterFunc:
    """
    :param start_time_str: format: 20240101
    :param end_time_str: format: 20240101
    """
    start_time_utc = None
    if start_time_str is not None:
        try:
            target_date = datetime.strptime(start_time_str, "%Y%m%d").date()
            start_time_utc = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(
                f"Invalid time format '{start_time_str}', please use 'YYYYMMDD' format（e.g. 20240101）") from e

    end_time_utc = None
    if end_time_str is not None:
        try:
            target_date = datetime.strptime(end_time_str, "%Y%m%d").date()
            end_time_utc = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        except ValueError as e:
            raise ValueError(
                f"Invalid time format '{end_time_str}', please use 'YYYYMMDD' format（e.g. 20240101）") from e

    def _filter(instance: Activity) -> bool:
        try:
            instance_time = datetime.strptime(instance.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            condition_start = start_time_utc is None or instance_time >= start_time_utc
            condition_end = end_time_utc is None or instance_time <= end_time_utc
            return condition_start and condition_end
        except (ValueError, TypeError):
            return False

    return _filter


def write_json_line(activity, file, is_instance=False, is_prompt=False) -> None:
    """Serialize ActivityInstance to JSON and write to file (with newline)"""
    try:
        if is_instance:
            json_line = json.dumps(activity.to_instance_dict(is_prompt))
        else:
            json_line = json.dumps(asdict(activity))
        print(json_line, end="\n", file=file, flush=True)
    except (TypeError, IOError) as e:
        logger.error(f"Failed to write {activity.instance_id}: {str(e)}")


def load_datasets_from_jsonl(file_path: str) -> Iterator[Activity]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            seen_ids = set()
            for i, ins in enumerate(f):
                line_number = i + 1
                try:
                    act_json = json.loads(ins)
                    act = Activity(**act_json)
                    if act.instance_id in seen_ids:
                        # print(f"⚠️ the {line_number}-th line has duplicate id: {act.instance_id}")
                        continue
                    seen_ids.add(act.instance_id)
                    yield act
                except json.JSONDecodeError as e:
                    pass
                except Exception as e:
                    # print(f"⚠️ the {line_number}-th line has unknown error: {str(e)}")
                    pass
    except FileNotFoundError:
        print(f"Error: file {file_path} has not found.")
    except Exception as e:
        print(f"The process of reading file {file_path} has unknown error: {str(e)}")
