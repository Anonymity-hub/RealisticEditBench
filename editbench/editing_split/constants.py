import os
from pathlib import Path

current_file_path = __file__
current_absolute_path = os.path.abspath(current_file_path)
root_path_obj = Path(current_absolute_path).parent.parent.parent

# Constants - Editing_split
EDITING_SPLIT_DIR = root_path_obj / "patch_histories"
REPO_AND_LOG_DIR = root_path_obj / "tmp"
