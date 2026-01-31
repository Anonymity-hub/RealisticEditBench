import os
from pathlib import Path

current_file_path = __file__
current_absolute_path = os.path.abspath(current_file_path)
root_path_obj = Path(current_absolute_path).parent.parent.parent

# Constants
EXPERIMENTAL_RESULTS = root_path_obj / "experiment_results"
