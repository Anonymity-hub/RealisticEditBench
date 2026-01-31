from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_RAW = os.getenv('GITHUB_RAW')
GITHUB_RAW_PROXY = os.getenv('GITHUB_RAW_PROXY')
GITHUB_ORI = os.getenv('GITHUB_ORI')
GITHUB_ORI_PROXY = os.getenv('GITHUB_ORI_PROXY')
API_GITHUB = os.getenv('API_GITHUB')
GITHUB_TOKEN = os.getenv('GITHUB_TOKENS').split(',')[0]



current_file_path = __file__
current_absolute_path = os.path.abspath(current_file_path)
root_path_obj = Path(current_absolute_path).parent.parent.parent



# Constants - DATA
SRC_DATA = root_path_obj / "crawled_data"
SRC_DATA.mkdir(parents=True, exist_ok=True)
SRC_EXECUTION_FILTER_DATA = SRC_DATA / "execution_filter"
SRC_EXECUTION_FILTER_DATA.mkdir(parents=True, exist_ok=True)
SRC_ACTIVITY_DATA = SRC_DATA / "activity"
SRC_ACTIVITY_DATA.mkdir(parents=True, exist_ok=True)
SRC_BENCHMARK_DATA = SRC_DATA / "bench"
SRC_BENCHMARK_DATA.mkdir(parents=True, exist_ok=True)
SRC_INF_BENCHMARK_DATA = SRC_DATA / "infbench"
SRC_INF_BENCHMARK_DATA.mkdir(parents=True, exist_ok=True)

# Constants - Collection
TAG_VERSION = SRC_DATA / "tag_version"

# Constants - LOG
LOG_DIR = root_path_obj / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_EXECUTION_FILTER = LOG_DIR / "execution_filter"
LOG_EXECUTION_FILTER.mkdir(parents=True, exist_ok=True)

# Experimental results

SRC_EXPERIMENTS = root_path_obj / "experiment_results"
SRC_EXPERIMENTS.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(GITHUB_TOKEN)
