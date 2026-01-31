from enum import Enum
from pathlib import Path
import os
from pathlib import Path

current_file_path = __file__
current_absolute_path = os.path.abspath(current_file_path)
root_path_obj = Path(current_absolute_path).parent.parent.parent


TEST_PYTEST = "pytest -rA"
TEST_PYTEST_SCILEARN = "pytest -rA"
TEST_PYTEST_MATPLOTLIB = "pytest -rA"
TEST_ASTROPY_PYTEST = "pytest -rA -vv -o console_output_style=classic --tb=no --run-slow"
TEST_DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
TEST_DJANGO_NO_PARALLEL = "./tests/runtests.py --verbosity 2"
TEST_SPHINX = "tox --current-env -epy39 -v --"
TEST_SPHINX_311 = "tox --current-env -epy311 -v --"

# Constants - Evaluation Log Directories
BASE_IMAGE_BUILD_DIR = root_path_obj / "logs/build_images/base"
ENV_IMAGE_BUILD_DIR = root_path_obj / "logs/build_images/env"
INSTANCE_IMAGE_BUILD_DIR = root_path_obj / "logs/build_images/instances"
RUN_EVALUATION_LOG_DIR = root_path_obj / "logs/run_evaluation"
EXECUTION_FILTER_LOG_DIR = root_path_obj / "logs/execution_filter"


# Constants - Evaluation
DOCKER_PATCH = "/tmp/patch.diff"
DOCKER_USER = "root"
DOCKER_WORKDIR = "/testbed"
LOG_REPORT = "report.json"
LOG_INSTANCE = "run_instance.log"
LOG_TEST_OUTPUT = "test_output.txt"
LOG_TEST_OUTPUT_BEFORE = "test_output_before.txt"
LOG_TEST_OUTPUT_AFTER = "test_output_after.txt"
UTF8 = "utf-8"

# Constants - Evaluation Keys
KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name"
KEY_PREDICTION = "model_patch"

TEST_SYMPY = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"


class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


# Constants - Miscellaneous
NON_TEST_EXTS = [
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
]

# Constants - Installation Specifications
SPECS_SKLEARN = {
    k: {
        "python": "3.6",
        "packages": "numpy scipy cython pytest pandas matplotlib",
        "install": "python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
        "pip_packages": [
            "cython",
            "numpy==1.19.2",
            "setuptools",
            "scipy==1.5.2",
        ],
        "test_cmd": TEST_PYTEST_SCILEARN,
    }
    for k in ["0.20", "0.21", "0.22"]
}
SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.9",
            "packages": "'numpy==1.19.2' 'scipy==1.5.2' 'cython==3.0.10' pytest 'pandas<2.0.0' 'matplotlib<3.9.0' setuptools pytest joblib threadpoolctl",
            # "install": "python -m pip install -v --no-use-pep517 --no-build-isolation -e .",
            "install": "python -m pip install -v -e .",
            "pip_packages": ["cython", "setuptools", "numpy", "scipy"],
            "after_install": [
                "meson --version",
                "cd /testbed && rm -rf build/cp311 && mkdir -p build/cp311",
                'export PATH="/opt/miniconda3/envs/testbed/bin:$PATH"',
                "meson setup build/cp311 && cd build/cp311 && ninja && cd /testbed",
            ],
            "test_cmd": TEST_PYTEST_SCILEARN,
            "eval_with_install": False
        }
        for k in ["1.3", "1.4", ]
    }
)

SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.11",
            "packages": "'numpy==2.0.0' 'scipy>1.8.0' 'cython==3.1.2' 'ninja' 'meson' pytest 'pandas>=1.4.0' 'matplotlib>=3.5.0' setuptools pytest joblib threadpoolctl",
            "install": "python -m pip install -v -e .",
            "pip_packages": ["cython==3.1.2", "setuptools", "numpy==2.0.0", "scipy"],
            "test_cmd": TEST_PYTEST_SCILEARN,
            "after_install": [
                "meson --version",
                "cd /testbed && rm -rf build/cp311 && mkdir -p build/cp311",
                'export PATH="/opt/miniconda3/envs/testbed/bin:$PATH"',
                "meson setup build/cp311 && cd build/cp311 && ninja && cd /testbed",
            ],
            "eval_with_install": False
        }
        for k in ["1.5", "1.6", "1.7", "default"]
    }
)

SPECS_DJANGO = {
    k: {
        "python": "3.5",
        "packages": "requirements.txt",
        "pre_install": [
            "apt-get update && apt-get install -y locales",
            "echo 'en_US UTF-8' > /etc/locale.gen",
            "locale-gen en_US.UTF-8",
        ],
        "install": "python setup.py install",
        "pip_packages": ["setuptools"],
        "eval_commands": [
            "export LANG=en_US.UTF-8",
            "export LC_ALL=en_US.UTF-8",
            "export PYTHONIOENCODING=utf8",
            "export LANGUAGE=en_US:en",
        ],
        "test_cmd": TEST_DJANGO,
    }
    for k in ["1.7", "1.8", "1.9", "1.10", "1.11", "2.0", "2.1", "2.2"]
}
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.5",
            "install": "python setup.py install",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["1.4", "1.5", "1.6"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.6",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "eval_commands": [
                "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
                "export LANG=en_US.UTF-8",
                "export LANGUAGE=en_US:en",
                "export LC_ALL=en_US.UTF-8",
            ],
            "test_cmd": TEST_DJANGO,
        }
        for k in ["3.0", "3.1", "3.2"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.8",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["4.0"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.9",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["4.1", "4.2"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.11",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["5.0", "5.1", "5.2"]
    }
)

SPECS_DJANGO.update(
    {
        k: {
            "python": "3.12",
            "pre_packages": """apt update && \
                        apt install -y \
                          libmemcached-dev \
                          gcc \
                          build-essential && \
                        apt clean && \
                        rm -rf /var/lib/apt/lists/*""",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["default"]
    }
)

SPECS_DJANGO["1.9"]["test_cmd"] = TEST_DJANGO_NO_PARALLEL

SPECS_ASTROPY = {
    k: {
        "python": "3.9",
        "install": "python -m pip install -e .[test] --verbose",
        "pip_packages": [
            "attrs==23.1.0",
            "exceptiongroup==1.1.3",
            "execnet==2.0.2",
            "hypothesis==6.82.6",
            "iniconfig==2.0.0",
            "numpy==1.25.2",
            "packaging==23.1",
            "pluggy==1.3.0",
            "psutil==5.9.5",
            "pyerfa==2.0.0.3",
            "pytest-arraydiff==0.5.0",
            "pytest-astropy-header==0.2.2",
            "pytest-astropy==0.10.0",
            "pytest-cov==4.1.0",
            "pytest-doctestplus==1.0.0",
            "pytest-filter-subpackage==0.1.2",
            "pytest-mock==3.11.1",
            "pytest-openfiles==0.5.0",
            "pytest-remotedata==0.4.0",
            "pytest-xdist==3.3.1",
            "pytest==7.4.0",
            "PyYAML==6.0.1",
            "setuptools==68.0.0",
            "sortedcontainers==2.4.0",
            "tomli==2.0.1",
        ],
        "test_cmd": TEST_PYTEST
    }
    for k in ["3.0", "3.1", "3.2", "4.1", "4.2", "4.3", "5.0", "5.1", "5.2", "5.3", "default"]
}

SPECS_ASTROPY.update(
    {
        k: {
            "python": "3.11",
            "install": "python -m pip install -e .[test] --verbose",
            "pip_packages": [
                "attrs==23.1.0",
                "exceptiongroup==1.1.3",
                "execnet==2.0.2",
                "hypothesis==6.135.1",
                "iniconfig==2.0.0",
                "numpy==2.3",
                "packaging==23.1",
                "pluggy==1.3.0",
                "psutil==5.9.5",
                "pyerfa==2.0.0.3",
                "pytest-arraydiff==0.5.0",
                "pytest-astropy-header==0.2.2",
                "pytest-astropy==0.10.0",
                "pytest-cov==4.1.0",
                "pytest-doctestplus==1.4.0",
                "pytest-filter-subpackage==0.1.2",
                "pytest-mock==3.11.1",
                "pytest-openfiles==0.5.0",
                "pytest-remotedata==0.4.0",
                "pytest-xdist==3.3.1",
                "pytest==7.4.0",
                "PyYAML==6.0.1",
                "setuptools==68.0.0",
                "sortedcontainers==2.4.0",
                "tomli==2.0.1",
            ],
            "test_cmd": TEST_PYTEST
        }
        for k in ["7.1", "7.0", "6.1"]
    }
)

SPECS_ASTROPY.update(
    {
        k: {
            "python": "3.11",
            "install": "python -m pip install -e .[test] --verbose",
            "pip_packages": [
                "attrs==23.1.0",
                "exceptiongroup==1.1.3",
                "execnet==2.0.2",
                "hypothesis==6.135.1",
                "iniconfig==2.0.0",
                "numpy==1.26.4",
                "packaging==23.1",
                "pluggy==1.3.0",
                "psutil==5.9.5",
                "pyerfa==2.0.0.3",
                "pytest-arraydiff==0.5.0",
                "pytest-astropy-header==0.2.2",
                "pytest-astropy==0.10.0",
                "pytest-cov==4.1.0",
                "pytest-doctestplus==1.4.0",
                "pytest-filter-subpackage==0.1.2",
                "pytest-mock==3.11.1",
                "pytest-openfiles==0.5.0",
                "pytest-remotedata==0.4.0",
                "pytest-xdist==3.3.1",
                "pytest==7.4.0",
                "PyYAML==6.0.1",
                "setuptools==68.0.0",
                "sortedcontainers==2.4.0",
                "tomli==2.0.1",
                "matplotlib==3.5.3",
                "pandas"
            ],
            "test_cmd": TEST_PYTEST
        }
        for k in ["6.0"]
    }
)

SPECS_ASTROPY.update(
    {
        k: {
            "python": "3.6",
            "install": "python -m pip install -e .[test] --verbose",
            "packages": "setuptools==38.2.4",
            "pip_packages": [
                "attrs==17.3.0",
                "exceptiongroup==0.0.0a0",
                "execnet==1.5.0",
                "hypothesis==3.44.2",
                "cython==0.27.3",
                "jinja2==2.10",
                "MarkupSafe==1.0",
                "numpy==1.16.0",
                "packaging==16.8",
                "pluggy==0.6.0",
                "psutil==5.4.2",
                "pyerfa==1.7.0",
                "pytest-arraydiff==0.1",
                "pytest-astropy-header==0.1",
                "pytest-astropy==0.2.1",
                "pytest-cov==2.5.1",
                "pytest-doctestplus==0.1.2",
                "pytest-filter-subpackage==0.1",
                "pytest-forked==0.2",
                "pytest-mock==1.6.3",
                "pytest-openfiles==0.2.0",
                "pytest-remotedata==0.2.0",
                "pytest-xdist==1.20.1",
                "pytest==3.3.1",
                "PyYAML==3.12",
                "sortedcontainers==1.5.9",
                "tomli==0.2.0",
            ],
            "test_cmd": TEST_ASTROPY_PYTEST,
        }
        for k in ["0.1", "0.2", "0.3", "0.4", "1.1", "1.2", "1.3"]
    }
)

for k in ["4.1", "4.2", "4.3", "5.0", "5.1", "5.2", "5.3", "6.0", "6.1", "7.0", "7.1", "default"]:
    SPECS_ASTROPY[k]["pre_install"] = [
        'sed -i \'s/requires = \\["setuptools",/requires = \\["setuptools==68.0.0",/\' pyproject.toml'
    ]

for k in ["6.0", "6.1", "default", "7.0", "7.1"]:
    SPECS_ASTROPY[k]["pre_install"] = [
        """
    if grep -rlq "relative_to(Path.cwd())" /testbed/astropy/; then
        echo "Found files with 'relative_to(Path.cwd())', modifying..."
        grep -rl "relative_to(Path.cwd())" /testbed/astropy/ | xargs sed -i "s/relative_to(Path.cwd())/resolve()/"
    else
        echo "No files found with 'relative_to(Path.cwd())', skipping modification."
    fi
        """,
        'sed -i "s/sources=\[str(s) for s in sources\]/sources=\[str(source)\]/" /testbed/astropy/table/setup_package.py'
    ]
    # SPECS_ASTROPY[k]["after_install"] = [
    #     'grep -rl "resolve()" /testbed/astropy/ | xargs sed -i "s/resolve()/relative_to(Path.cwd())/"'
    # ]

for k in ["5.3"]:
    SPECS_ASTROPY[k]["python"] = "3.10"

# hypothesis = 6.135.1
# pytest-doctestplus-1.4.0
# for k in ["default"]:
#     SPECS_ASTROPY[k]["python"] = "3.11"

SPECS_SYMPY = {
    k: {
        "python": "3.9",
        "packages": "mpmath flake8",
        "pip_packages": ["mpmath==1.3.0", "flake8-comprehensions"],
        "install": "python -m pip install -e .",
        "test_cmd": TEST_SYMPY,
    }
    for k in ["0.7", "1.0", "1.1", "1.10", "1.11", "1.12", "1.2", "1.4", "1.5", "1.6"]
             + ["1.7", "1.8", "1.9"] + ["1.10", "1.11", "1.12", "1.13", "1.14"]
}
SPECS_SYMPY.update(
    {
        k: {
            "python": "3.9",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pip_packages": ["mpmath==1.3.0"],
            "test_cmd": TEST_SYMPY,
        }
        for k in ["1.13", "1.14"]
    }
)

SPECS_MATPLOTLIB = {
    k: {
        "python": "3.11",
        "packages": "environment.yml",
        "install": "python -m pip install -e .",
        # "pre_install": [
        #     "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super dvipng",
        #     "QHULL_URL=\"http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz\"",
        #     "QHULL_TAR=\"/tmp/qhull-2020-src-8.0.2.tgz\"",
        #     "QHULL_BUILD_DIR=\"/testbed/build\"",
        #     "wget -O \"$QHULL_TAR\" \"$QHULL_URL\"",
        #     "mkdir -p \"$QHULL_BUILD_DIR\"",
        #     "tar -xvzf \"$QHULL_TAR\" -C \"$QHULL_BUILD_DIR\""
        # ],
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && (DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common || true) && (add-apt-repository universe || true) && apt-get -y update && (DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super dvipng || DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex cm-super dvipng)",
            # "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super dvipng",
            # "QHULL_URL=\"http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz\"",
            # "QHULL_TAR=\"/tmp/qhull-2020-src-8.0.2.tgz\"",
            # "QHULL_BUILD_DIR=\"/testbed/build\"",
            # "QHULL_EXTRACT_DIR=\"/testbed/extern/qhull\"",
            # "wget -O \"$QHULL_TAR\" \"$QHULL_URL\" --timeout=30 --tries=5 --no-check-certificate",
            # "mkdir -p \"$QHULL_EXTRACT_DIR\"",
            # "tar -xvzf \"$QHULL_TAR\" -C \"$QHULL_EXTRACT_DIR\" --strip-components=1",
            # "echo '[wrap-static]' > /testbed/extern/qhull.wrap",
            # "echo 'directory = qhull' >> /testbed/extern/qhull.wrap",
            # "echo 'provide = qhull' >> /testbed/extern/qhull.wrap"
            # 2. Qhull configuration (use qhull.org source, increase network tolerance)
            "QHULL_URL=\"http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz\"",
            "QHULL_TAR=\"/tmp/qhull-2020-src-8.0.2.tgz\"",
            "QHULL_EXTRACT_DIR=\"/testbed/extern/qhull\"",
            # backup source: if the official website times out, try domestic mirror (such as Tsinghua University mirror)
            "QHULL_FALLBACK_URL=\"https://mirrors.ustc.edu.cn/qhull/qhull-2020-src-8.0.2.tgz\"",

            # 3. multi-source download: first try the official website, if failed, use the backup source
            "if ! wget -O \"$QHULL_TAR\" \"$QHULL_URL\" --timeout=30 --tries=3 --no-check-certificate; then",
            "    echo \"Qhull official website download failed, trying backup source...\";",
            "    if ! wget -O \"$QHULL_TAR\" \"$QHULL_FALLBACK_URL\" --timeout=30 --tries=3 --no-check-certificate; then",
            "        echo \"ERROR: Qhull all sources download failed, please check the network\";",
            "        exit 1;",
            "    fi;",
            "fi;",

            # 4. decompression adaptation: the internal directory of qhull-2020-src-8.0.2.tgz is "qhull-2020.2/", need to adjust --strip-components
            "mkdir -p \"$QHULL_EXTRACT_DIR\"",
            # the first layer directory of the original compressed package is "qhull-2020.2", need to strip this layer directory (--strip-components=1)
            "tar -xvzf \"$QHULL_TAR\" -C \"$QHULL_EXTRACT_DIR\" --strip-components=1 || { echo \"Qhull decompression failed\"; exit 1; }",
            "rm -f \"$QHULL_TAR\"",

            # 5. force meson to use local Qhull (important: overwrite wrap configuration)
            "echo '[wrap-static]' > /testbed/extern/qhull.wrap",
            "echo 'directory = qhull' >> /testbed/extern/qhull.wrap",
            "echo 'provide = qhull' >> /testbed/extern/qhull.wrap",

            # 6. Freetype mirror configuration (avoid other dependencies timing out)
            "FREETYPE_WRAP=\"/testbed/extern/freetype.wrap\"",
            "if [ -f \"$FREETYPE_WRAP\" ]; then",
            "    sed -i 's|https://download.savannah.gnu.org/releases/freetype|https://mirrors.tuna.tsinghua.edu.cn/gnu/freetype|g' \"$FREETYPE_WRAP\"",
            "    echo 'source_fallback_url = https://mirrors.aliyun.com/gnu/freetype/freetype-old/freetype-2.6.1.tar.gz' >> \"$FREETYPE_WRAP\"",
            "fi"
        ],
        "pip_packages": [
            "contourpy==1.1.0",
            "cycler==0.11.0",
            "fonttools==4.42.1",
            "ghostscript",
            "kiwisolver==1.4.5",
            "numpy==1.25.2",
            "packaging==23.1",
            "pillow==10.0.0",
            "pikepdf",
            "pyparsing==3.0.9",
            "python-dateutil==2.8.2",
            "six==1.16.0",
            "setuptools==68.1.2",
            "setuptools-scm==7.1.0",
            "typing-extensions==4.7.1",
        ],
        "test_cmd": TEST_PYTEST_MATPLOTLIB,
    }
    for k in ["3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "default"]
}
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.8",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg libfreetype6-dev pkg-config texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super",
                "QHULL_URL=\"http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz\"",
                "QHULL_TAR=\"/tmp/qhull-2020-src-8.0.2.tgz\"",
                "QHULL_BUILD_DIR=\"/testbed/build\"",
                "wget -O \"$QHULL_TAR\" \"$QHULL_URL\"",
                "mkdir -p \"$QHULL_BUILD_DIR\"",
                "tar -xvzf \"$QHULL_TAR\" -C \"$QHULL_BUILD_DIR\""
            ],
            "pip_packages": ["pytest", "ipython"],
            "test_cmd": TEST_PYTEST_MATPLOTLIB,
        }
        for k in ["3.1", "3.2", "3.3", "3.4"]
    }
)
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.7",
            "packages": "requirements.txt",
            "install": "python -m pip install -e .",
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && apt-get install -y imagemagick ffmpeg libfreetype6-dev pkg-config",
                "QHULL_URL=\"http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz\"",
                "QHULL_TAR=\"/tmp/qhull-2020-src-8.0.2.tgz\"",
                "QHULL_BUILD_DIR=\"/testbed/build\"",
                "wget -O \"$QHULL_TAR\" \"$QHULL_URL\"",
                "mkdir -p \"$QHULL_BUILD_DIR\"",
                "tar -xvzf \"$QHULL_TAR\" -C \"$QHULL_BUILD_DIR\""
            ],
            "pip_packages": ["pytest"],
            "test_cmd": TEST_PYTEST_MATPLOTLIB,
        }
        for k in ["3.0"]
    }
)
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.5",
            "install": "python setup.py build; python setup.py install",
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && && apt-get install -y imagemagick ffmpeg"
            ],
            "pip_packages": ["pytest"],
            "execute_test_as_nonroot": True,
            "test_cmd": TEST_PYTEST_MATPLOTLIB,
        }
        for k in ["2.0", "2.1", "2.2", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
    }
)
# for k in ["3.8", "3.9"]:
#     SPECS_MATPLOTLIB[k]["install"] = (
#         'python -m pip install --no-build-isolation -e ".[dev]"'
#     )

SPECS_XARRAY = {
    k: {
        "python": "3.10",
        "packages": "environment.yml",
        "install": "python -m pip install -e .",
        "pip_packages": [
            "numpy==1.23.0",
            "packaging==23.1",
            "pandas==1.5.3",
            "pytest==7.4.0",
            "python-dateutil==2.8.2",
            "pytz==2023.3",
            "six==1.16.0",
            "scipy==1.11.1",
            "setuptools==68.0.0",
            "dask==2022.8.1",
        ],
        "no_use_env": True,
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "0.12",
        "0.18",
        "0.19",
        "0.20",
        "2022.03",
        "2022.06",
        "2022.09",
        "2023.07",
        "2024.05",
    ]
}

SPECS_XARRAY.update(
    {
        k: {
            "python": "3.11",
            "packages": "environment.yml",
            "install": "python -m pip install -e .",
            "pip_packages": [
                "numpy==1.26.0",
                "packaging==24.1",
                "pandas==2.2",
                "pytest==7.4.0",
                "python-dateutil==2.8.2",
                "pytz==2023.3",
                "six==1.16.0",
                "scipy==1.13",
                "setuptools==77.0.3",
                "dask==2022.8.1",
            ],
            "no_use_env": True,
            "test_cmd": TEST_PYTEST,
        }
        for k in [
        "2024.9",
        "2024.10",
        "2024.11",
        "2024.12",
        "2025.01",
        "2025.02",
        "2025.03",
        "2025.04",
        "2025.05",
        "2025.06",
        "2025.07",
        "default"
    ]
    })

SPECS_SPHINX = {
    k: {
        "python": "3.9",
        "pip_packages": ["tox==4.16.0", "tox-current-env==0.0.11", "Jinja2==3.0.3", "pytest>=8.0", "pytest-xdist[psutil]>=3.4", "cython>=3.0", "defusedxml>=0.7.1", "typing_extensions>=4.9"],
        "install": "python -m pip install -e .[test]",
        "pre_install": ["sed -i 's/pytest/pytest -rA/' tox.ini",
                        # "sed -i '/\[project\]/a [project.optional-dependencies]' pyproject.toml",
                        # "sed -n '/\[dependency-groups\]/,/test = \[/ { /test = \[/ {p; :a; n; /\]/!ba; p; q } }' pyproject.toml | sed 's/\[dependency-groups\]/\[project.optional-dependencies\]/' >> pyproject.toml",
                        # "sed -i '/test = \[/,/\]/ s/^/#/' pyproject.toml"
                        ],
        "test_cmd": TEST_SPHINX,
    }
    for k in ["1.5", "1.6", "1.7", "1.8", "2.0", "2.1", "2.2", "2.3", "2.4", "3.0"]
             + ["3.1", "3.2", "3.3", "3.4", "3.5", "4.0", "4.1", "4.2", "4.3", "4.4"]
             + ["4.5", "5.0", "5.1", "5.2", "5.3", "6.0", "6.2", "7.0", "7.1", "7.2"]
             + ["7.3", "7.4", "8.0", "8.1", "8.2", "default"]
}
for k in ["3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "4.0", "4.1", "4.2", "4.3", "4.4"]:
    SPECS_SPHINX[k]["pre_install"].extend(
        [
            "sed -i 's/Jinja2>=2.3/Jinja2<3.0/' setup.py",
            "sed -i 's/sphinxcontrib-applehelp/sphinxcontrib-applehelp<=1.0.7/' setup.py",
            "sed -i 's/sphinxcontrib-devhelp/sphinxcontrib-devhelp<=1.0.5/' setup.py",
            "sed -i 's/sphinxcontrib-qthelp/sphinxcontrib-qthelp<=1.0.6/' setup.py",
            "sed -i 's/alabaster>=0.7,<0.8/alabaster>=0.7,<0.7.12/' setup.py",
            "sed -i \"s/'packaging',/'packaging', 'markupsafe<=2.0.1',/\" setup.py",
        ]
    )
    if k in ["4.2", "4.3", "4.4"]:
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                "sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py",
                "sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py",
            ]
        )
    elif k == "4.1":
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                (
                    "grep -q 'sphinxcontrib-htmlhelp>=2.0.0' setup.py && "
                    "sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py || "
                    "sed -i 's/sphinxcontrib-htmlhelp/sphinxcontrib-htmlhelp<=2.0.4/' setup.py"
                ),
                (
                    "grep -q 'sphinxcontrib-serializinghtml>=1.1.5' setup.py && "
                    "sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py || "
                    "sed -i 's/sphinxcontrib-serializinghtml/sphinxcontrib-serializinghtml<=1.1.9/' setup.py"
                ),
            ]
        )
    else:
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                "sed -i 's/sphinxcontrib-htmlhelp/sphinxcontrib-htmlhelp<=2.0.4/' setup.py",
                "sed -i 's/sphinxcontrib-serializinghtml/sphinxcontrib-serializinghtml<=1.1.9/' setup.py",
            ]
        )
for k in ["7.2", "7.3", "7.4", "8.0", "8.1", "8.2", "default"]:
    SPECS_SPHINX[k]["pre_install"] += ["apt-get update && apt-get install -y graphviz"]
# for k in ["8.0"]:
#     SPECS_SPHINX[k]["python"] = "3.10"

for k in ["8.0", "8.1", "8.2", "default"]:
    SPECS_SPHINX[k]["python"] = "3.11"
    SPECS_SPHINX[k]["test_cmd"] = TEST_SPHINX_311


SPECS_PYLINT = {
    k: {
        "python": "3.11",
        "packages": "requirements.txt",
        "install": "python -m pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "2.10",
        "2.11",
        "2.13",
        "2.14",
        "2.15",
        "2.16",
        "2.17",
        "2.8",
        "2.9",
        "3.0",
        "3.1",
        "3.2",
        "3.3",
        "4.0",
    ]
}
SPECS_PYLINT["2.8"]["pip_packages"] = ["pyenchant==3.2"]
SPECS_PYLINT["2.8"]["pre_install"] = [
    "apt-get update && apt-get install -y libenchant-2-dev hunspell-en-us"
]
SPECS_PYLINT.update(
    {
        k: {
            **SPECS_PYLINT[k],
            "pip_packages": ["astroid==3.0.0a6", "setuptools"],
            # "pip_packages": ["setuptools"],
        }
        for k in ["3.0", "3.1", "3.2", "3.3", "4.0"]
    }
)

for v in ["2.14", "2.15", "2.17", "3.0", "3.1", "3.2", "3.3", "4.0"]:
    SPECS_PYLINT[v]["nano_cpus"] = int(2e9)

for v in ["3.0", "3.1", "3.2", "3.3", "4.0"]:
    SPECS_PYLINT[v]["after_install"] = ["pip install astroid==3.3.8"]
    SPECS_PYLINT[v][""] = ["pip install astroid==3.3.8"]
    SPECS_PYLINT[v]["eval_with_install"] = False



# Constants - Test Types, Statuses, Commands
FAIL_TO_PASS = "FAIL_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_PASS = "PASS_TO_PASS"
PASS_TO_FAIL = "PASS_TO_FAIL"

# Constants - Logging
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
INSTALL_FAIL = ">>>>> Init Failed"
INSTALL_PASS = ">>>>> Init Succeeded"
INSTALL_TIMEOUT = ">>>>> Init Timed Out"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_FAILED = ">>>>> Some Tests Failed"
TESTS_PASSED = ">>>>> All Tests Passed"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"

MAP_INSTALLED_REPO = {
    "astropy/astropy": f"{Path(__file__).parent.parent.parent}/tmp/beds/astropy",
    "sympy/sympy": f"{Path(__file__).parent.parent.parent}/tmp/beds/sympy",
    "django/django": f"{Path(__file__).parent.parent.parent}/tmp/beds/django",
    "matplotlib/matplotlib": f"{Path(__file__).parent.parent.parent}/tmp/beds/matplotlib",
    "scikit-learn/scikit-learn": f"{Path(__file__).parent.parent.parent}/tmp/beds/scikit-learn",
    "pydata/xarray": f"{Path(__file__).parent.parent.parent}/tmp/beds/xarray",
    "sphinx-doc/sphinx": f"{Path(__file__).parent.parent.parent}/tmp/beds/sphinx",
    "pylint-dev/pylint": f"{Path(__file__).parent.parent.parent}/tmp/beds/pylint",
}

# Constants - Map task Instance (Repository) to Installation Environment
MAP_REPO_VERSION_TO_SPECS = {
    "astropy/astropy": SPECS_ASTROPY,
    "django/django": SPECS_DJANGO,
    "matplotlib/matplotlib": SPECS_MATPLOTLIB,
    "pydata/xarray": SPECS_XARRAY,
    "pylint-dev/pylint": SPECS_PYLINT,
    "scikit-learn/scikit-learn": SPECS_SKLEARN,
    "sphinx-doc/sphinx": SPECS_SPHINX,
    "sympy/sympy": SPECS_SYMPY,
}

# Constants - Repository Specific Installation Instructions
MAP_REPO_TO_INSTALL = {}

# Constants - Task Instance Requirements File Paths
MAP_REPO_TO_REQS_PATHS = {
    "dbt-labs/dbt-core": ["dev-requirements.txt", "dev_requirements.txt"],
    "django/django": ["tests/requirements/py3.txt"],
    "matplotlib/matplotlib": [
        "requirements/dev/dev-requirements.txt",
        "requirements/testing/travis_all.txt",
    ],
    "pallets/flask": ["requirements/dev.txt"],
    "pylint-dev/pylint": ["requirements_test.txt"],
    "pyvista/pyvista": ["requirements_test.txt", "requirements.txt"],
    "sqlfluff/sqlfluff": ["requirements_dev.txt"],
    "sympy/sympy": ["requirements-dev.txt", "requirements-test.txt"],
}

# Constants - Task Instance environment.yml File Paths
MAP_REPO_TO_ENV_YML_PATHS = {
    "matplotlib/matplotlib": ["environment.yml"],
    "pydata/xarray": ["ci/requirements/environment.yml", "environment.yml"],
}

PIP_ASTROPY = {
    'astropy__astropy-pull-17749': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-17499': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-17444': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-17175': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-17020': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-17006': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16985': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16938': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16688': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16686': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16685': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16662': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16438': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16406': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16347': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-16293': ['matplotlib==3.5.3'],
    'astropy__astropy-pull-17296': ['dask'],
    'astropy__astropy-pull-17273': ['dask'],
    'astropy__astropy-pull-17041': ['dask'],
    'astropy__astropy-pull-16999': ['scipy'],
    'astropy__astropy-pull-16290': ['scipy'],
    'astropy__astropy-pull-16280': ['scipy'],
    'astropy__astropy-pull-16259': ['scipy'],
    'astropy__astropy-pull-16786': ['pyarrow'],
    'astropy__astropy-pull-16785': ['pyarrow'],
    'astropy__astropy-pull-16255': ['pyarrow'],
    'astropy__astropy-pull-16237': ['pyarrow'],
}

MAP_SPECS_ENV_INSTANCE_PIP = {
    "astropy/astropy": PIP_ASTROPY,
}

PRE_INSTALL_ASTROPY = {
    'astropy__astropy-pull-18627': [
        "python -m pip install --force-reinstall --no-deps numpy==1.26.4",
        """
python3 << 'PYEOF'
import sys

file_path = "/testbed/astropy/units/quantity_helper/function_helpers.py"
try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # find and replace the line "helps = getattr(module, f.__name__)"
    old_line = "helps = getattr(module, f.__name__)"
    if old_line in content:
        lines = content.split('\\n')
        new_lines = []
        for i, line in enumerate(lines):
            if old_line in line:
                indent = len(line) - len(line.lstrip())
                # replace with the version with exception handling
                new_lines.append(' ' * indent + "try:")
                new_lines.append(' ' * (indent + 4) + "helps = getattr(module, f.__name__)")
                new_lines.append(' ' * indent + "except AttributeError:")
                new_lines.append(' ' * (indent + 4) + "# numpy 2.0+ removed in1d, skip this function")
                new_lines.append(' ' * (indent + 4) + "if f.__name__ == 'in1d':")
                new_lines.append(' ' * (indent + 8) + "helps = None")
                new_lines.append(' ' * (indent + 4) + "else:")
                new_lines.append(' ' * (indent + 8) + "raise")
            else:
                new_lines.append(line)
        
        content = '\\n'.join(new_lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed numpy.in1d compatibility in {file_path}")
    else:
        print(f"Warning: Could not find '{old_line}' in {file_path}")
except Exception as e:
    print(f"Warning: Could not fix numpy.in1d compatibility: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(0)  # non-fatal error, continue installation
PYEOF
        """,
    ],
    'astropy__astropy-pull-18614': [
        "python -m pip install --force-reinstall --no-deps numpy==1.26.4",
        """
python3 << 'PYEOF'
import sys

file_path = "/testbed/astropy/units/quantity_helper/function_helpers.py"
try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # find and replace the line "helps = getattr(module, f.__name__)"
    old_line = "helps = getattr(module, f.__name__)"
    if old_line in content:
        lines = content.split('\\n')
        new_lines = []
        for i, line in enumerate(lines):
            if old_line in line:
                indent = len(line) - len(line.lstrip())
                # replace with the version with exception handling
                new_lines.append(' ' * indent + "try:")
                new_lines.append(' ' * (indent + 4) + "helps = getattr(module, f.__name__)")
                new_lines.append(' ' * indent + "except AttributeError:")
                new_lines.append(' ' * (indent + 4) + "# numpy 2.0+ removed in1d, skip this function")
                new_lines.append(' ' * (indent + 4) + "if f.__name__ == 'in1d':")
                new_lines.append(' ' * (indent + 8) + "helps = None")
                new_lines.append(' ' * (indent + 4) + "else:")
                new_lines.append(' ' * (indent + 8) + "raise")
            else:
                new_lines.append(line)
        
        content = '\\n'.join(new_lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed numpy.in1d compatibility in {file_path}")
    else:
        print(f"Warning: Could not find '{old_line}' in {file_path}")
except Exception as e:
    print(f"Warning: Could not fix numpy.in1d compatibility: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(0)  # non-fatal error, continue installation
PYEOF
        """,
    ],
    'astropy__astropy-pull-18574': [
        "python -m pip install --force-reinstall --no-deps numpy==1.26.4",
        """
python3 << 'PYEOF'
import sys

file_path = "/testbed/astropy/units/quantity_helper/function_helpers.py"
try:
    with open(file_path, 'r') as f:
        content = f.read()
    
    # find and replace the line "helps = getattr(module, f.__name__)"
    old_line = "helps = getattr(module, f.__name__)"
    if old_line in content:
        lines = content.split('\\n')
        new_lines = []
        for i, line in enumerate(lines):
            if old_line in line:
                indent = len(line) - len(line.lstrip())
                # replace with the version with exception handling
                new_lines.append(' ' * indent + "try:")
                new_lines.append(' ' * (indent + 4) + "helps = getattr(module, f.__name__)")
                new_lines.append(' ' * indent + "except AttributeError:")
                new_lines.append(' ' * (indent + 4) + "# numpy 2.0+ removed in1d, skip this function")
                new_lines.append(' ' * (indent + 4) + "if f.__name__ == 'in1d':")
                new_lines.append(' ' * (indent + 8) + "helps = None")
                new_lines.append(' ' * (indent + 4) + "else:")
                new_lines.append(' ' * (indent + 8) + "raise")
            else:
                new_lines.append(line)
        
        content = '\\n'.join(new_lines)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Fixed numpy.in1d compatibility in {file_path}")
    else:
        print(f"Warning: Could not find '{old_line}' in {file_path}")
except Exception as e:
    print(f"Warning: Could not fix numpy.in1d compatibility: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(0)  # non-fatal error, continue installation
PYEOF
        """,
    ],
}

AFTER_INSTALL_ASTROPY = {
    # 'astropy__astropy-pull-18627': [
    #     "python -m pip install --force-reinstall --no-deps numpy==1.26.4",
    # ],
    # 'astropy__astropy-pull-18614': [
    #     "python -m pip install --force-reinstall --no-deps numpy==1.26.4",
    # ],
    # 'astropy__astropy-pull-18574': [
    #     "python -m pip install --force-reinstall --no-deps numpy==1.26.4",
    # ],
}

MAP_SPECS_INSTANCE_PRE_INSTALL = {
    "astropy/astropy": PRE_INSTALL_ASTROPY,
}

MAP_SPECS_INSTANCE_AFTER_INSTALL = {
    "astropy/astropy": AFTER_INSTALL_ASTROPY,
}

USE_X86 = {

}
