import re
from enum import Enum

from editbench.evaluation.constants import TestStatus


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if option.startswith("/") and not option.startswith("//") and "*" not in option:
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    # match the main pattern of test methods - enhanced version
    # strictly limit the format: test_xxx (module.Class.test_xxx) [possible results or new line]
    test_patterns = [
        # match the case where there is a result on the same line (strict word boundary)
        r"\b(test_\w+)\s*\(([^)]+)\)\s*\.\.\.\s*(ok|FAIL|ERROR|skipped)\b",

        # match the case where there is a result on the next line (non-greedy matching intermediate content)
        r"\b(test_\w+)\s*\(([^)]+)\)\r?\n(?:.*?)\.\.\.\s*(ok|FAIL|ERROR|skipped)\b"
    ]

    # process the test for special migration-related prefixes (enhanced non-greedy matching)
    migration_test_patterns = [
        r"Applying .*?\.\.\.(test_\w+)\s*\(([^)]+)\)\r?\n(?:.*?)\.\.\.\s*(ok|FAIL|ERROR|skipped)\b"
    ]

    # merge all patterns and process
    all_patterns = test_patterns + migration_test_patterns

    for pattern in all_patterns:
        # use DOTALL to process multiple lines, and use VERBOSE to ignore spaces
        for match in re.finditer(pattern, log, re.DOTALL | re.VERBOSE):
            test_name = f"{match.group(1)} ({match.group(2)})"
            status = match.group(3).lower()

            # map status to TestStatus
            if status == "ok":
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == "fail":
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status == "error":
                test_status_map[test_name] = TestStatus.ERROR.value
            elif status == "skipped":
                test_status_map[test_name] = TestStatus.SKIPPED.value

    # process the special case of --version
    if "--version is equivalent to version" in log:
        test_status_map["--version is equivalent to version"] = TestStatus.PASSED.value

    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f" {TestStatus.PASSED.value} " in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    # pattern = r"(_*) (.*)\.py:(.*) (_*)"
    pattern = r"FAILED (.+?)\.py::(\w+)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[0]}.py::{match[1]}"
        test_status_map[test_case] = TestStatus.FAILED.value

    pattern = r"(.+?\.py::\w+)\s+(\w+)"
    for line in log.split("\n"):
        if ".py::" in line:
            match = re.search(pattern, line)
            if match:
                test_case = match.group(1)  # extract "path::test name"
                status = match.group(2)  # extract status (PASSED/SKIPPED, etc.)
                if status == "PASSED":
                    test_status_map[test_case] = TestStatus.PASSED.value
                elif status == "FAILED":
                    test_status_map[test_case] = TestStatus.FAILED.value
                elif status == "SKIPPED":
                    test_status_map[test_case] = TestStatus.SKIPPED.value
                elif status == "ERROR":
                    test_status_map[test_case] = TestStatus.ERROR.value
        # line = line.strip()
        # if line.startswith("test_"):
        #     if line.endswith("[FAIL]") or line.endswith("[OK]"):
        #         line = line[: line.rfind("[")]
        #         line = line.strip()
        #     if line.endswith(" E"):
        #         test = line.split()[0]
        #         test_status_map[test] = TestStatus.ERROR.value
        #     if line.endswith(" F"):
        #         test = line.split()[0]
        #         test_status_map[test] = TestStatus.FAILED.value
        #     if line.endswith(" ok"):
        #         test = line.split()[0]
        #         test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


def parse_log_matplotlib(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_astropy,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_marshmallow,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_flask,
    "psf/requests": parse_log_requests,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pydata/xarray": parse_log_xarray,
    "pydicom/pydicom": parse_log_pydicom,
    "pylint-dev/astroid": parse_log_astroid,
    "pylint-dev/pylint": parse_log_pylint,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pyvista,
    "scikit-learn/scikit-learn": parse_log_scikit,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "sympy/sympy": parse_log_sympy,
}
