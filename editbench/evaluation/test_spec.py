import hashlib
import platform
import re
from dataclasses import dataclass
from typing import Union, cast, Optional

from editbench.collection.instance.activity import Activity
from editbench.config import GITHUB_ORI_PROXY, GITHUB_ORI
from editbench.evaluation.constants import MAP_REPO_VERSION_TO_SPECS, MAP_REPO_TO_INSTALL, USE_X86, UTF8, \
    MAP_INSTALLED_REPO, MAP_SPECS_ENV_INSTANCE_PIP, MAP_SPECS_INSTANCE_PRE_INSTALL, MAP_SPECS_INSTANCE_AFTER_INSTALL
from editbench.evaluation.utils import get_requirements, get_environment_yml, get_test_directives
from editbench.evaluation.dockerfiles import get_dockerfile_base, get_dockerfile_env, get_dockerfile_instance, \
    get_dockerfile_instance_on_install

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"


@dataclass
class TestSpec:
    """
     A dataclass that represents a test specification for a single instance of Edit-bench.
    """
    instance_id: str
    repo: str
    version: str
    instance: Activity
    repo_script_list: list[str]
    eval_script_list: list[str]
    env_script_list: list[str]
    arch: str
    FAIL_TO_PASS: Optional[list[str]] = None
    PASS_TO_PASS: Optional[list[str]] = None

    @property
    def setup_env_script(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.env_script_list) + "\n"

    @property
    def eval_script(self):
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list) + "\n"
        # Don't exit early because we need to revert tests at the end

    # @property
    # def eval_script_rep(self):
    #     return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list[:-3] + self.eval_script_list[-2:]) + "\n"
    #     # Don't exit early because we need to revert tests at the end

    @property
    def install_repo_script(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.repo_script_list) + "\n"

    @property
    def install_repo_script_no_install(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.repo_script_list[1:]) + "\n"

    @property
    def base_image_key(self):
        return f"editb.base.{self.arch}:latest"

    @property
    def env_image_key(self):
        """
        The key for the environment image is based on the hash of the environment script list.
        If the environment script list changes, the image will be rebuilt automatically.

        Note that old images are not automatically deleted, so consider cleaning up old images periodically.
        """
        hash_object = hashlib.sha256()
        hash_object.update(str(self.env_script_list).encode(UTF8))
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]  # 22 characters is still very likely to be unique
        return f"editb.env.{self.arch}.{val}:latest"

    @property
    def instance_image_key(self):
        return f"editb.eval.{self.arch}.{self.instance_id}:latest"

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"editb.eval.{self.instance_id}"
        return f"editb.eval.{self.instance_id}.{run_id}"

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")

    @property
    def base_dockerfile(self):
        return get_dockerfile_base(self.platform, self.arch)

    @property
    def env_dockerfile(self):
        return get_dockerfile_env(self.platform, self.arch)

    @property
    def instance_dockerfile(self):
        return get_dockerfile_instance(self.platform, self.env_image_key)

    @property
    def instance_dockerfile_on_install(self):
        return get_dockerfile_instance_on_install(self.platform, self.env_image_key)


def get_test_specs_from_dataset(dataset: Union[list[Activity], list[TestSpec]]) -> list[TestSpec]:
    """
        Idempotent function that converts a list of InfInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)
    return list(map(make_test_spec, cast(list[Activity], dataset)))


def make_test_spec(instance: Activity) -> TestSpec:
    """
    Create TestSpec from the dataset instance for docker image creating
    Include:
    1. env_script: set conda, contains "python" version,
    "package" with conda, and "pip_packages" with pip

    2. repo_script: install repo and add some install command

    3. eval_script: apply test patches and run "test_command"
    """
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance.instance_id
    repo = instance.repo
    version = instance.version_fmt
    base_commit = instance.base_commit
    test_patches = instance.test_patch

    # TODO: FAIL_TO_PASS and PASS_TO_PASS to be added
    FAIL_TO_PASS = instance.fail_to_pass
    PASS_TO_PASS = instance.pass_to_pass

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    # Get the configuration information of the specified version of the target repository.
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    # set conda env
    env_script_list = make_env_script_list(instance, specs, env_name)
    repo_script_list = make_repo_script_list(specs, repo, repo_directory, base_commit, env_name, instance)
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patches
    )

    # if platform.machine() in ["aarch64", "arm64"]:
    #     # use arm64 unless explicitly specified
    #     arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    # else:
    #     arch = "x86_64"

    arch = "x86_64"
    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        version=version,
        instance=instance,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        arch=arch,
        FAIL_TO_PASS=FAIL_TO_PASS,
        PASS_TO_PASS=PASS_TO_PASS
    )


def make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    test_command = " ".join(
        [
            MAP_REPO_VERSION_TO_SPECS[instance.repo]
            [instance.version_fmt]["test_cmd"],
            *get_test_directives(instance)
        ]
    )
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        if "eval_with_install" not in specs or specs["eval_with_install"]:
            eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        test_command,
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


def apply_script(env_name, repo_directory, base_commit, code_patch, pre_edits=None):
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] +
                     make_patch_apply_script_list(env_name, repo_directory, base_commit, code_patch, pre_edits)) + "\n"


def make_patch_apply_script_list(env_name, repo_directory, base_commit, code_patch, pre_edits=None):
    pre_edits = pre_edits if pre_edits else []

    HEREDOC_DELIMITER = "EOF_114329324912"
    edit_files = re.findall(DIFF_MODIFIED_FILE_REGEX, code_patch)
    # Reset test files to the state they should be in before the patch.
    reset_edits_command = f"git checkout {base_commit} {' '.join(edit_files)}"

    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]

    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]

    eval_commands.append(reset_edits_command)

    for idx, patch in enumerate(pre_edits):
        eval_commands.append(f'echo "Start apply step patch-{idx + 1} of file: {re.findall(r"--- a/(.*)", patch)}..."')
        apply_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{patch}\n{HEREDOC_DELIMITER}"
        eval_commands.append(apply_patch_command)

    eval_commands.append(f'echo "Start apply prediction patch..."')
    apply_edit_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{code_patch}\n{HEREDOC_DELIMITER}"
    )

    eval_commands += [
        apply_edit_patch_command,
    ]

    return eval_commands


def make_repo_script_list(specs: dict, repo: str, repo_directory: str, base_commit: str, env_name: str, instance=None) -> list[str]:
    """
    Create a list of bash commands to set up the **repository for testing**.
    This is the setup script for the instance image.
    
    Args:
        specs: The specifications for the repository version
        repo: The repository name (e.g., "astropy/astropy")
        repo_directory: The directory where the repository will be cloned
        base_commit: The commit hash to checkout
        env_name: The conda environment name
        instance: The Activity instance (optional, used for instance-specific pre_install and after_install)
    """
    # Download the repository, switch commits, and prepare for configuring runtime environment.
    setup_commands = [
        f"git clone -o origin https://{GITHUB_ORI}/{repo} {repo_directory}",  # china
        # f"git clone -o origin https://{GITHUB_ORI}/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}", # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",
        # Remove the remote so the agent won't see newer commits.
        f"git remote remove origin",
        # Make sure conda is available for later use
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ]
    if repo in MAP_REPO_TO_INSTALL:
        setup_commands.append(MAP_REPO_TO_INSTALL[repo])

    # Run pre-install set up if provided (When there is content
    # that needs pre-install for the specific version of the task instance.)
    if "pre_install" in specs:
        for pre_install in specs["pre_install"]:
            setup_commands.append(pre_install)
    
    # Append instance-specific pre_install commands if available
    if instance is not None:
        instance_pre_install = MAP_SPECS_INSTANCE_PRE_INSTALL.get(repo, {}).get(instance.instance_id, [])
        if instance_pre_install:
            # If it's a list, extend; if it's a string, append
            if isinstance(instance_pre_install, list):
                setup_commands.extend(instance_pre_install)
            else:
                setup_commands.append(instance_pre_install)

    if "install" in specs:
        setup_commands.append(specs["install"])

    if "after_install" in specs:
        for after_install in specs["after_install"]:
            setup_commands.append(after_install)
    
    # Append instance-specific after_install commands if available
    if instance is not None:
        instance_after_install = MAP_SPECS_INSTANCE_AFTER_INSTALL.get(repo, {}).get(instance.instance_id, [])
        if instance_after_install:
            # If it's a list, extend; if it's a string, append
            if isinstance(instance_after_install, list):
                setup_commands.extend(instance_after_install)
            else:
                setup_commands.append(instance_after_install)

    return setup_commands


def make_env_script_list(instance: Activity, specs: dict, env_name: str) -> list[str]:
    """
        Creates the list of commands to set up the **conda environment for testing**.
        This is the setup script for the environment image.

        Returns:
            list[str]: List of commands to set up the conda environment
        """
    HEREDOC_DELIMITER = "EOF_59812759871"

    reqs_commands = [
        "source /opt/miniconda3/bin/activate"
    ]

    if "pre_packages" in specs:
        reqs_commands.append(specs["pre_packages"])

    # Create conda environment according to install instructinos
    pkgs = specs.get("packages", "")
    if pkgs == "requirements.txt":
        # Create conda environment with target python version
        cmd = f"conda create -n {env_name} python={specs['python']} -y"
        reqs_commands.append(cmd)

        # Install dependencies, str with '\n'
        reqs = replace_uninstallable_packages_requirements_txt(get_requirements(instance))
        # write reqs to path_to_reqs
        path_to_reqs = "$HOME/requirements.txt"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        cmd = f"conda activate {env_name} && python -m pip install -r {path_to_reqs}"
        reqs_commands.append(cmd)
        reqs_commands.append(f"rm {path_to_reqs}")

    elif pkgs == "environment.yml":
        # Create environment from yml
        reqs = get_environment_yml(instance, env_name)
        path_to_reqs = "environment.yml"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "no_use_env" in specs and specs["no_use_env"]:
            # `conda create` based installation
            cmd = f"conda create -c conda-forge -n {env_name} python={specs['python']} -y"
            reqs_commands.append(cmd)

            # Install dependencies
            cmd = f"conda env update -f {path_to_reqs}"
            reqs_commands.append(cmd)
        else:
            # `conda env create` based installation
            cmd = f"conda env create --file {path_to_reqs}"
            reqs_commands.append(cmd)

            cmd = f"conda activate {env_name} && conda install python={specs['python']} -y"
            reqs_commands.append(cmd)

        # Remove environment.yml
        reqs_commands.append(f"rm {path_to_reqs}")
    else:
        # Create environment + install dependencies
        cmd = f"conda create -n {env_name} python={specs['python']} {pkgs} -y"
        reqs_commands.append(cmd)

    reqs_commands.append(f"conda activate {env_name}")

    # Install additional packages if specified
    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        cmd = f"python -m pip install {pip_packages}"
        reqs_commands.append(cmd)

    # SPEC_PIP
    pip_values = MAP_SPECS_ENV_INSTANCE_PIP.get(instance.repo, {}).get(instance.instance_id, [])

    if pip_values:
        pip_packages = " ".join(pip_values)
        cmd = f"python -m pip install {pip_packages}"
        reqs_commands.append(cmd)

    return reqs_commands


def replace_uninstallable_packages_requirements_txt(requirement_str: str) -> str:
    """
    Replaces certain packages in a requirements.txt-like string.
        For example, some packages have been yanked and we need to replace them with compatible alternatives.
    """
    replacements = {
        "types-pkg_resources": "types-pkg-resources==0.1.3",
    }
    requirements = [req.strip() for req in requirement_str.split("\n") if req.strip()]
    requirements_replaced = []
    for requirement in requirements:
        if requirement in replacements:
            print(f"Replaced {requirement!r} with {replacements[requirement]!r} (replace_uninstallable_packages)")
            requirements_replaced.append(replacements[requirement])
        else:
            requirements_replaced.append(requirement)

    return "\n".join(requirements_replaced) + "\n"
