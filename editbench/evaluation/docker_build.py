import logging
import os.path
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Union, Optional

import docker
import docker.errors
from tqdm import tqdm

from editbench.evaluation.constants import BASE_IMAGE_BUILD_DIR, ENV_IMAGE_BUILD_DIR, INSTANCE_IMAGE_BUILD_DIR, \
    MAP_REPO_VERSION_TO_SPECS, DOCKER_USER, MAP_INSTALLED_REPO
from editbench.evaluation.docker_utils import list_images, remove_image, find_dependent_images, cleanup_container
from editbench.evaluation.test_spec import get_test_specs_from_dataset, TestSpec

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def build_env_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False,
        max_workers: int = 4
):
    """
    Builds the environment images required for the dataset.
    """
    # Get the environment images to build from the datasets
    if force_rebuild:
        env_image_keys = {x.env_image_key for x in get_test_specs_from_dataset(dataset)}
        for key in env_image_keys:
            remove_image(client, key, "quiet")
    build_base_images(client, dataset, force_rebuild)
    # collect the configurations of the environment images to build
    configs_to_build = get_env_configs_to_build(client, dataset)
    if len(configs_to_build) == 0:
        print("No environment images need to be built.")
        return [], []
    print(f"Total environment images need to build: {len(configs_to_build)}")

    # Build the environment images
    successful, failed = list(), list()
    
    # Filter out images that already exist (unless force_rebuild is True)
    # batch check to avoid opening too many file descriptors
    images_to_build = {}
    try:
        # batch get all image tags
        all_images = list_images(client)
    except Exception as e:
        print(f"Warning: Failed to list all images: {e}. Will check images individually.")
        all_images = set()
    
    for image_name, config in configs_to_build.items():
        image_exists = image_name in all_images
        if not image_exists:
            # if the list does not contain the image, try to check directly (possibly incomplete list)
            try:
                client.api.inspect_image(image_name)
                image_exists = True
            except Exception:
                image_exists = False
        
        if image_exists:
            if force_rebuild:
                # remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
                images_to_build[image_name] = config
            else:
                print(f"Environment image {image_name} already exists, skipping build.")
        else:
            images_to_build[image_name] = config
    
    with tqdm(
            total=len(images_to_build), smoothing=0, desc="Building environment images"
    ) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each image to build
            futures = {
                executor.submit(build_image, image_name, {"setup_env.sh": config["setup_script"]},
                                config["dockerfile"], config["platform"],
                                client, ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__")
                                ): image_name
                for image_name, config in images_to_build.items()
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if image built successfully
                    future.result()
                    successful.append(futures[future])
                except BuildImageError as e:
                    print(f"BuildImageError {e.image_name}")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
                except Exception:
                    print("Error building image")
                    traceback.print_exc()
                    failed.append(futures[future])
                    continue
    # Show how many images failed to build
    if len(failed) == 0:
        print("All environment images built successfully.")
    else:
        print(f"{len(failed)} environment images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def setup_logger(instance_id: str, log_file: Path, mode="w"):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    return logger


def close_logger(logger):
    # To avoid too many open files
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def build_image(
        image_name: str,
        setup_scripts: dict,
        dockerfile: str,
        platform: str,
        client: docker.DockerClient,
        build_dir: Path,
        nocache: bool = True,
        repo_path: str = ""
):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    # Create a logger for the build process
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    try:
        # Write the setup scripts to the build directory
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )
        if repo_path:
            logger.info(f"Adding repo {repo_path} to image build repo")
            target_dir = build_dir / "testbed"

            if target_dir.exists():
                logger.info(f"Target directory {target_dir} exists, removing...")
                try:
                    shutil.rmtree(target_dir)
                except Exception as e:
                    logger.error(f"Failed to remove {target_dir}: {e}")
            try:
                shutil.copytree(repo_path, target_dir)
                logger.info(f"Successfully copied {repo_path} to {target_dir}")
            except Exception as e:
                logger.error(f"Failed to copy {repo_path} to {target_dir}: {e}")
        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build the image
        logger.info(
            f"Building docker image {image_name} in {build_dir} with platform {platform}"
        )

        response = client.api.build(
            path=str(build_dir),
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform=platform,
            nocache=nocache,
            container_limits={
                "memory": 8 * 1024 * 1024 * 1024,  # 8GB memory limit
                "memswap": 10 * 1024 * 1024 * 1024,  # 10GB total memory (including swap)
            }
        )

        # Log the build process continuously
        buildlog = ""
        for chunk in response:
            if "stream" in chunk:
                # Remove ANSI escape sequences from the log
                chunk_stream = ansi_escape.sub("", chunk["stream"])
                logger.info(chunk_stream.strip())
                buildlog += chunk_stream
            elif "errorDetail" in chunk:
                # Decode error message, raise BuildError
                logger.error(
                    f"Error: {ansi_escape.sub('', chunk['errorDetail']['message'])}"
                )
                raise docker.errors.BuildError(
                    chunk["errorDetail"]["message"], buildlog
                )
        logger.info("Image built successfully!")
    except docker.errors.BuildError as e:
        logger.error(f"docker.errors.BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)  # functions that create loggers should close them


def build_base_images(
        client: docker.DockerClient,
        dataset: list,
        force_rebuild: bool = False
):
    """
    Builds the base images (contains only the target platform (linux/x86_64 or linux/arm64) and essential tools (e.g., wget, Miniconda, Python))
    required for the dataset.
    """
    # Get the base images to build from the datasets
    test_specs = get_test_specs_from_dataset(dataset)
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }
    if force_rebuild:
        for key in base_images:
            remove_image(client, key, "quiet")

    # Build the base images
    # batch check if the images exist, to avoid opening too many file descriptors
    try:
        all_images = list_images(client)
    except Exception as e:
        print(f"Warning: Failed to list all images: {e}. Will check images individually.")
        all_images = set()
    
    for image_name, (dockerfile, platform) in base_images.items():
        image_exists = image_name in all_images
        if not image_exists:
            # if the list does not contain the image, try to check directly (possibly incomplete list)
            try:
                client.api.inspect_image(image_name)
                image_exists = True
            except Exception:
                image_exists = False
        
        if image_exists:
            if force_rebuild:
                # remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
            else:
                print(f"Base image {image_name} already exists, skipping build.")
                continue
        # build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )
    print("Base images built successfully.")


def build_container(
        test_spec: TestSpec,
        client: docker.DockerClient,
        run_id: Optional[str],
        logger: logging.Logger,
        nocache: bool = False,
        force_rebuild: bool = False
):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    """
    # Build corresponding instance image
    if force_rebuild:
        remove_image(client, test_spec.instance_image_key, "quiet")
    build_instance_image(test_spec, client, logger, nocache)

    container = None
    try:
        # Get configurations for how container should be created
        config = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
        user = DOCKER_USER if not config.get("execute_test_as_nonroot", False) else "nonroot"
        nano_cpus = config.get("nano_cpus")

        # Create the container
        logger.info(f"Creating container for {test_spec.instance_id}...")
        container = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=user,
            detach=True,
            command="tail -f /dev/null",
            nano_cpus=nano_cpus,
            platform=test_spec.platform,
        )
        logger.info(f"Container for {test_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e


def get_env_configs_to_build(
        client: docker.DockerClient,
        dataset: list,
):
    """
    Returns a dictionary of image names to build scripts and dockerfiles for environment images.
    Returns only the environment images that need to be built.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
    """
    image_scripts = dict()
    base_images = dict()
    env_images = dict()  # cache environment image information
    test_specs = get_test_specs_from_dataset(dataset)

    # step 1: batch get all the image information needed, to reduce the number of file descriptors
    # collect all unique image keys
    unique_base_image_keys = set()
    unique_env_image_keys = set()
    for test_spec in test_specs:
        unique_base_image_keys.add(test_spec.base_image_key)
        unique_env_image_keys.add(test_spec.env_image_key)
    
    # batch get base image information (using API method to reduce the number of connections)
    all_images = set()
    try:
        all_images = list_images(client)
    except Exception as e:
        print(f"Warning: Failed to list all images: {e}. Will check images individually.")
    
    # batch get base image information
    for base_image_key in unique_base_image_keys:
        if base_image_key in all_images:
            try:
                # use API method to get image information, to reduce the number of connections
                base_images[base_image_key] = client.api.inspect_image(base_image_key)
            except Exception as e:
                raise Exception(
                    f"Base image {base_image_key} not found or cannot be inspected: {e}\n"
                    "Please build the base images first."
                )
        else:
            # if the list does not contain the image, try to get directly (possibly incomplete list)
            try:
                base_images[base_image_key] = client.api.inspect_image(base_image_key)
            except Exception:
                raise Exception(
                    f"Base image {base_image_key} not found.\n"
                    "Please build the base images first."
                )
    
    # batch check if the environment images exist
    for env_image_key in unique_env_image_keys:
        if env_image_key in all_images:
            try:
                # use API method to get image information, to reduce the number of connections
                env_images[env_image_key] = client.api.inspect_image(env_image_key)
            except Exception:
                env_images[env_image_key] = None
        else:
            # try to get directly, if not exist, set to None
            try:
                env_images[env_image_key] = client.api.inspect_image(env_image_key)
            except Exception:
                env_images[env_image_key] = None

    # step 2: process each test_spec, decide if it needs to be built
    for test_spec in test_specs:
        base_image_info = base_images.get(test_spec.base_image_key)
        if base_image_info is None:
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.env_image_key}\n."
                "Please build the base images first."
            )
        
        # check if the environment image exists
        env_image_info = env_images.get(test_spec.env_image_key)
        image_exists = env_image_info is not None

        if image_exists:
            # check the image creation time, decide if it needs to be rebuilt
            # api.inspect_image returns a dictionary, Created is a string timestamp
            base_created_str = base_image_info.get("Created", "")
            env_created_str = env_image_info.get("Created", "")
            
            # compare the timestamp strings (ISO 8601 format)
            if base_created_str and env_created_str and env_created_str < base_created_str:
                # the environment image is older than the base image, needs to be rebuilt
                for dep in find_dependent_images(client, test_spec.env_image_key):
                    remove_image(client, dep, "quiet")
                remove_image(client, test_spec.env_image_key, "quiet")
                image_exists = False
        
        if not image_exists:
            # Add the environment image to the list of images to build
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
            }
    
    return image_scripts


def build_instance_image(
        test_spec: TestSpec,
        client: docker.DockerClient,
        logger: Union[logging.Logger, None],
        nocache: bool,
):
    """
    Builds the instance image for the given test spec if it does not already exist.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
    """
    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    # Get the image names and dockerfile for the instance image
    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key


    # Check that the env. image the instance image is based on exists
    # use API method to check the image, to reduce the number of file descriptors
    try:
        env_image_info = client.api.inspect_image(env_image_name)
    except Exception as e:
        raise BuildImageError(
            test_spec.instance_id,
            f"Environment image {env_image_name} not found for {test_spec.instance_id}",
            logger,
        ) from e
    logger.info(
        f"Environment image {env_image_name} found for {test_spec.instance_id}\n"
        f"Building instance image {image_name} for {test_spec.instance_id}"
    )

    # Check if the instance image already exists
    image_exists = False
    try:
        instance_image_info = client.api.inspect_image(image_name)
        # compare the image creation time (API returns a dictionary)
        env_created = env_image_info.get("Created", "")
        instance_created = instance_image_info.get("Created", "")
        if instance_created and env_created and instance_created < env_created:
            # the environment image is newer than the instance image, meaning the instance image may be outdated
            remove_image(client, image_name, "quiet")
            image_exists = False
        else:
            image_exists = True
    except Exception:
        pass

    # Build the instance image
    if not image_exists:
        # change dockerfile and setup_repo.sh
        if MAP_INSTALLED_REPO.get(test_spec.repo, None) and os.path.exists(MAP_INSTALLED_REPO[test_spec.repo]):
            print(f"Building instance image {image_name} for {test_spec.instance_id} with installed repo {MAP_INSTALLED_REPO[test_spec.repo]}")
            install_repo_script = test_spec.install_repo_script_no_install
            dockerfile = test_spec.instance_dockerfile_on_install
            repo_path = MAP_INSTALLED_REPO[test_spec.repo]
        else:
            install_repo_script = test_spec.install_repo_script
            dockerfile = test_spec.instance_dockerfile
            repo_path = ""

        build_image(
            image_name=image_name,
            setup_scripts={
                "setup_repo.sh": install_repo_script,
            },
            dockerfile=dockerfile,
            platform=test_spec.platform,
            client=client,
            build_dir=build_dir,
            nocache=nocache,
            repo_path=repo_path
        )
    else:
        logger.info(f"Image {image_name} already exists, skipping build.")

    if new_logger:
        close_logger(logger)
