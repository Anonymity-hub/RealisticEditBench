import os
import signal
import tarfile
import threading
import time
import traceback
from logging import Logger
from pathlib import Path
from typing import Union, Optional

import docker
from docker.models.containers import Container


def list_images(client: docker.DockerClient):
    """
    List all images from the Docker client.
    Optimized to avoid opening too many file descriptors at once.
    Uses Docker API directly to minimize file descriptor usage.
    """
    try:
        # use Docker API directly to get the image list, to avoid the advanced method of the Python library
        # to reduce the number of file descriptors
        api_client = client.api
        images = api_client.images(all=True)
        
        # only extract the image tags, to avoid the extra inspect operation
        tags = set()
        for image in images:
            if 'RepoTags' in image and image['RepoTags']:
                tags.update(image['RepoTags'])
        return tags
    except OSError as e:
        if e.errno == 24:  # Too many open files
            print(f"Warning: Too many open files when listing images. Error: {e}")
            print("Returning empty image list to continue...")
            return set()
    except Exception as e:
        # if the API method fails, fall back to the Python library method
        print(f"Warning: Failed to list images using API method: {e}")
        print("Attempting fallback method...")
        try:
            image_list = client.images.list(all=True)
            tags = set()
            for image in image_list:
                if image.tags:
                    tags.update(image.tags)
            return tags
        except Exception as e2:
            print(f"Fallback method also failed: {e2}")
            return set()


def is_image_in_use(client: docker.DockerClient, image_name: str) -> bool:
    """
    Check if an image is currently being used by any container.
    This is important in multi-threaded environments to avoid deleting images
    that are currently in use.
    
    Args:
        client (docker.DockerClient): Docker client
        image_name (str): Image name to check
        
    Returns:
        bool: True if image is in use, False otherwise
    """
    try:
        # get the image ID
        try:
            image_obj = client.images.get(image_name)
            image_id = image_obj.id
            # also get all the image tags, for matching
            image_tags = set(image_obj.tags) if image_obj.tags else set()
        except docker.errors.ImageNotFound:
            return False
        
        # check all containers (including stopped ones)
        # note: only check the running containers, stopped containers will not prevent image deletion
        try:
            containers = client.containers.list(all=True)
            for container in containers:
                try:
                    # check if the container image ID matches
                    container_image_id = container.image.id
                    if container_image_id == image_id:
                        # check the container status, only the running containers will prevent deletion
                        container_status = container.attrs.get('State', {}).get('Status', '')
                        if container_status == 'running':
                            return True
                except Exception:
                    # if cannot get the container information, skip (for safety, assume not in use)
                    continue
        except Exception:
            # if cannot list the containers, for safety, assume the image is in use
            return True
        
        return False
    except Exception as e:
        # if the check fails, for safety, assume the image is in use
        return True


def remove_image(client, image_id, logger: Optional[Union[str, Logger]] = None):
    """
    Remove a Docker image by ID.

    Args:
        client (docker.DockerClient): Docker client.
        image_id (str): Image ID.
        logger (logging.Logger): Logger to use for output. If None, print to stdout.
    """
    if not logger:
        # if logger is None, print to stdout
        log_info = print
        log_error = print
        raise_error = True
    elif logger == "quiet":
        # if logger is "quiet", don't print anything
        log_info = lambda x: None
        log_error = lambda x: None
        raise_error = True
    else:
        # if logger is a logger object, use it
        log_error = logger.info
        log_info = logger.info
        raise_error = False
    try:
        log_info(f"Attempting to remove image {image_id}...")
        client.images.remove(image_id, force=True)
        log_info(f"Image {image_id} removed.")
    except docker.errors.ImageNotFound:
        log_info(f"Image {image_id} not found, removing has no effect.")
    except Exception as e:
        if raise_error:
            raise e
        log_error(
            f"Failed to remove image {image_id}: {e}\n" f"{traceback.format_exc()}"
        )


def find_dependent_images(client: docker.DockerClient, image_name: str):
    """
    Find all images that are built upon `image_name` image

    Args:
        client (docker.DockerClient): Docker client.
        image_name (str): Name of the base image.
    """
    dependent_images = []

    # use the optimized list_images function, to avoid opening too many file descriptors
    try:
        all_image_tags = list_images(client)
    except Exception as e:
        print(f"Warning: Failed to list images in find_dependent_images: {e}")
        return []

    # Get the ID of the base image using API method
    try:
        base_image_info = client.api.inspect_image(image_name)
        base_image_id = base_image_info.get("Id", "")
        if not base_image_id:
            print(f"Base image {image_name} not found.")
            return []
    except Exception:
        print(f"Base image {image_name} not found.")
        return []

    # Check each image using API method to avoid opening too many connections
    for image_tag in all_image_tags:
        try:
            # Skip the base image itself
            if image_tag == image_name:
                continue
            
            # Get image info using API
            image_info = client.api.inspect_image(image_tag)
            image_id = image_info.get("Id", "")
            if image_id == base_image_id:
                continue
            
            # Check if the base image is in this image's parent chain
            parent_id = image_info.get("Parent", "")
            layers = image_info.get("RootFS", {}).get("Layers", [])
            
            # Check parent chain (simplified check - just check immediate parent)
            # This is a simplified version - the full history check would require more API calls
            if base_image_id in [parent_id] or any(base_image_id in layer for layer in layers):
                dependent_images.append(image_tag)
        except Exception:
            # Skip images that can't be inspected
            continue

    return dependent_images


def cleanup_container(client, container, logger):
    """
    Stop and remove a Docker container.
    Performs this forcefully if the container cannot be stopped with the python API.

    Args:
        client (docker.DockerClient): Docker client.
        container (docker.models.containers.Container): Container to remove.
        logger (logging.Logger): Logger to use for output. If None, print to stdout
    """
    if not container:
        return

    container_id = container.id

    if not logger:
        # if logger is None, print to stdout
        log_error = print
        log_info = print
        raise_error = True
    elif logger == "quiet":
        # if logger is "quiet", don't print anything
        log_info = lambda x: None
        log_error = lambda x: None
        raise_error = True
    else:
        # if logger is a logger object, use it
        log_error = logger.info
        log_info = logger.info
        raise_error = False

    # Attempt to stop the container
    try:
        if container:
            log_info(f"Attempting to stop container {container.name}...")
            container.stop(timeout=15)
    except Exception as e:
        log_error(
            f"Failed to stop container {container.name}: {e}. Trying to forcefully kill..."
        )
        try:
            # Get the PID of the container
            container_info = client.api.inspect_container(container_id)
            pid = container_info["State"].get("Pid", 0)

            # If container PID found, forcefully kill the container
            if pid > 0:
                log_info(
                    f"Forcefully killing container {container.name} with PID {pid}..."
                )
                os.kill(pid, signal.SIGKILL)
            else:
                log_error(f"PID for container {container.name}: {pid} - not killing.")
        except Exception as e2:
            if raise_error:
                raise e2
            log_error(
                f"Failed to forcefully kill container {container.name}: {e2}\n"
                f"{traceback.format_exc()}"
            )

    # Attempt to remove the container
    try:
        log_info(f"Attempting to remove container {container.name}...")
        container.remove(force=True)
        log_info(f"Container {container.name} removed.")
    except Exception as e:
        if raise_error:
            raise e
        log_error(
            f"Failed to remove container {container.name}: {e}\n"
            f"{traceback.format_exc()}"
        )


def copy_to_container(container: Container, src: Path, dst: Path):
    """
    Copy a file from local to a docker container

    Args:
        container (Container): Docker container to copy to
        src (Path): Source file path
        dst (Path): Destination file path in the container
    """
    # Check if destination path is valid
    if os.path.dirname(dst) == "":
        raise ValueError(
            f"Destination path parent directory cannot be empty!, dst: {dst}"
        )

    # temporary tar file
    tar_path = src.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(src, arcname=src.name)

    # get bytes for put_archive cmd
    with open(tar_path, "rb") as tar_file:
        data = tar_file.read()

    # Make directory if necessary
    container.exec_run(f"mkdir -p {dst.parent}")

    # Send tar file to container and extract
    container.put_archive(os.path.dirname(dst), data)
    container.exec_run(f"tar -xf {dst}.tar -C {dst.parent}")

    # clean up in locally and in container
    tar_path.unlink()
    container.exec_run(f"rm {dst}.tar")


def exec_run_with_timeout(container, cmd, timeout=60):
    """
    Run a command in a container with a timeout.

    Args:
        container (docker.Container): Container to run the command in.
        cmd (str): Command to run.
        timeout (int): Timeout in seconds.
    """
    # Local variables to store the result of executing the command
    exec_result = b''
    exec_id = None
    exception = None
    timed_out = False

    # Wrapper function to run the command
    def run_command():
        nonlocal exec_result, exec_id, exception
        try:
            exec_id = container.client.api.exec_create(container.id, cmd)["Id"]
            exec_stream = container.client.api.exec_start(exec_id, stream=True)
            for chunk in exec_stream:
                exec_result += chunk
        except Exception as e:
            exception = e

    # Start the command in a separate thread
    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)

    if exception:
        raise exception

    # If the thread is still alive, the command timed out
    if thread.is_alive():
        if exec_id is not None:
            exec_pid = container.client.api.exec_inspect(exec_id)["Pid"]
            container.exec_run(f"kill -TERM {exec_pid}", detach=True)
        timed_out = True
    end_time = time.time()

    return exec_result.decode('utf-8', errors='replace'), timed_out, end_time - start_time


def should_remove(
        image_name: str,
        cache_level: str,
        clean: bool,
        prior_images: set,
        client: Optional[docker.DockerClient] = None,
        max_eval_images: Optional[int] = None
    ):
    """
    Determine if an image should be removed based on cache level and clean flag.
    
    Args:
        image_name (str): Name of the image to check
        cache_level (str): Cache level
        clean (bool): Whether to clean images above cache level
        prior_images (set): Set of images that existed before the current run
        client (docker.DockerClient, optional): Docker client for checking eval image count
        max_eval_images (int, optional): Maximum number of eval images to keep when cache_level="eval"
    """
    existed_before = image_name in prior_images
    if image_name.startswith("editb.base"):
        if cache_level in {"none"} and (clean or not existed_before):
            return True
    elif image_name.startswith("editb.env"):
        if cache_level in {"none", "base"} and (clean or not existed_before):
            return True
    elif image_name.startswith("editb.eval"):
        # if cache_level="eval", need to check if it exceeds max_eval_images
        if cache_level == "eval" and max_eval_images is not None and max_eval_images > 0:
            if client is not None:
                try:
                    images = list_images(client)
                    eval_images = [img for img in images if img.startswith("editb.eval")]
                    # if exceeds the limit, return True (should be deleted)
                    if len(eval_images) > max_eval_images:
                        return True
                except Exception:
                    # if the check fails, use the original logic
                    pass
        # the original logic: if cache_level is not "eval", or max_eval_images is not specified
        if cache_level in {"none", "base", "env"} and (clean or not existed_before):
            return True
    return False


def cleanup_excess_eval_images(
        client: docker.DockerClient,
        max_eval_images: int,
        exclude_images: Optional[set] = None,
        logger: Optional[Logger] = None
    ) -> int:
    """
    Clean up excess eval images when the number exceeds max_eval_images.
    Removes the oldest eval images first.
    
    Args:
        client (docker.DockerClient): Docker client
        max_eval_images (int): Maximum number of eval images to keep
        exclude_images (set, optional): Set of image names to exclude from removal
        logger (Logger, optional): Logger for logging
        
    Returns:
        int: Number of images removed
    """
    if max_eval_images <= 0:
        return 0
    
    log_info = logger.info if logger else print
    log_warning = logger.warning if logger else print
    
    try:
        images = list_images(client)
        eval_images = [img for img in images if img.startswith("editb.eval")]
        
        # if not exceeds the limit, no need to clean
        if len(eval_images) <= max_eval_images:
            return 0
        
        # get the image creation time, sorted by time (oldest first)
        eval_images_with_time = []
        for img_name in eval_images:
            # if the image is in the exclude list, skip
            if exclude_images and img_name in exclude_images:
                continue
            try:
                img_obj = client.images.get(img_name)
                # get the image creation timestamp (seconds)
                created_time = img_obj.attrs.get('Created', 0)
                # if it is a string format, need to convert
                if isinstance(created_time, str):
                    try:
                        from datetime import datetime
                        # try to parse the ISO format time
                        if 'Z' in created_time:
                            created_time = datetime.fromisoformat(created_time.replace('Z', '+00:00')).timestamp()
                        else:
                            # try other formats
                            created_time = datetime.fromisoformat(created_time).timestamp()
                    except (ValueError, AttributeError):
                        # if the parsing fails, use the current time (prioritize retention)
                        created_time = float('inf')
                elif created_time == 0:
                    # if the time is 0, use the current time (prioritize retention)
                    created_time = float('inf')
                eval_images_with_time.append((img_name, created_time))
            except Exception as e:
                log_warning(f"Could not get creation time for image {img_name}: {e}")
                # if cannot get the time, put it at the end (prioritize retention)
                eval_images_with_time.append((img_name, float('inf')))
        
        # sort by creation time (oldest first)
        eval_images_with_time.sort(key=lambda x: x[1])
        
        # calculate the number of images to delete
        # total eval images = len(eval_images) (including exclude)
        # exclude images = number of images excluded from eval_images_with_time
        # removable images = number of images that can be removed (excluding exclude)
        # target: keep max_eval_images images (including exclude)
        # therefore, the number of images to keep removable = max(0, max_eval_images - num_excluded)
        num_total = len(eval_images)  # total eval images (including exclude)
        num_removable = len(eval_images_with_time)  # removable images (excluding exclude)
        num_excluded = num_total - num_removable  # exclude images
        num_to_keep_removable = max(0, max_eval_images - num_excluded)  # number of images to keep removable
        num_to_remove = max(0, num_removable - num_to_keep_removable)  # number of images to delete
        
        if num_to_remove <= 0:
            return 0
        
        # delete the oldest images (skip images in use)
        removed_count = 0
        skipped_in_use = 0
        for img_name, _ in eval_images_with_time:
            if removed_count >= num_to_remove:
                break
            
            # check if the image is in use by any container (multi-thread safe)
            if is_image_in_use(client, img_name):
                skipped_in_use += 1
                log_info(f"Skipping image {img_name} - currently in use by a container")
                continue
            
            try:
                remove_image(client, img_name, logger or "quiet")
                removed_count += 1
                log_info(f"Removed old eval image to stay within limit: {img_name}")
            except docker.errors.ImageNotFound:
                # the image has already been deleted (possibly by another thread), skip
                log_info(f"Image {img_name} already removed (possibly by another thread)")
                continue
            except Exception as e:
                log_warning(f"Error removing eval image {img_name}: {e}")
                continue
        
        if skipped_in_use > 0:
            log_info(f"Skipped {skipped_in_use} images that are currently in use")
        
        if removed_count > 0:
            log_info(f"Removed {removed_count} eval images to stay within limit of {max_eval_images}")
        
        return removed_count
    except Exception as e:
        log_warning(f"Error in cleanup_excess_eval_images: {e}")
        return 0


def clean_images(
        client: docker.DockerClient,
        prior_images: set,
        cache_level: str,
        clean: bool,
        max_eval_images: Optional[int] = None
    ):
    """
    Clean Docker images based on cache level and clean flag.

    Args:
        client (docker.DockerClient): Docker client.
        prior_images (set): Set of images that existed before the current run.
        cache_level (str): Cache level to use.
        clean (bool): Whether to clean; remove images that are higher in the cache hierarchy than the current
            cache level. E.g. if cache_level is set to env, remove all previously built instances images. if
            clean is false, previously built instances images will not be removed, but instance images built
            in the current run will be removed.
        max_eval_images (int, optional): Maximum number of eval images to keep when cache_level="eval".
            If None, no limit. Only applies when cache_level="eval".
    """
    images = list_images(client)
    removed = 0
    
    # if cache_level="eval" and max_eval_images is specified, need special handling
    if cache_level == "eval" and max_eval_images is not None and max_eval_images > 0:
        print(f"Cleaning cached images (cache_level=eval, max_eval_images={max_eval_images})...")
        
        # use cleanup_excess_eval_images function to clean up excess eval images
        # exclude images in prior_images (unless clean=True)
        exclude_images = set() if clean else prior_images
        eval_removed = cleanup_excess_eval_images(
            client,
            max_eval_images,
            exclude_images=exclude_images,
            logger=None  # use print to output
        )
        removed += eval_removed
        
        # then process other types of images (base and env), using the original logic
        other_images = [img for img in images if not img.startswith("editb.eval")]
        for image_name in other_images:
            if should_remove(image_name, cache_level, clean, prior_images, client, max_eval_images):
                # check if the image is in use by any container (multi-thread safe)
                if is_image_in_use(client, image_name):
                    print(f"Skipping image {image_name} - currently in use by a container")
                    continue
                try:
                    remove_image(client, image_name, "quiet")
                    removed += 1
                except Exception as e:
                    print(f"Error removing image {image_name}: {e}")
                    continue
    else:
        # the original logic: cache_level is not "eval" or max_eval_images is not specified
        print("Cleaning cached images...")
        for image_name in images:
            if should_remove(image_name, cache_level, clean, prior_images, client, max_eval_images):
                # check if the image is in use by any container (multi-thread safe)
                if is_image_in_use(client, image_name):
                    print(f"Skipping image {image_name} - currently in use by a container")
                    continue
                try:
                    remove_image(client, image_name, "quiet")
                    removed += 1
                except Exception as e:
                    print(f"Error removing image {image_name}: {e}")
                    continue
    
    print(f"Removed {removed} images.")