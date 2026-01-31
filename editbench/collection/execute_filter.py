import argparse
import json
import os.path
import resource
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import docker
from tqdm import tqdm

from editbench.collection.instance.activity import Activity, write_json_line
from editbench.config.constants import LOG_EXECUTION_FILTER, GITHUB_TOKEN
from editbench.evaluation.constants import EXECUTION_FILTER_LOG_DIR, INSTANCE_IMAGE_BUILD_DIR, \
    LOG_INSTANCE, LOG_REPORT, DOCKER_PATCH, DOCKER_WORKDIR, DOCKER_USER, APPLY_PATCH_FAIL, UTF8, APPLY_PATCH_PASS, \
    LOG_TEST_OUTPUT_BEFORE, LOG_TEST_OUTPUT_AFTER, TestStatus
from editbench.evaluation.docker_build import build_env_images, setup_logger, build_container, BuildImageError, \
    close_logger
from editbench.evaluation.docker_utils import list_images, clean_images, should_remove, copy_to_container, \
    exec_run_with_timeout, cleanup_container, remove_image
from editbench.evaluation.grading import get_logs_eval
from editbench.evaluation.run_evaluation import \
    EvaluationError
from editbench.evaluation.test_spec import make_test_spec, TestSpec, apply_script
from editbench.utils.dataset_utils import get_inf_datasets
from editbench.collection.utils import Repo


def evaluate_pass_and_failed(
        instances: list[Activity],
        client: docker.DockerClient,
        existing_images: set,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        timeout: int,
):
    test_specs = list(map(make_test_spec, instances))
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_for_instances = {
        tag for tag in existing_images if tag in instance_image_ids
    }

    if not force_rebuild and len(existing_for_instances):
        print(f"Found {len(existing_for_instances)} existing instance images. Will reuse them.")

    print(f"Running {len(instances)} instances with {max_workers} workers...")
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    get_pass_and_failed,
                    test_spec,
                    instance,
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    timeout,
                ): (test_spec, instance)
                for test_spec, instance in zip(test_specs, instances)
            }
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    future.result()
                except Exception:
                    traceback.print_exc()
    print("All instances run.")


def get_pass_and_failed(
        test_spec: TestSpec,
        instance: Activity,
        rm_image: bool,
        force_rebuild: bool,
        client: docker.DockerClient,
        timeout=None,
):
    # Set up logging directory
    instance_id = test_spec.instance_id
    log_dir = EXECUTION_FILTER_LOG_DIR / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            # link the image build dir in the log dir
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except:
            # some error, idk why
            pass
    log_file = log_dir / LOG_INSTANCE

    # Set up report file + logger
    report_path = log_dir / LOG_REPORT
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())
    logger = setup_logger(instance_id, log_file)

    # Run the instance
    container = None
    try:
        # Build + start instance container (instance image should already be built)
        container = build_container(test_spec, client, None, logger, rm_image, force_rebuild)
        container.start()
        logger.info(f"Container for {instance_id} started: {container.id}")

        # Run eval script, before apply patch
        eval_file = Path(log_dir / "eval.sh")
        # eval_rep_file = Path(log_dir / "eval_rep.sh")
        eval_file.write_text(test_spec.eval_script)
        # eval_rep_file.write_text(test_spec.eval_script_rep)
        logger.info(
            f"Eval script for {instance_id} written to {eval_file}; copying to container..."
            # f"Eval_rep script for {instance_id} written to {eval_rep_file}; copying to container..."
        )
        copy_to_container(container, eval_file, Path("/eval.sh"))
        # copy_to_container(container, eval_rep_file, Path("/eval_rep.sh"))

        # Run eval script, write output to logs
        logger.info(f"RUN eval script before apply patch...")
        test_output_before, timed_out, total_runtime = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout)
        test_output_before_path = log_dir / LOG_TEST_OUTPUT_BEFORE
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_before_path, "w") as f:
            f.write(test_output_before)
            logger.info(f"Test output for {instance_id} written to {test_output_before_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )
        # Get git diff before running eval script
        git_diff_output_before = (
            container.exec_run("git diff", workdir=DOCKER_WORKDIR).output.decode(UTF8, errors='ignore').strip()
        )
        logger.info(f"Git patch diff before:\n{git_diff_output_before}")

        # Copy patch file to container
        patch_file = Path(log_dir / "patch.diff")
        patch_file.write_text(instance.work_patch or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, Path(DOCKER_PATCH))

        # set patch bash
        patch_script_file = Path(log_dir / "apply.sh")
        apply_patch_script = apply_script("testbed", "/testbed",
                                          instance.base_commit, instance.work_patch)
        patch_script_file.write_text(apply_patch_script)
        copy_to_container(container, patch_script_file, Path("/apply.sh"))

        val = container.exec_run(
            cmd="/bin/bash /apply.sh",
            workdir=DOCKER_WORKDIR,
            user=DOCKER_USER,
        )

        if val.exit_code != 0:
            logger.info(f"Failed to apply patch to container, trying again...")
            # Attempt to apply patch to container
            val = container.exec_run(
                f"git apply --allow-empty -v {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR,
                user=DOCKER_USER,
            )
            if val.exit_code != 0:
                # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
                val = container.exec_run(
                    f"patch --batch --fuzz=5 -p1 -i {DOCKER_PATCH}",
                    workdir=DOCKER_WORKDIR,
                    user=DOCKER_USER,
                )
                if val.exit_code != 0:
                    logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8, errors='ignore')}")
                    raise EvaluationError(
                        instance_id,
                        f"{APPLY_PATCH_FAIL}:\n{val.output.decode(UTF8, errors='ignore')}",
                        logger,
                    )
                else:
                    logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8, errors='ignore')}")
            else:
                logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8, errors='ignore')}")
        else:
            logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode(UTF8, errors='ignore')}")

        # Get git diff after applying patch
        git_diff_output_after = (
            container.exec_run("git diff", workdir=DOCKER_WORKDIR).output.decode(UTF8, errors='ignore').strip()
        )

        # Check if git diff changed after running eval script
        logger.info(f"Git diff after:\n{git_diff_output_after}")
        if git_diff_output_after != git_diff_output_before:
            logger.info(f"Git diff changed after applying patch")

        # Re-run eval script, write output to logs
        test_output_after, timed_out, total_runtime = exec_run_with_timeout(container,
                                                                            "/bin/bash /eval.sh", timeout)
        test_output_after_path = log_dir / LOG_TEST_OUTPUT_AFTER
        logger.info(f'Test runtime: {total_runtime:_.2f} seconds')
        with open(test_output_after_path, "w") as f:
            f.write(test_output_after)
            logger.info(f"Test output for {instance_id} written to {test_output_after_path}")
            if timed_out:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                raise EvaluationError(
                    instance_id,
                    f"Test timed out after {timeout} seconds.",
                    logger,
                )

        # Get report from test output
        logger.info(f"Grading answer for {instance_id}...")

        report = get_pass_and_fail_report(test_spec, instance, test_output_before_path, test_output_after_path)

        logger.info(
            f"report: {report}\n"
            f"Result for {instance_id}:\n pass_to_pass: {report[instance_id]['p2p']}\n"
            f"pass_to_fail: {report[instance_id]['p2f']}\n"
            f"fail_to_fail: {report[instance_id]['f2f']}\n"
            f"fail_to_pass: {report[instance_id]['f2p']}\n"
        )
        # Write report to report.json
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=4))
        return instance_id, report
    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                     f"{traceback.format_exc()}\n"
                     f"Check ({logger.log_file}) for more information.")
        logger.error(error_msg)
    finally:
        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        # delete testbed
        target_dir = build_dir / "testbed"
        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
                print(f"Successfully deleted directory: {target_dir}")
            except OSError as e:
                # catch possible errors (e.g. permission issues, directory not found, etc.)
                print(f"Error deleting directory: {e.filename} - {e.strerror}")
        if rm_image:
            remove_image(client, test_spec.instance_image_key, logger)
        close_logger(logger)
    return


def get_pass_and_fail_report(
        test_spec: TestSpec,
        instance: Activity,
        log_before_path: str,
        log_after_path: str,
) -> dict:
    report_map = {}

    instance_id = instance.instance_id
    report_map[instance_id] = {
        "patch_is_None": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "p2p": [],
        "p2f": [],
        "f2f": [],
        "f2p": [],
    }

    # Check if the model patch exists
    if instance.work_patch is None:
        report_map[instance_id]["patch_is_None"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    # Get evaluation logs
    eval_sm_before, found_before = get_logs_eval(log_before_path, instance.repo)
    eval_sm_after, found_after = get_logs_eval(log_after_path, instance.repo)

    if not found_before and not found_after:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    # the previous status is "ERROR" and the method name does not contain "::", before_method does not contain "::" (maybe a test suite rather than a single test case), and the previous status is TestStatus.ERROR.value (error)
    for before_method, before_status in eval_sm_before.items():
        if "::" not in before_method and before_status == TestStatus.ERROR.value:
            true_methods = []
            false_methods = []
            for method, state in eval_sm_after.items():
                # if method.startswith(before_method) and method != before_method:
                if state in [TestStatus.PASSED.value, TestStatus.XFAIL.value]:
                    true_methods.append(method)
                elif state in [TestStatus.FAILED.value, TestStatus.ERROR.value]:
                    false_methods.append(method)
            report_map[instance_id]["f2p"].extend(true_methods)
            report_map[instance_id]["f2f"].extend(false_methods)
            continue
        if not eval_sm_after.get(before_method, None):
            continue
        if before_status in [TestStatus.FAILED.value, TestStatus.ERROR.value]:
            if eval_sm_after[before_method] in [TestStatus.FAILED.value, TestStatus.ERROR.value]:
                report_map[instance_id]["f2f"].append(before_method)
            elif eval_sm_after[before_method] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]:
                report_map[instance_id]["f2p"].append(before_method)
        elif before_status in [TestStatus.PASSED.value, TestStatus.XFAIL.value]:
            if eval_sm_after[before_method] in [TestStatus.FAILED.value, TestStatus.ERROR.value]:
                report_map[instance_id]["p2f"].append(before_method)
            elif eval_sm_after[before_method] in [TestStatus.PASSED.value, TestStatus.XFAIL.value]:
                report_map[instance_id]["p2p"].append(before_method)

    return report_map


def collect_json_results():
    f2p_cnt = 0
    valid_instacne = []
    path = LOG_EXECUTION_FILTER
    res_file = Path(path) / "report.json"
    res_file_valid = Path(path) / "report_valid.json"
    res_json = dict()
    valid_json = dict()
    for root, dirs, files in os.walk(path):
        if root == LOG_EXECUTION_FILTER:
            for dir in dirs:
                instance_id = dir
                json_path = Path(root) / instance_id / "report.json"
                if json_path.exists():
                    with open(json_path, "r") as f:
                        result = json.load(f)
                        res_json[dir] = result[instance_id]
                        if result[instance_id]["f2p"]:
                            f2p_cnt += 1
                            valid_instacne.append(instance_id)
                            valid_json[dir] = result[instance_id]
    print(f"f2p: {f2p_cnt},\ninstances: {valid_instacne}")
    with open(res_file, "w") as f:
        json.dump(res_json, f, indent=4)
    with open(res_file_valid, "w") as f:
        json.dump(valid_json, f, indent=4)
    return valid_instacne, valid_json


def gather_all_viable_activity(origin_jsonl, save_path, assure_jsonls: Optional[list[str]] = None):
    """
    Gather all viable activity to form executed version.
    :param origin_jsonl: collect activity based on report.json
    :param assure_jsonls: valid activities
    :return: None
    """
    all_activities = []
    seen_ids = set()
    for activity in get_inf_datasets(save_path, ""):
        all_activities.append(activity)
        seen_ids.add(activity.instance_id)

    # with open(save_path, "a") as fw:
    if assure_jsonls:
        for assure_jsonl in assure_jsonls:
            for activity in get_inf_datasets(assure_jsonl, ""):
                if activity.instance_id not in seen_ids:
                    # write_json_line(activity, fw)
                    all_activities.append(activity)
                    seen_ids.add(activity.instance_id)

    if origin_jsonl != "":
        valid_instance, valid_json = collect_json_results()
        for activity in get_inf_datasets(origin_jsonl, ""):
            if activity.instance_id not in seen_ids and activity.instance_id in valid_instance:
                activity.fail_to_pass = valid_json[activity.instance_id]["f2p"]
                activity.pass_to_pass = valid_json[activity.instance_id]["p2p"]
                activity.fail_to_fail = valid_json[activity.instance_id]["f2f"]
                activity.pass_to_fail = valid_json[activity.instance_id]["p2f"]
                all_activities.append(activity)
                # write_json_line(activity, fw)
                seen_ids.add(activity.instance_id)

        all_activities.sort(key=lambda x: x.created_at, reverse=True)
        with open(save_path, "w") as fw:
            for activity in all_activities:
                write_json_line(activity, fw)
        print(f"Gather completely, {len(seen_ids)}")


def reset_base_commit(dataset_path, new_path):
    """
    Gather all viable activity to form executed version.
    :param origin_jsonl: collect activity based on report.json
    :return: None
    """
    dataset = get_inf_datasets(dataset_path, "")

    repo_str = dataset[0].repo
    repo = Repo.from_full_name(repo_str, token=GITHUB_TOKEN)

    # save as jsonl, all_releases is a dict array
    new_ds = []
    seen_ids = set()
    for activity in get_inf_datasets(str(new_path), ""):
        seen_ids.add(activity.instance_id)
    with open(new_path, "a") as fw:
        for dataset_ in tqdm(dataset, total=len(dataset)):
            if dataset_.instance_id in seen_ids:
                continue
            pull_id = dataset_.instance_num
            pull = repo.call_api(repo.api.pulls.get, owner=repo.owner, repo=repo.name, pull_number=pull_id)
            # pull = repo.api.pulls.get(repo.owner, repo.name, pull_id)
            dataset_.base_commit = pull["base"]["sha"]
            new_ds.append(dataset_)
            write_json_line(dataset_, fw)


def main(
        dataset_name: str,
        instance_ids: list,
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
        open_file_limit: int,
        timeout: int,
):
    """
    Run execution filter harness for the given dataset.
    """
    # set open file limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    client = docker.from_env()
    dataset = get_inf_datasets(dataset_name, instance_ids=instance_ids)

    # Only run instances that do not have a result yet
    to_run = [
        d for d in dataset
        if not (LOG_EXECUTION_FILTER / d.instance_id).exists()
    ]
    existing_images = list_images(client)

    print(f"Running {len(to_run)} unevaluated instances (skipping {len(dataset) - len(to_run)} already done)...")
    if not to_run:
        print("No instance to run.")
    else:
        build_env_images(client, to_run, force_rebuild, max_workers)
        evaluate_pass_and_failed(
            to_run,
            client,
            existing_images,
            cache_level=cache_level,
            clean=clean,
            force_rebuild=force_rebuild,
            max_workers=max_workers,
            timeout=timeout,
        )
        clean_images(client, existing_images, cache_level, clean)
        collect_json_results()


if __name__ == "__main__":
    # Example:
    #   python -m editbench.collection.execute_filter \
    #     --dataset-name ./crawled_data/activity/astropy-astropy-task-instances.jsonl \
    #     --max-workers 4 --timeout 1800
    #   python -m editbench.collection.execute_filter \
    #     --dataset-name ./activity/django-django.jsonl ./activity/scikit-learn.jsonl --max-workers 4
    parser = argparse.ArgumentParser(description="Run execution filter harness (Docker run + pass/fail report).")
    parser.add_argument(
        "--dataset-name",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to task instance jsonl; can pass multiple, main will run over each in turn.",
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of instance_ids to run; if omitted, run all.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild Docker images.",
    )
    parser.add_argument(
        "--cache-level",
        type=str,
        default="env",
        choices=("none", "base", "env", "eval"),
        help="Docker image cache level (default: env).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean images above cache level after run.",
    )
    parser.add_argument(
        "--open-file-limit",
        type=int,
        default=4096,
        help="RLIMIT_NOFILE for the process (default: 4096).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per instance in seconds (default: 1800).",
    )
    args = parser.parse_args()
    for dataset_name in args.dataset_name:
        print(f"\n>>> Running execution filter for: {dataset_name}")
        main(
            dataset_name=dataset_name,
            instance_ids=args.instance_ids or [],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            timeout=args.timeout,
        )