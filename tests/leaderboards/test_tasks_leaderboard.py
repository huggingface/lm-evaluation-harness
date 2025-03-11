import os
from dataclasses import asdict
from typing import Any, Dict, Tuple

import pytest

from lm_eval import evaluator
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager
from lm_eval.utils import hash_string

from .utils import ConfigParser, filter_dict, load_all_configs


# The device to use for evaluation
TEST_DEVICE = os.getenv("TESTS_DEVICE", "cpu")


def update_results(results: Dict, evaluation_tracker: EvaluationTracker) -> Dict:
    """
    Update the results dictionary with hashes and data from the evaluation tracker.

    Calculate hashes for each task's samples and adds them to the results.
    Incorporate general configuration data from the evaluation tracker.
    Adapted from the `lm_eval.loggers.evaluation_tracker` module's
    `save_results_aggregated` method.

    Args:
        results (Dict): The initial results dictionary to update.
            Expected to have a 'samples' key with task-specific sample data.
        evaluation_tracker (EvaluationTracker): An object containing evaluation metadata
            and a `general_config_tracker`.

    Returns:
        Dict: The updated results dictionary, including:
            - 'task_hashes': A dictionary with hashes for each task's aggregated samples.
            - General configuration data from the `evaluation_tracker`.
    """
    samples = results.get("samples")
    evaluation_tracker.general_config_tracker.log_end_time()

    task_hashes = {}
    for task_name, task_samples in samples.items():
        sample_hashes = hash_string(
            "".join(
                s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                for s in task_samples
            )
        )
        task_hashes[task_name] = sample_hashes

    # update initial results dict with the evaluation tracker data
    results.update({"task_hashes": task_hashes})
    results.update(asdict(evaluation_tracker.general_config_tracker))
    return results


def compare_results(
    reference: Dict[str, Any],
    observed: Dict[str, Any],
    config_name: str,
    module_name: str,
    recursive: bool = False,
):
    """
    Compare values between the reference and observed dictionaries,
    checking for equality or approximate equality for floats.
    Compare complex nested structures (e.g., dictionaries or lists) only if `recursive` is set to True.
    Raise an error if a key from the reference object is not present in the observed object.

    Args:
        reference (Dict[str, Any]): The reference dictionary to compare with.
        observed (Dict[str, Any]): The observed dictionary to compare.
        config_name (str): The name of the config to display in the error message.
        module_name (str): The associated name of the subresults to display in the error message.
        recursive (bool): Whether to compare nested structures.

    Example:
        reference = {
            "transformers_version": "4.41.1",
            "system_instruction_sha": None,
            "versions": {"drop": 3.0}  # Nested structure, compared only if recursive is set to True
        }
        observed = {
            "transformers_version": "4.41.1",
            "system_instruction_sha": None,
            "versions": {"drop": 3.0}  # Nested structure, compared only if recursive is set to True
        }
        compare_results(reference, observed, "example_config", "example_module")
    """
    if not reference:
        raise ValueError("Reference dictionary is empty.")

    for key, reference_val in reference.items():
        if isinstance(reference_val, Dict):
            if recursive:
                assert key in observed, (
                    f"Config: '{config_name} - {module_name}' failed. "
                    f"Key '{key}' is missing in the observed dictionary."
                )
                compare_results(
                    reference_val, observed[key], config_name, key, recursive
                )
        else:
            assert key in observed, (
                f"Config: '{config_name} - {module_name}' failed. "
                f"Key '{key}' is missing in the observed dictionary."
            )
            observed_val = observed[key]
            if isinstance(reference_val, float):
                assert reference_val == pytest.approx(observed_val, abs=1e-4), (
                    f"Config: '{config_name} - {module_name}' failed. {key}: "
                    f"Expected: {repr(reference_val)}, got: {repr(observed_val)}"
                )
            else:
                assert reference_val == observed_val, (
                    f"Config: '{config_name} - {module_name}' failed. {key}: "
                    f"Expected: {repr(reference_val)}, got: {repr(observed_val)}"
                )


@pytest.fixture(scope="module", params=load_all_configs(TEST_DEVICE))
def load_config(request):
    """
    Pytest fixture that loads and yields evaluation configurations with expected results.

    Use the `load_all_configs` function to retrieve configurations based on
    the device specified in the `TESTS_DEVICE` environment variable. If the variable is
    not set, it defaults to "cpu".

    Args:
        request: Pytest request object containing the parameter from the fixture decoration.

    Yields:
        Dict: A configuration dictionary containing:
            - Evaluation settings
            - Expected results for the given configuration
    """
    return request.param


@pytest.fixture(scope="module")
def evaluation_results(load_config: Dict) -> Tuple[ConfigParser, Dict]:
    """
    Pytest fixture that runs evaluations for all loaded configurations and returns the results.

    This fixture parses the configuration dictionary provided by the `load_config` fixture,
    and evaluates all specified tasks using the evaluator. The evaluation is ran separately for each task
    since `simple_evaluate` does not support providing separate sample limits
    and few-shot examples for each task.

    Args:
        load_config (Dict): A dictionary containing evaluation configurations and expected results.

    Returns:
        Tuple[ConfigParser, Dict]: A tuple containing:
            - config (ConfigParser): The parsed configuration with the expected results.
            - all_results (Dict): A dictionary containing the evaluation results for each task.
    """
    config = ConfigParser(load_config)
    evaluation_tracker = EvaluationTracker()
    task_manager = TaskManager(config.params.verbosity, include_path=None)

    all_results = {}
    for task_name, task in config.tasks.items():
        results = evaluator.simple_evaluate(
            model=config.params.model,
            model_args=config.params.model_args,
            tasks=[task_name],
            num_fewshot=task.num_fewshot,
            batch_size=config.params.batch_size,
            max_batch_size=config.params.max_batch_size,
            device=config.params.device,
            use_cache=config.params.use_cache,
            limit=task.limit,
            check_integrity=config.params.check_integrity,
            write_out=config.params.write_out,
            log_samples=config.params.log_samples,
            evaluation_tracker=evaluation_tracker,
            system_instruction=config.params.system_instruction,
            apply_chat_template=config.params.apply_chat_template,
            fewshot_as_multiturn=config.params.fewshot_as_multiturn,
            gen_kwargs=config.params.gen_kwargs,
            task_manager=task_manager,
            verbosity=config.params.verbosity,
            predict_only=config.params.predict_only,
            random_seed=config.params.random_seed,
            numpy_random_seed=config.params.numpy_seed,
            torch_random_seed=config.params.torch_seed,
            fewshot_random_seed=config.params.fewshot_seed,
        )
        results = update_results(results, evaluation_tracker)
        all_results[task_name] = results

    return config, all_results


def test_general_output(evaluation_results: Tuple[ConfigParser, Dict]):
    """
    Compare the general output of the evaluation results against the expected results.

    This function compares each task's corresponding output from the `all_results` dictionary,
    against the expected output from the configuration.
    The comparison checks the lowest level of the results dictionary, ensuring that
    all keys and values from the expected configuration match the observed results.

    Args:
        evaluation_results (Tuple[ConfigParser, Dict]): A tuple containing the evaluation configuration
        and the results for all tasks evaluated.

    Examples of keys and values compared in this test:
        - "transformers_version": "4.41.1"
        - "eot_token_id": 32000
    """
    config, all_results = evaluation_results

    for task_name in config.tasks.keys():
        results = all_results[task_name]
        compare_results(
            reference=config.to_dict(),
            observed=results,
            config_name=config.params.name,
            module_name="general_output",
            recursive=False,
        )


def test_evaluation_config(evaluation_results: Tuple[ConfigParser, Dict]):
    """
    Compare the provided evaluation configuration against the observed configuration.

    This function compares each task's configuration from the `all_results` dictionary,
    against the configuration which was used for evaluation.
    Configuration is a flat dictionary, so the comparison is done at the lowest level.

    Args:
        evaluation_results (Tuple[ConfigParser, Dict]): A tuple containing the evaluation configuration
        and the results for all tasks evaluated.

    Examples of keys and values compared in this test:
        - "model": "hf"
        - "model_revision": "main"
        - max_batch_size": 1
    """
    config, all_results = evaluation_results

    # Parameters to exclude from comparison as they are not present in the evaluation output
    # These parameters are only used to set up the evaluation
    excluded_params = [
        "evaluation_tracker_args",
        "system_instruction",
        "chat_template",
        "max_batch_size",
        "write_out",
        "check_integrity",
        "log_samples",
        "apply_chat_template",
        "fewshot_as_multiturn",
        "verbosity",
        "predict_only",
    ]
    # Prepare the expected configuration dictionary, it's task independent
    expected_config = {
        k: v for k, v in config.params.items() if k not in excluded_params
    }
    config_name = expected_config.pop("name")

    for task_name in config.tasks.keys():
        results = all_results[task_name]
        compare_results(
            reference=expected_config,
            observed=results["config"],
            config_name=config_name,
            module_name="config",
            recursive=False,
        )


def test_tasks_configs(evaluation_results: Tuple[ConfigParser, Dict]):
    """
    Compare the task configurations against the observed task configurations.

    This function compares each task's detailed configuration from the `all_results` dictionary,
    against the expected configuration for each task.
    If the configuration has more than one key, i.e. task consists of multiple subtasks,
    the function filters out only the subtasks of the currently considered task.

    Args:
        evaluation_results (Tuple[ConfigParser, Dict]): A tuple containing the evaluation configuration
        and the results for all tasks evaluated.

    Examples of configuration structure compared in this test:
        - "fewshot_delimiter": "\n\n"
        - "num_fewshot": 0
        - "output_type": "generate_until"
    """
    config, all_results = evaluation_results

    for task_name, task_configs in config.tasks.items():
        results = all_results[task_name]
        is_multitask = len(results["configs"]) > 1
        subtasks_configs = (
            filter_dict(task_configs.to_dict(), task_name)
            if is_multitask
            else {
                task_name: {
                    k: v for k, v in task_configs.to_dict().items() if k != "limit"
                }
            }
        )

        for subtask_name, subtask_config in subtasks_configs.items():
            compare_results(
                reference=subtask_config,
                observed=results["configs"][subtask_name],
                config_name=config.params.name,
                module_name=subtask_name,
                recursive=False,
            )


def test_tasks_results(evaluation_results: Tuple[ConfigParser, Dict]):
    """
    Compares the expected metrics values against the observed results for each task.

    This function compares model scores for each task or subtask from the `all_results` dictionary,
    against the expected results for each task.

    Args:
        evaluation_results (Tuple[ConfigParser, Dict]): A tuple containing the evaluation configuration
        and the results for all tasks evaluated.

    Examples of results structure compared in this test:
        - "exact_match,none": 0.0
        - "exact_match_stderr,none": "N/A"
    """
    config, all_results = evaluation_results

    for task_name, expected_results in config.results.items():
        results = all_results[task_name]
        observed_results = (
            results["results"]
            if len(results["results"]) > 1
            else results["results"][task_name]
        )

        compare_results(
            reference=expected_results.to_dict(),
            observed=observed_results,
            config_name=config.params.name,
            module_name=task_name,
            recursive=True,
        )


def test_tasks_n_samples(evaluation_results: Tuple[ConfigParser, Dict]):
    """
    Compares n_samples for each task and subtask.

    This function compares the number of samples for each task or subtask from the `all_results` dictionary,
    against the expected number of samples for each task.
    Subtasks are compared recursively due to the nested structure of the `n_samples` dictionary.

    Args:
        evaluation_results (Tuple[ConfigParser, Dict]): A tuple containing the evaluation configuration
        and the results for all tasks evaluated.

    Example of n samples structure compared in this test:
        - "leaderboard_mmlu_pro": {"effective": 10, "original": 11873},
    """
    config, all_results = evaluation_results

    for task_name, expected_n_samples in config.n_samples.items():
        results = all_results[task_name]
        observed_n_samples = (
            results["n-samples"]
            if len(results["n-samples"]) > 1
            else results["n-samples"][task_name]
        )

        compare_results(
            reference=expected_n_samples.to_dict(),
            observed=observed_n_samples,
            config_name=config.params.name,
            module_name=task_name,
            recursive=True,
        )


def test_tasks_hashes(evaluation_results: Tuple[ConfigParser, Dict]):
    """
    Compares task hashes for each task and subtask.

    This function compares hashes for each task or subtask from the `all_results` dictionary,
    against the expected hashes for each task.
    Subtasks are compared recursively due to the nested structure of the `task_hashes` dictionary.

    Args:
        evaluation_results (Tuple[ConfigParser, Dict]): A tuple containing the evaluation configuration
        and the results for all tasks evaluated.

    Examples of hashes structure compared in this test:
        - "leaderboard_gpqa": {
            "leaderboard_gpqa_diamond": "712e6d48eb8ef734cae32ac77163da1de61367f7a37cbb160592e5dd79629b14",
            "leaderboard_gpqa_extended": "f21cf1ce376bafcf42a5850189636241ab89839910e2ee7b22a7a3b13d6afd3b",
            "leaderboard_gpqa_main": "050a087c16e8a483a8916f1c982bf4197b542b41f9dbd88a37b58dd68ca936df"
          }
    """
    config, all_results = evaluation_results

    for task_name, expected_hashes in config.task_hashes.items():
        results = all_results[task_name]
        is_multitask = len(results["configs"].keys()) > 1

        if is_multitask:
            compare_results(
                reference=expected_hashes.to_dict(),
                observed=results["task_hashes"],
                config_name=config.params.name,
                module_name=task_name,
                recursive=True,
            )
        else:
            observed_hash = results["task_hashes"][task_name]
            assert expected_hashes == observed_hash, (
                f"Config: '{config.params.name}' failed. {task_name}: "
                f"Expected: {repr(expected_hashes)}, got: {repr(observed_hash)}"
            )


def test_tasks_versions(evaluation_results: Tuple[ConfigParser, Dict]):
    """
    Compares versions for each task and subtask.

    This function compares versions for each task or subtask from the `all_results` dictionary,
    against the expected versions for each task.
    Subtasks are compared recursively due to the nested structure of the `versions` dictionary.

    Args:
        evaluation_results (Tuple[ConfigParser, Dict]): A tuple containing the evaluation configuration
        and the results for all tasks evaluated.

    Examples of versions structure compared in this test:
        - "leaderboard_gpqa": {
            "leaderboard_gpqa_diamond": 1.0,
            "leaderboard_gpqa_extended": 1.0,
            "leaderboard_gpqa_main": 1.0
          }
    """
    config, all_results = evaluation_results

    for task_name, expected_versions in config.versions.items():
        results = all_results[task_name]
        is_multitask = len(results["versions"].keys()) > 1

        if is_multitask:
            compare_results(
                reference=expected_versions.to_dict(),
                observed=results["versions"],
                config_name=config.params.name,
                module_name=task_name,
                recursive=True,
            )
        else:
            observed_version = results["versions"][task_name]
            assert expected_versions == observed_version, (
                f"Config: '{config.params.name}' failed. {task_name}: "
                f"Expected: {repr(expected_versions)}, got: {repr(observed_version)}"
            )
