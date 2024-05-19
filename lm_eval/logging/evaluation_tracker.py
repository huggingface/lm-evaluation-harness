import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import (
    DatasetCard,
    DatasetCardData,
    HfApi,
    hf_hub_url,
)
from collections import defaultdict

from lm_eval.utils import (
    eval_logger,
    handle_non_serializable,
    hash_string,
)


@dataclass(init=False)
class GeneralConfigTracker:
    """
    Tracker for the evaluation parameters.

    Attributes:
        model_source (str): Source of the model (e.g. Hugging Face, GGUF, etc.)
        model_name (str): Name of the model.
        model_name_sanitized (str): Sanitized model name for directory creation.
        start_time (float): Start time of the experiment. Logged at class init.
        end_time (float): Start time of the experiment. Logged when calling [`GeneralConfigTracker.log_end_time`]
        total_evaluation_time_seconds (str): Inferred total evaluation time in seconds (from the start and end times).
    """

    model_source: str = None
    model_name: str = None
    model_name_sanitized: str = None
    start_time: float = None
    end_time: float = None
    total_evaluation_time_seconds: str = None

    def __init__(self) -> None:
        """Starts the evaluation timer."""
        self.start_time = time.perf_counter()

    @staticmethod
    def _get_model_name(model_args: str) -> str:
        """Extracts the model name from the model arguments."""

        def extract_model_name(model_args: str, key: str) -> str:
            """Extracts the model name from the model arguments using a key."""
            args_after_key = model_args.split(key)[1]
            return args_after_key.split(",")[0]

        # order does matter, e.g. peft and delta are provided together with pretrained
        prefixes = ["peft=", "delta=", "pretrained=", "model=", "path=", "engine="]
        for prefix in prefixes:
            if prefix in model_args:
                return extract_model_name(model_args, prefix)
        return ""

    def log_experiment_args(
        self,
        model_source: str,
        model_args: str,
    ) -> None:
        """Logs model parameters and job ID."""
        self.model_source = model_source
        self.model_name = GeneralConfigTracker._get_model_name(model_args)
        self.model_name_sanitized = re.sub(
            r"[\"<>:/\|\\?\*\[\]]+", "__", self.model_name
        )

    def log_end_time(self) -> None:
        """Logs the end time of the evaluation and calculates the total evaluation time."""
        self.end_time = time.perf_counter()
        self.total_evaluation_time_seconds = str(self.end_time - self.start_time)


class EvaluationTracker:
    """
    Keeps track and saves relevant information of the evaluation process.
    Compiles the data from trackers and writes it to files, which can be published to the Hugging Face hub if requested.
    """

    def __init__(
        self,
        output_path: str = None,
        hub_results_org: str = "",
        hub_repo_name: str = "",
        push_results_to_hub: bool = False,
        push_samples_to_hub: bool = False,
        public_repo: bool = False,
        token: str = "",
    ) -> None:
        """
        Creates all the necessary loggers for evaluation tracking.

        Args:
            output_path (str): Path to save the results. If not provided, the results won't be saved.
            hub_results_org (str): The Hugging Face organisation to push the results to. If not provided, the results won't be pushed.
            hub_repo_name (str): The name of the Hugging Face repository to push the results to. If not provided, the results will be pushed to `lm-eval-results`.
            push_results_to_hub (bool): Whether to push the results to the Hugging Face hub.
            push_samples_to_hub (bool): Whether to push the samples to the Hugging Face hub.
            public_repo (bool): Whether to push the results to a public or private repository.
            token (str): Token to use when pushing to the Hugging Face hub. This token should have write access to `hub_results_org`.
        """
        self.general_config_tracker = GeneralConfigTracker()

        self.output_path = output_path
        self.hub_results_org = hub_results_org
        hub_repo_name = hub_repo_name if hub_repo_name else "lm-eval-results"
        self.hub_results_repo = f"{hub_results_org}/{hub_repo_name}"
        self.hub_results_repo_private = f"{hub_results_org}/{hub_repo_name}-private"
        self.push_results_to_hub = push_results_to_hub
        self.push_samples_to_hub = push_samples_to_hub
        self.public_repo = public_repo
        self.api = HfApi(token=token) if token else None

    def save_results_aggregated(
        self,
        results: dict,
        samples: dict,
    ) -> None:
        """
        Saves the aggregated results and samples to the output path and pushes them to the Hugging Face hub if requested.

        Args:
            results (dict): The aggregated results to save.
            samples (dict): The samples results to save.
        """
        self.general_config_tracker.log_end_time()

        if self.output_path:
            try:
                eval_logger.info("Saving results aggregated")

                # calculate cumulative hash for each task - only if samples are provided
                task_hashes = {}
                if samples:
                    for task_name, task_samples in samples.items():
                        sample_hashes = [
                            s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                            for s in task_samples
                        ]
                        task_hashes[task_name] = hash_string("".join(sample_hashes))

                # update initial results dict
                results.update({"task_hashes": task_hashes})
                results.update(asdict(self.general_config_tracker))
                dumped = json.dumps(
                    results,
                    indent=2,
                    default=handle_non_serializable,
                    ensure_ascii=False,
                )

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                self.date_id = datetime.now().isoformat().replace(":", "-")
                file_results_aggregated = path.joinpath(f"results_{self.date_id}.json")
                file_results_aggregated.open("w", encoding="utf-8").write(dumped)

                if self.api and self.push_results_to_hub:
                    self.api.create_repo(
                        repo_id=self.hub_results_repo
                        if self.public_repo
                        else self.hub_results_repo_private,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    self.api.upload_folder(
                        repo_id=self.hub_results_repo
                        if self.public_repo
                        else self.hub_results_repo_private,
                        folder_path=str(path),
                        path_in_repo=self.general_config_tracker.model_name_sanitized,
                        repo_type="dataset",
                        commit_message=f"Adding aggregated results for {self.general_config_tracker.model_name}",
                    )

            except Exception as e:
                eval_logger.warning("Could not save results aggregated")
                eval_logger.info(repr(e))
        else:
            eval_logger.info(
                "Output path not provided, skipping saving results aggregated"
            )


    def sanitize_list(self, sub):
        """
        takes possible nested list and recursively converts all inner component to strings
        """
        if isinstance(sub, list):
            return [self.sanitize_list(item) for item in sub]
        if isinstance(sub, tuple):
            return tuple(self.sanitize_list(item) for item in sub)
        else:
            return str(sub)

    def save_results_samples(
        self,
        task_name: str,
        samples: dict,
    ) -> None:
        """
        Saves the samples results to the output path and pushes them to the Hugging Face hub if requested.

        Args:
            task_name (str): The task name to save the samples for.
            samples (dict): The samples results to save.
        """
        if self.output_path:
            try:
                eval_logger.info("Saving samples results")
                # for each sample, dump the dict into a jsonl file

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                file_results_samples = path.joinpath(
                    f"samples_{task_name}_{self.date_id}.json"
                )

                for sample in samples:
                    # we first need to sanitize arguments and resps
                    # otherwise we won't be able to load the dataset
                    # using the datasets library
                    arguments = {}
                    for i, arg in enumerate(sample["arguments"]):
                        arguments[f"gen_args_{i}"] = {}
                        for j, tmp in enumerate(arg):
                            arguments[f"gen_args_{i}"][f"arg_{j}"] = tmp

                    sample["resps"] = self.sanitize_list(sample["resps"])
                    sample["filtered_resps"] = self.sanitize_list(sample["filtered_resps"])
                    sample["arguments"] = arguments

                    sample_dump = json.dumps(
                        sample,
                        default=handle_non_serializable,
                        ensure_ascii=False,
                    ) + "\n"

                    with open(file_results_samples, "a") as f:
                        f.write(sample_dump)

                if self.api and self.push_samples_to_hub:
                    self.api.create_repo(
                        self.hub_results_repo
                        if self.public_repo
                        else self.hub_results_repo_private,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    self.api.upload_folder(
                        repo_id=self.hub_results_repo
                        if self.public_repo
                        else self.hub_results_repo_private,
                        folder_path=str(path),
                        path_in_repo=self.general_config_tracker.model_name_sanitized,
                        repo_type="dataset",
                        commit_message=f"Adding samples results for {task_name} to {self.general_config_tracker.model_name}",
                    )
                    self.recreate_metadata_card()

            except Exception as e:
                eval_logger.warning("Could not save sample results")
                eval_logger.info(repr(e))
        else:
            eval_logger.info("Output path not provided, skipping saving sample results")

    def recreate_metadata_card(self) -> None:
        """
        Creates a metadata card for the evaluation results dataset and pushes it to the Hugging Face hub.
        """

        def get_file_task_name(filename: str) -> str:
            return filename[filename.find("_") + 1 : filename.rfind("_")]

        def get_file_datetime(filename: str) -> str:
            return filename[filename.rfind("_") + 1 :].replace(".json", "")

        repo_id = (
            self.hub_results_repo if self.public_repo else self.hub_results_repo_private
        )

        files_in_repo = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        results_files = [f for f in files_in_repo if "/results_" in f and ".json" in f]
        sample_files = [f for f in files_in_repo if "/samples_" in f and ".json" in f]
        multiple_results = len(results_files) > 1

        # build a dict with the latest datetime for each task
        # i.e. {"gsm8k": "2021-09-01T12:00:00", "ifeval": "2021-09-01T12:00:00"}
        latest_task_results_datetime = defaultdict(lambda: datetime.min.isoformat())

        for filename in sample_files:
            filename = os.path.basename(filename)
            task_name = get_file_task_name(filename)
            results_datetime = get_file_datetime(filename)
            latest_task_results_datetime[task_name] = (
                max(latest_task_results_datetime[task_name], results_datetime)
            )

        # get latest datetime and convert to isoformat
        max_latest_task_results_datetime = max(latest_task_results_datetime.values())

        # create metadata card
        card_metadata = MetadataConfigs()

        # add new results to the metatdata card
        for filename in results_files:
            results_filename = os.path.basename(filename)
            eval_date = get_file_datetime(results_filename)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = os.path.join("**", results_filename)
            sanitized_last_eval_date_results = re.sub(
                r"[^\w\.]", "_", max_latest_task_results_datetime
            )

            current_results = card_metadata.get("results", {"data_files": []})
            current_results["data_files"].append({"split": eval_date_sanitized, "path": [results_filename]})
            card_metadata["results"] = current_results

            if eval_date_sanitized == sanitized_last_eval_date_results:
                card_metadata["results"]["data_files"].append({"split": "latest", "path": [results_filename]})

        # Add the tasks details configs
        for filename in sample_files:
            filename = os.path.basename(filename)
            task_name = get_file_task_name(filename)
            eval_date = get_file_datetime(filename)
            task_name_sanitized = re.sub(r"\W", "_", task_name)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = os.path.join("**", os.path.basename(filename))
            sanitized_last_eval_date_results = re.sub(
                r"[^\w\.]", "_", latest_task_results_datetime[task_name]
            )

            current_details_for_task = card_metadata.get(task_name_sanitized, {"data_files": []})
            current_details_for_task["data_files"].append(
                {"split": eval_date_sanitized, "path": [results_filename]}
            )
            card_metadata[task_name_sanitized] = current_details_for_task

            if eval_date_sanitized == sanitized_last_eval_date_results:
                card_metadata[task_name_sanitized]["data_files"].append(
                    {"split": "latest", "path": [results_filename]}
                )

            # Special case for MMLU with a single split covering it all
            # We add another config with all MMLU splits results together for easy inspection
            # we add an `_` to the tasks name to avoid putting `mmlu` maths tasks to the `math` config
            SPECIAL_TASKS = [
                "mmlu_",
                "gpqa_",
                "math_"
            ]
            for special_task in SPECIAL_TASKS:
                if special_task in task_name_sanitized:
                    former_entry = card_metadata.get(special_task, {"data_files": []})

                    former_split = [(i, entry) for i, entry in enumerate(former_entry["data_files"]) if entry.get("split", None) == eval_date_sanitized]

                    if len(former_split) == 0:
                        former_entry["data_files"].append(
                            {"split": eval_date_sanitized, "path": [results_filename]}
                        )
                    else:
                        split_index, _ = former_split[0]
                        former_entry["data_files"][split_index]["path"].append(results_filename)

                    if eval_date_sanitized == sanitized_last_eval_date_results:
                        latest_split = [(i, entry) for i, entry in enumerate(former_entry["data_files"]) if entry.get("split", None) == "latest"]
                        if len(latest_split) == 0:
                            former_entry["data_files"].append({"split": "latest", "path": [results_filename]})
                        else:
                            latest_index, _ = latest_split[0]
                            former_entry["data_files"][latest_index]["path"].append(results_filename)

                    card_metadata[special_task] = former_entry

        # Cleanup a little the dataset card
        # Get the top results
        last_results_file = [
            f
            for f in results_files
            if max_latest_task_results_datetime.replace(":", "-") in f
        ][0]
        last_results_file_path = hf_hub_url(
            repo_id=repo_id, filename=last_results_file, repo_type="dataset"
        )
        f = load_dataset("json", data_files=last_results_file_path, split="train")
        results_dict = f["results"][0]
        new_dictionary = {"all": results_dict}
        new_dictionary.update(results_dict)
        results_string = json.dumps(new_dictionary, indent=4)

        # TODO change this
        # # If we are pushing to the Oppen LLM Leaderboard, we'll store specific data in the model card.
        # is_open_llm_leaderboard = repo_id.split("/")[0] == "open-llm-leaderboard"
        # if is_open_llm_leaderboard:
        #     org_string = (
        #         "on the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)."
        #     )
        #     leaderboard_url = "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard"
        #     point_of_contact = "clementine@hf.co"
        # else:
        #     org_string = ""
        #     leaderboard_url = None
        #     point_of_contact = None
        org_string = ""
        leaderboard_url = None
        point_of_contact = None

        dataset_summary = (
            "Dataset automatically created during the evaluation run of model "
        )
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += f"[{self.general_config_tracker.model_name}](https://huggingface.co/{self.general_config_tracker.model_name})"
        else:
            dataset_summary += f"{self.general_config_tracker.model_name}"
        dataset_summary += f"{org_string}.\n\n"
        f"The dataset is composed of {len(card_metadata)-1} configuration(s), each one corresponding to one of the evaluated task.\n\n"
        f"The dataset has been created from {len(results_files)} run(s). Each run can be found as a specific split in each "
        'configuration, the split being named using the timestamp of the run.The "train" split is always pointing to the latest results.\n\n'
        'An additional configuration "results" store all the aggregated results of the run.\n\n'
        "To load the details from a run, you can for instance do the following:\n"
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += f'```python\nfrom datasets import load_dataset\ndata = load_dataset(\n\t"{repo_id}",\n\t"{task_name_sanitized}",\n\tsplit="latest"\n)\n```\n\n'
        dataset_summary += "## Latest results\n\n"
        f'These are the [latest results from run {max_latest_task_results_datetime}]({last_results_file_path.replace("/resolve/", "/blob/")}) '
        "(note that their might be results for other tasks in the repos if successive evals didn't cover the same tasks. "
        'You find each in the results and the "latest" split for each eval):\n\n'
        (f"```python\n{results_string}\n```",)

        card_data = DatasetCardData(
            dataset_summary=dataset_summary,
            repo_url=f"https://huggingface.co/{self.general_config_tracker.model_name}",
            pretty_name=f"Evaluation run of {self.general_config_tracker.model_name}",
            leaderboard_url=leaderboard_url,
            point_of_contact=point_of_contact,
        )
        card_metadata.to_dataset_card_data(card_data)
        card = DatasetCard.from_template(
            card_data,
            pretty_name=card_data.pretty_name,
        )
        card.push_to_hub(repo_id, repo_type="dataset")
