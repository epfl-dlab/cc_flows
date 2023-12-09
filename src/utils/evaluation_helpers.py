import numpy as np
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import hydra
from jsonlines import jsonlines

from src import utils
from src.utils.general_helpers import get_predictions_dir_path, write_jsonlines


log = utils.get_pylogger(__name__)


def unflatten_dict(dictionary: dict) -> dict:
    result_dict = dict()
    for key, value in dictionary.items():
        parts = key.split("/")
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict


def _sync_helper_download_file(wandb_file, exp_dir, replace):
    full_path = os.path.join(exp_dir, wandb_file.name)

    if os.path.exists(full_path) and not replace:
        log.info(f"File `{full_path}` already exists, skipping download.")
    elif os.path.exists(full_path) and replace:
        log.info(f"File `{full_path}` already exists, but will be replaced.")
        wandb_file.download(root=exp_dir, replace=replace)
    else:
        log.info(f"Downloading file `{full_path}` to `{exp_dir}`.")
        wandb_file.download(root=exp_dir, replace=replace)


def sync_results_from_wandb(wandb_run, exp_dir, replace):
    for f in wandb_run.files():
        if f.name == "results.json":
            _sync_helper_download_file(f, exp_dir, replace)


def sync_evaluation_output_from_wandb(wandb_run, exp_dir, replace):
    for f in wandb_run.files():
        if f.name == "evaluation_output.jsonl":
            _sync_helper_download_file(f, exp_dir, replace)


def sync_predictions_from_wandb(wandb_run, exp_dir, replace):
    for f in wandb_run.files():
        full_path = f.name
        file_name = full_path.split("/")[-1]
        if file_name.startswith("predictions") and file_name.endswith(".jsonl"):
            _sync_helper_download_file(f, exp_dir, replace)


def sync_experiment_data(
    wandb_run_path,
    log_func=log.info,
    sync_predictions=True,
    replace_predictions=False,
    sync_evaluation_output=True,
    replace_evaluation_output=True,
    sync_results=True,
    replace_results=True,
    work_dir=".",
):
    import wandb

    api = wandb.Api()
    run = api.run(wandb_run_path)

    wandb_run_config = unflatten_dict(run.config)
    wandb_run_hydra_config = wandb_run_config["hydra_config"]
    rel_path_to_exp_dir = wandb_run_hydra_config["output_dir"]
    exp_dir = os.path.join(work_dir, rel_path_to_exp_dir)

    log_func(f"Synchronizing experiment data at: {exp_dir}")

    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    get_predictions_dir_path(exp_dir, create_if_not_exists=True)

    if sync_predictions:
        sync_predictions_from_wandb(run, exp_dir, replace_predictions)
    if sync_evaluation_output:
        sync_evaluation_output_from_wandb(run, exp_dir, replace_evaluation_output)
    if sync_results:
        sync_results_from_wandb(run, exp_dir, replace_results)

    return wandb_run_config, wandb_run_hydra_config, exp_dir


def read_predictions(outputs_dir):
    items_dict = defaultdict(dict)
    for filename in os.listdir(outputs_dir):
        if not filename.endswith(".jsonl"):
            continue

        input_file_path = os.path.join(outputs_dir, filename)
        with open(input_file_path, "r+") as fp:
            # reader = jsonlines.Reader(fp)
            # for element in reader:
            for idx, line in enumerate(fp):
                try:
                    element = json.loads(line)
                except json.decoder.JSONDecodeError:
                    log.error(f"Failed to decode line {idx} in file {input_file_path}")
                    continue
                assert "id" in element
                # due to potentially non-even splits across processes, inference with ddp might result in duplicates
                # (i.e., the same datapoint might have been seen multiple times)
                # however we will always consider only one prediction (the last one)
                items_dict[element["id"]].update(element)

    items = [items_dict[_id] for _id in sorted(items_dict.keys())]
    return items


def read_evaluation_output(exp_dir):
    input_file_path = os.path.join(exp_dir, "evaluation_output.jsonl")
    if not os.path.isfile(input_file_path):
        return []

    items_dict = {}
    with open(input_file_path, "r+") as fp:
        reader = jsonlines.Reader(fp)
        for element in reader:
            assert "id" in element
            assert element["id"] not in items_dict

            items_dict[element["id"]] = element

    items = [items_dict[_id] for _id in sorted(items_dict.keys())]

    return items


def write_evaluation_output(exp_dir, items):
    output_file_path = os.path.join(exp_dir, "evaluation_output.jsonl")
    write_jsonlines(output_file_path, items)


def read_results(exp_dir):
    results_path = os.path.join(exp_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    return results


def write_results(exp_dir, results):
    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as outfile:
        json.dump(results, outfile)


def _update_paths_relative_to_data_dir(old_path, old_data_dir, new_data_dir):
    path_relative_to_data_dir = old_path.removeprefix(old_data_dir).strip("/")
    new_path = os.path.join(new_data_dir, path_relative_to_data_dir)
    return new_path


def get_dataset_used_in_run(hydra_config, split, data_dir=None):
    if data_dir is not None:
        old_data_dir = hydra_config["data_dir"]

        # dataset location
        old_path = hydra_config["datamodule"]["dataset_parameters"][split]["dataset"]["load_dataset_params"]["data_dir"]
        updated_path = _update_paths_relative_to_data_dir(old_path, old_data_dir, data_dir)
        hydra_config["datamodule"]["dataset_parameters"][split]["dataset"]["load_dataset_params"][
            "data_dir"
        ] = updated_path

        # evaluation buckets directory
        old_path = hydra_config["datamodule"]["dataset_parameters"][split]["dataset"]["evaluation_buckets_dir"]
        updated_path = _update_paths_relative_to_data_dir(old_path, old_data_dir, data_dir)
        hydra_config["datamodule"]["dataset_parameters"][split]["dataset"]["evaluation_buckets_dir"] = updated_path

    def fix_nones(items):
        result = {}
        for key, value in items:
            if value == "None":
                value = None
            result[key] = value
        return result

    dataset_cfg = json.dumps(hydra_config["datamodule"]["dataset_parameters"][split]["dataset"])
    dataset_cfg = json.loads(dataset_cfg, object_pairs_hook=fix_nones)

    dataset = hydra.utils.instantiate(dataset_cfg)
    return dataset


class EvaluationOutput:
    def __init__(self, exp_dir=None, data={}, problems_dataset=None):
        self.data = data
        self.dataset_name = None

        if exp_dir is not None:
            self.data = read_evaluation_output(exp_dir)

        if problems_dataset is not None:
            id2problem_data = {problem_data["id"]: problem_data for problem_data in problems_dataset}
            for item in self.data:
                item["problem_data"] = id2problem_data[item["id"]]
            self.dataset_name = problems_dataset.params["dataset_name"]

        log.info(f"Loaded {len(self.data)} datapoints from experiment dir {exp_dir}.")

    def get_bootstrapped_data(self, seed):
        data = self.data
        num_datapoints = len(data)

        random_state = np.random.RandomState(seed)
        bootstrap_ids = random_state.choice(len(self.data), num_datapoints, replace=True)

        bootstrap_data = [data[i] for i in bootstrap_ids]
        return bootstrap_data

    def get_filtered_data(self, ids_to_keep):
        ids_to_keep = set(ids_to_keep)
        return [dp for dp in self.data if dp["id"] in ids_to_keep]


class Results:
    def __init__(self, exp_dir=None, data={}):
        self.data = data

        if exp_dir is not None:
            self.data = read_results(exp_dir)

    def get_score(self, metric_id, reduce_buckets_to_mean):
        score = self.data[metric_id]["score"]
        if isinstance(score, dict) and reduce_buckets_to_mean:
            return np.mean(list(score.values()))
            # Below supports nested dicts, but when would we expect to have buckets of buckets?
            # is_subdict = False
            # for k, v in score.items():
            #     if isinstance(v, dict):
            #         is_subdict = True
            # if is_subdict == False:
            #     return np.mean(list(score.values()))
            # else:
            #     score_list = []
            #     for k, v in score.items():
            #         score_list.append(np.mean(list(v.values())))
            #     return np.mean(score_list)
        return score

    def get_bootstrap_runs_scores(self, metric_id, reduce_buckets_to_mean=False, group_by_bucket=False):
        assert not (reduce_buckets_to_mean and group_by_bucket)

        if not reduce_buckets_to_mean and not group_by_bucket:
            return self.data[metric_id].get("bootstrap_runs_scores", {})

        seed2score = self.get_bootstrap_runs_scores(metric_id, reduce_buckets_to_mean=False, group_by_bucket=False)
        if seed2score == {}:
            return {}

        is_bucketed = isinstance(list(seed2score.values())[0], dict)
        if not is_bucketed:
            return seed2score

        # ~~~ Precautionary check for bucketed metrics ~~~
        bucket_ids = None
        for seed, results in seed2score.items():
            if bucket_ids is None:
                bucket_ids = set(results.keys())
            else:
                assert bucket_ids == set(results.keys()), "The bucket ids across all bootstrap runs aren't the same"
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if reduce_buckets_to_mean:
            return {seed: np.mean(list(score.values())) for seed, score in seed2score.items()}

        # if group_by_bucket
        bucket2seed2scores = defaultdict(dict)
        for seed, results in seed2score.items():
            for bucket_id, score in results.items():
                bucket2seed2scores[bucket_id][seed] = score

        return bucket2seed2scores

    @staticmethod
    def _select_bootstrap_scores_for_ci(seed2score, n_bootstrap_samples):
        return [score for seed, score in sorted(seed2score.items(), key=lambda x: int(x[0]))[:n_bootstrap_samples]]

    def get_std_based_ci(self, metric_id, n_bootstrap_samples, reduce_buckets_to_mean=False):
        """Returns the 95% confidence interval based on the standard deviation of the bootstrap samples"""
        if reduce_buckets_to_mean:
            seed2score = self.get_bootstrap_runs_scores(metric_id, reduce_buckets_to_mean=True, group_by_bucket=False)
            return get_std_based_ci(self._select_bootstrap_scores_for_ci(seed2score, n_bootstrap_samples))

        br_scores = self.get_bootstrap_runs_scores(metric_id, reduce_buckets_to_mean=False, group_by_bucket=True)

        is_bucketed = isinstance(list(br_scores.values())[0], dict)
        if not is_bucketed:
            return get_std_based_ci(self._select_bootstrap_scores_for_ci(br_scores, n_bootstrap_samples))

        # if bucketed
        return {
            bucket_id: get_std_based_ci(self._select_bootstrap_scores_for_ci(seed2score, n_bootstrap_samples))
            for bucket_id, seed2score in br_scores.items()
        }

    def get_percentile_based_ci(self, metric_id, confidence_level, n_bootstrap_samples, reduce_buckets_to_mean=False):
        """Returns the `confidence_level`% confidence interval based on the empirical dist. of the bootstrap samples."""
        if reduce_buckets_to_mean:
            seed2score = self.get_bootstrap_runs_scores(metric_id, reduce_buckets_to_mean=True, group_by_bucket=False)
            return get_percentile_based_ci(
                self._select_bootstrap_scores_for_ci(seed2score, n_bootstrap_samples), confidence_level
            )

        br_scores = self.get_bootstrap_runs_scores(metric_id, reduce_buckets_to_mean=False, group_by_bucket=True)

        is_bucketed = isinstance(list(br_scores.values())[0], dict)
        if not is_bucketed:
            return get_percentile_based_ci(
                self._select_bootstrap_scores_for_ci(br_scores, n_bootstrap_samples), confidence_level
            )

        # if bucketed
        return {
            bucket_id: get_percentile_based_ci(
                self._select_bootstrap_scores_for_ci(seed2score, n_bootstrap_samples), confidence_level
            )
            for bucket_id, seed2score in br_scores.items()
        }


def get_percentile_based_ci(scores, confidence_level):
    alpha = (1 - confidence_level) / 2

    interval = alpha, 1 - alpha

    def percentile_fun(a, q):
        return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(scores, interval[0] * 100)
    ci_u = percentile_fun(scores, interval[1] * 100)
    return ci_l, np.mean(scores), ci_u


def get_std_based_ci(scores):
    std = np.std(scores)
    mean = np.mean(scores)

    ci_l = mean - 1.96 * std
    ci_u = mean + 1.96 * std
    return ci_l, mean, ci_u


def read_bucketing_data(evaluation_buckets_dir, bucketing_id):
    buckets_data_path = os.path.join(evaluation_buckets_dir, f"{bucketing_id}.json")
    with open(buckets_data_path, "r") as f:
        buckets_data_path = json.load(f)
    return buckets_data_path


def write_bucketing_data(evaluation_buckets_dir, bucketing_id, bucket_data):
    log.info(
        f"Writing the bucketing data, corresponding to {sum([len(dp_ids) for dp_ids in bucket_data.values()])} datapoints in {len(bucket_data)} buckets to `{os.path.join(evaluation_buckets_dir, f'{bucketing_id}.json')}`"
    )

    buckets_data_path = os.path.join(evaluation_buckets_dir, f"{bucketing_id}.json")
    with open(buckets_data_path, "w") as f:
        json.dump(bucket_data, f)


def is_username_pwd(proxy):
    return re.compile("([^:]+):([^\@]+)\@([\d\.]+):(\d+)").search(proxy)


def filter_proxy(proxy_list, filter_type):
    assert filter_type in [
        "all",
        "username_pwd",
        "ip_whitelist",
    ], f"filter_type must be one of ['all', 'username_pwd', 'ip_whitelist'], got {filter_type}"

    if filter_type == "all":
        return proxy_list

    res = []
    for proxy in proxy_list:
        if is_username_pwd(proxy) and filter_type == "username_pwd":
            res.append(proxy)
        elif is_username_pwd(proxy) is None and filter_type == "ip_whitelist":
            res.append(proxy)

    return res


def get_chrome_proxy_extension(proxy):

    CHROME_PROXY_HELPER_DIR = "chrome-proxy-extensions/Chrome-proxy-helper"
    CUSTOM_CHROME_PROXY_EXTENSIONS_DIR = "chrome-proxy-extensions"

    m = re.compile("([^:]+):([^\@]+)\@([\d\.]+):(\d+)").search(proxy)
    if m:
        username = m.groups()[0]
        password = m.groups()[1]
        ip = m.groups()[2]
        port = m.groups()[3]
        if not os.path.exists(CUSTOM_CHROME_PROXY_EXTENSIONS_DIR):
            os.mkdir(CUSTOM_CHROME_PROXY_EXTENSIONS_DIR)
        extension_file_path = os.path.join(CUSTOM_CHROME_PROXY_EXTENSIONS_DIR, "{}.zip".format(proxy.replace(":", "_")))
        if not os.path.exists(extension_file_path):
            zf = zipfile.ZipFile(extension_file_path, mode="w")
            if not os.path.exists(CHROME_PROXY_HELPER_DIR):
                os.mkdir(CHROME_PROXY_HELPER_DIR)
            zf.write(os.path.join(CHROME_PROXY_HELPER_DIR, "manifest.json"), "manifest.json")
            background_content = open(os.path.join(CHROME_PROXY_HELPER_DIR, "background.js")).read()
            background_content = background_content.replace("%proxy_host", ip)
            background_content = background_content.replace("%proxy_port", port)
            background_content = background_content.replace("%username", username)
            background_content = background_content.replace("%password", password)
            zf.writestr("background.js", background_content)
            zf.close()
        return extension_file_path
    else:
        raise Exception("Invalid proxy format. Should be username:password@ip:port")
