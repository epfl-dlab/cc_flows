import json
import os

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def is_config_compatible(config: DictConfig, dict_to_compare: dict):
    for key, value in dict_to_compare.items():
        value_in_config = OmegaConf.select(config, key.replace("+", ""))
        if value_in_config != value:
            return False
    return True


def existing_configs(directory: str,
                     return_result_paths: bool = False,
                     target_result_filename: str = "predictions.jsonl"):
    """
    Goes recursively inside the `directory` folder and searches for existing runs by checking hydra configs
    and checking whether the runs have a result file with the expected name.

    Args:
        directory: the log directory under which to search for existing configs and results.
        return_result_paths: whether to return the file path of the result files as well as the configs.
        target_result_filename: the expected name of the results file, if it exists.

    Returns: a list of dictionaries [{'config', 'result_file'}] if return_result_paths=True
    and [{'config'}] if return_result_paths=False

    """
    parent_directory = Path(directory).parent

    configs = []

    for run_directory in os.listdir(parent_directory):  # over 0,1,2,3
        run_path = os.path.join(parent_directory, run_directory)  # abs path to parent/run_idx/
        if not os.path.isdir(run_path):
            continue

        for experiment_directory in os.listdir(run_path):
            experiment_path = os.path.join(run_path, experiment_directory)
            if not os.path.isdir(experiment_path):
                continue

            # Skip unsuccessful runs if required
            result_file = os.path.join(experiment_path, target_result_filename)
            if not os.path.exists(result_file):
                continue

            conf = OmegaConf.load(os.path.join(experiment_path, '.hydra', 'config.yaml'))
            dict_res = {'config': conf}
            if return_result_paths:
                dict_res['result_file'] = result_file
            configs.append(dict_res)
    return configs

def gather_results(hydra_config: DictConfig):
    output_fp = hydra_config.output_dir
    hp_name = hydra_config.parent_run_name
    results = existing_configs(output_fp, return_result_paths=True)
    results_fp = os.path.join(str(Path(output_fp).parent), f"{hp_name}_results.json")

    search_space = OmegaConf.to_object(hydra_config.search.search_space)
    keys_to_log = []
    for search_param in search_space:
        # cleaning the parameter names to be easily readable
        keys_to_log.extend(search_param.keys())

    results_json = []
    for result in results:
        conf = result['config']
        keys_values = {}
        for key in keys_to_log:
            key = key.replace("+", "")
            keys_values[key] = OmegaConf.select(conf, key)

        relative_run_path = os.path.relpath(result['result_file'], Path(output_fp).parent)
        results_json.append({'param_config': keys_values,
                             'result_filepath': relative_run_path})

    with open(results_fp, 'w') as writer:
        writer.write(json.dumps(results_json))
