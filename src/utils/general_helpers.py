import collections
import gzip
import json

import hydra
import jsonlines
import warnings
import time
import uuid
import os

from dataclasses import is_dataclass
from typing import Callable, List

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from pathlib import Path
from src import utils
from src.utils import rich_utils
from importlib.util import find_spec
from copy import deepcopy

log = utils.get_pylogger(__name__)


def run_task(cfg: DictConfig, run_func: Callable, upload_predictions=False) -> None:
    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    extras(cfg)

    # execute the task
    try:
        start_time = time.time()
        run_func(cfg)
    except Exception as ex:
        log.exception("")  # save exception to `.log` file
        raise ex
    finally:
        if upload_predictions:
            upload_predictions_to_wandb(
                base_path=cfg.output_dir, predictions_dir=get_predictions_dir_path(cfg.output_dir)
            )

        current_time_stamp = f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
        path = Path(cfg.output_dir, f"exec_time_{current_time_stamp}.log")
        content = (
            f"Execution time: {time.time() - start_time:.2f} seconds "
            f"-- {(time.time() - start_time) / 60:.2f} minutes "
            f"-- {(time.time() - start_time) / 3600:.2f} hours"
        )
        log.info(content)
        save_string_to_file(path, content)  # save task execution time (even if exception occurs)
        close_loggers()  # close loggers (even if exception occurs so multirun won't fail)
        log.info(f"Output directory: `{os.path.join(cfg.work_dir, cfg.output_dir)}`")


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings
    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            log.info(f"WandB run path: `{wandb.run.path}`")
            log.info("Closing wandb!")
            wandb.finish()


def get_predictions_dir_path(output_dir, create_if_not_exists=True):
    if output_dir is not None:
        predictions_folder = os.path.join(output_dir, "predictions")
    else:
        predictions_folder = "predictions"

    if create_if_not_exists:
        Path(predictions_folder).mkdir(parents=True, exist_ok=True)

    return predictions_folder


def write_outputs(output_file, summary, mode):
    # Custom serializer function for JSON
    def dataclass_serializer(obj):
        if is_dataclass(obj):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Custom dumps function
    def dataclass_dumps(obj):
        return json.dumps(obj, default=dataclass_serializer)

    with open(output_file, mode) as fp:
        json_writer = jsonlines.Writer(fp, dumps=dataclass_dumps)
        json_writer.write_all(summary)


def read_jsonlines(path_to_file):
    with open(path_to_file, "r") as f:
        json_reader = jsonlines.Reader(f)
        return list(json_reader)


def write_jsonlines(path_to_file, data, mode="w"):
    with jsonlines.open(path_to_file, mode) as writer:
        writer.write_all(data)


def write_gzipped_jsonlines(path_to_file, data, mode="w"):
    with gzip.open(path_to_file, mode) as fp:
        json_writer = jsonlines.Writer(fp)
        json_writer.write_all(data)


def read_gzipped_jsonlines(path_to_file):
    with gzip.open(path_to_file, "r") as fp:
        json_reader = jsonlines.Reader(fp)
        return list(json_reader)


def recursive_dictionary_update(d, u):
    """Performs a recursive update of the values in dictionary d with the values of dictionary u"""
    if d is None:
        d = {}

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dictionary_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config."""
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def sanitize_dict(cfg, private_fields, current_path):
    for key, item in cfg.items():
        new_path = current_path + "." + key if current_path != "" else key
        if new_path in private_fields.get("private_variable_paths", []) or key in private_fields.get(
            "private_variable_names", []
        ):
            cfg[key] = "**PRIVATE**"
        elif isinstance(item, collections.abc.Mapping):
            sanitize_dict(cfg.get(key), private_fields, new_path)
        elif isinstance(item, list):
            cfg[key] = [sanitize_dict(x, private_fields, new_path) for x in item]
    return cfg


def sanitize_config(cfg):
    if "private_fields" in cfg and (
        len(cfg["private_fields"].get("private_variable_paths", [])) > 0
        or len(cfg["private_fields"].get("private_variable_names", [])) > 0
    ):
        cp_cfg = deepcopy(cfg)
        cp_cfg = sanitize_dict(cp_cfg, private_fields=cp_cfg["private_fields"], current_path="")
        return cp_cfg

    log.warning(f"No private fields specified, private information might be logged")
    return cfg


@rank_zero_only
def save_string_to_file(path: str, content: str, append_mode=True) -> None:
    """Save string to file in rank zero mode (only on the master process in multi-GPU setup)."""
    mode = "a+" if append_mode else "w+"
    with open(path, mode) as file:
        file.write(content)


@rank_zero_only
def log_hyperparameters(hydra_config, model, loggers) -> None:
    hparams = {}
    sanitized_config = sanitize_config(hydra_config)
    if not loggers:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams_to_log = OmegaConf.to_container(sanitized_config, resolve=True)
    if "local" in hparams_to_log:
        del hparams_to_log["local"]

    for key in hparams:
        if isinstance(hparams[key], DictConfig):
            hparams[key] = OmegaConf.to_container(hparams[key], resolve=True)

    hparams["hydra_config"] = hparams_to_log

    for lg in loggers:
        lg.log_hyperparams(hparams)


@rank_zero_only
def upload_predictions_to_wandb(base_path, predictions_dir):
    import wandb

    if wandb.run:
        wandb.save(f"{predictions_dir}/*", base_path=base_path, policy="now")


def upload_file_to_wandb(base_path, path_to_file):
    import wandb

    if wandb.run:
        wandb.save(path_to_file, base_path=base_path, policy="now")


def create_unique_id(existing_ids=set()):
    while True:
        unique_id = str(uuid.uuid4())
        if unique_id not in existing_ids:
            return unique_id


def get_current_datetime_ns():
    time_of_creation_ns = time.time_ns()

    # Convert nanoseconds to seconds and store as a time.struct_time object
    time_of_creation_struct = time.gmtime(time_of_creation_ns // 1000000000)

    # Format the time.struct_time object into a human-readable string
    formatted_time_of_creation = time.strftime("%Y-%m-%d %H:%M:%S", time_of_creation_struct)

    # Append the nanoseconds to the human-readable string
    formatted_time_of_creation += f".{time_of_creation_ns % 1000000000:09d}"

    return formatted_time_of_creation
