from aiflows import flow_verse

dependencies = [{"url": "aiflows/CCFlows", "revision": "main"}]
# For smoother development, to use an implementation from a local repository, pass the (absolute) local path as revision
# local_path = "<absolute_path_to_flow_directory>"
# dependencies = [{"url": "aiflows/CCFlows", "revision": local_path}]

flow_verse.sync_dependencies(dependencies)

from aiflows.flow_cache import CACHING_PARAMETERS, clear_cache

CACHING_PARAMETERS.do_caching = True
# clear_cache()

from typing import Any, Dict, List

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger

from src import utils

import hydra
import os

from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf

from src.utils import general_helpers
from aiflows.flow_launchers import MultiThreadedAPILauncher, FlowLauncher

log = utils.get_pylogger(__name__)


def instantiate_flows(cfg: DictConfig) -> List[Dict[str, Any]]:
    if cfg.model.get("single_threaded", True):
        num_threads = 1
    else:
        num_threads = cfg.model.n_workers

    flow_instances = []
    for _ in range(num_threads):
        flow_with_interfaces = {
            "flow": hydra.utils.instantiate(cfg.flow, _recursive_=False, _convert_="partial"),
            "input_interface": (
                None
                if getattr(cfg, "input_interface", None) is None
                else hydra.utils.instantiate(cfg.input_interface, _recursive_=False)
            ),
            "output_interface": (
                None
                if getattr(cfg, "output_interface", None) is None
                else hydra.utils.instantiate(cfg.output_interface, _recursive_=False)
            ),
        }
        flow_instances.append(flow_with_interfaces)

    return flow_instances


def run_inference(cfg: DictConfig):
    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.experiment_path_to_continue is not None:
        log.warning(
            f"Predictions from previous run will be loaded and ignored. "
            f"Predictions loaded from: `{cfg.experiment_path_to_continue}`"
        )

    # Initialize the loggers
    log.info("Instantiating loggers...")
    loggers: List[Logger] = general_helpers.instantiate_loggers(cfg.get("logger"))

    assert cfg.output_dir is not None, "Path to the directory in which the predictions will be written must be given"
    cfg.output_dir = os.path.relpath(cfg.output_dir)
    log.info(f"Output directory: `{os.path.join(cfg.work_dir, cfg.output_dir)}`")

    # Initialize flow
    log.info(f"Instantiating flow <{cfg.flow._target_}>")
    flows = instantiate_flows(cfg)

    # Initialize the model
    log.info(f"Instantiating model <{cfg.model._target_}>")

    OmegaConf.set_struct(cfg, False)
    launch_prediction = cfg.model.pop("launch_prediction")
    OmegaConf.set_struct(cfg, True)

    if not launch_prediction:
        model = hydra.utils.instantiate(cfg.model, output_dir=cfg.output_dir, _convert_="partial")
        model.loggers = loggers

    if loggers:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(cfg, model, loggers)

    # Initialize the data module
    log.info(f"Instantiating data module <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)

    # Use the model collator's collate_fn, if defined, otherwise do not collate
    if getattr(model, "collator", None):
        datamodule.set_collate_fn(model.collator.collate_fn)
    else:
        datamodule.set_collate_fn(lambda x: x)  # no_collate

    # if model is a subclass of APIModel
    if not launch_prediction:
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()
        flat_dataloader = [sample for batch in dataloader for sample in batch]

        model.predict_dataloader(flat_dataloader, flows)
    else:
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()
        flat_dataloader = [sample for batch in dataloader for sample in batch]

        FlowLauncher.launch(data=flat_dataloader, flow_with_interfaces=flows[0])


@hydra.main(version_base="1.2", config_path="configs", config_name="inference_root")
def main(hydra_config: DictConfig):
    utils.run_task(hydra_config, run_inference, upload_predictions=True)


if __name__ == "__main__":
    main()
