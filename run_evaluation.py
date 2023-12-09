from src.utils import hydra_custom_resolvers
from typing import List

from pytorch_lightning.loggers import Logger

from src import utils
from src.utils import general_helpers, evaluation_helpers

import hydra
import os

from omegaconf import DictConfig

import pytorch_lightning as pl
import wandb


log = utils.get_pylogger(__name__)


def run_evaluation(cfg: DictConfig):
    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Initialize the loggers
    log.info("Instantiating loggers...")
    loggers: List[Logger] = general_helpers.instantiate_loggers(cfg.get("logger"))
    if loggers:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(cfg, None, loggers)

    assert cfg.output_dir is not None, "Path to the directory in which the predictions will be written must be given"
    cfg.output_dir = os.path.relpath(cfg.output_dir)
    log.info(f"Output directory: {cfg.output_dir}")

    # Get the inference run's config &
    # Sync the predictions and the results from WandB in the exp_dir (downloads the data if is not found locally)
    api = wandb.Api()
    run = api.run(cfg.wandb_run_path)
    ir_wandb_config, ir_hydra_config, exp_dir = evaluation_helpers.sync_experiment_data(
        cfg.wandb_run_path, work_dir=cfg.work_dir
    )

    # Get the problem dataset (containing the metadata for the problems in the predictions dataset)
    problems_dataset = evaluation_helpers.get_dataset_used_in_run(ir_hydra_config, cfg.split_to_evaluate, cfg.data_dir)

    # Initialize the predictions dataset
    cfg.predictions_dataset.data_dir = general_helpers.get_predictions_dir_path(exp_dir)
    predictions_dataset = hydra.utils.instantiate(cfg.predictions_dataset, _recursive_=False)

    # Read the (potentially) existing evaluation output
    if cfg.local_results_cache_dir is not None:
        log.info(f"Loading evaluation output from local cache directory: {cfg.local_results_cache_dir}")
        evaluation_output = evaluation_helpers.read_evaluation_output(cfg.local_results_cache_dir)
    else:
        log.info("reading eval output from exp dir")
        evaluation_output = evaluation_helpers.read_evaluation_output(exp_dir)

    if cfg.complete_override:
        log.info("Complete override is set to True. The (potentially) existing evaluation output will be overwritten.")
        evaluation_output = []

        # keep only one id, for testing
        # pred_ids = [p['id'] for p in predictions_dataset]
        # _id = pred_ids[0]
        # problems_dataset = [p for p in problems_dataset if p['id'] == _id]
        # predictions_dataset = [p for p in predictions_dataset if p['id'] == _id]

        # evaluation_output = [
        #   {"id":_id, "online_judge":[
        #       {"evaluation_status": "completed", "compilation_status":True, "compilation_error_message":None, "hidden_tests_results":[{"status":True, "error_message":None} for _ in range(5)]},
        #       {"evaluation_status": "failed submission"}
        #   ]}]

    # Instantiate the code evaluator object(s)
    log.info(f"Instantiating the code evaluator(s)")
    code_evaluators = hydra.utils.instantiate(cfg.code_evaluator, _recursive_=True)

    # Evaluate the predictions
    for _, ce in code_evaluators.items():
        log.info(f"Evaluating {len(predictions_dataset)} predictions with {ce.name}.")
        evaluation_output = ce.evaluate_dataset(problems_dataset, predictions_dataset, evaluation_output, cfg.override)

    log.info(f"Writing the evaluation output to disk...")
    evaluation_helpers.write_evaluation_output(cfg.output_dir, evaluation_output)

    log.info(f"Output directory: {cfg.output_dir}")

    log.info(f"Uploading the evaluation output to WandB...")
    path_to_evaluation_output_file = os.path.join(cfg.output_dir, "evaluation_output.jsonl")
    general_helpers.upload_file_to_wandb(cfg.output_dir, path_to_evaluation_output_file)  # current run
    run.upload_file(path_to_evaluation_output_file, root=cfg.output_dir)  # original run


@hydra.main(version_base="1.2", config_path="configs", config_name="evaluation_root")
def main(hydra_config: DictConfig):
    utils.run_task(hydra_config, run_evaluation)


if __name__ == "__main__":
    main()

#
