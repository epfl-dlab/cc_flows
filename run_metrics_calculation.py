from src.utils import hydra_custom_resolvers

from pytorch_lightning.loggers import LightningLoggerBase
import hydra
from omegaconf import DictConfig
from queue import Queue

import concurrent
import os

from typing import List, Dict, Union
import wandb
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

import src.utils.general_helpers as general_helpers
import src.utils.evaluation_helpers as evaluation_helpers
from src.utils.evaluation_helpers import Results, EvaluationOutput
from src import utils

log = utils.get_pylogger(__name__)


def get_score_from_metric(cfg, evaluation_output, metric_key, seed=None):
    metric = hydra.utils.instantiate(cfg.metrics[metric_key], _recursive_=True)

    score = metric.compute(evaluation_output, seed)

    return score


def get_bootstrap_run_scores(
    cfg,
    evaluation_output_instances_queue,
    results,
    starting_seed,
    num_workers=1,
):
    seed2score = results.get("bootstrap_runs_scores", {})

    run_scores_for_ci = []

    # Use one instance of the metric for each worker
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(cfg.bootstrap_n)):
            seed = starting_seed + i

            # ~~~ Read the precomputed result for the seed (if it has already computed) ~~~
            precomputed_score = read_precomputed_bootstrap_run_score(seed2score, seed, cfg.silent)
            if precomputed_score is not None:
                run_scores_for_ci.append(precomputed_score)
                continue

            # ~~~ Compute the score for the specific seed ~~~
            evaluation_output = evaluation_output_instances_queue.get()

            if cfg.debug:
                seed2score[seed] = get_score_from_metric(
                    cfg=cfg, evaluation_output=evaluation_output, metric_key=results["alias"], seed=seed
                )
            else:
                future = executor.submit(
                    get_score_from_metric,
                    cfg=cfg,
                    evaluation_output=evaluation_output,
                    metric_key=results["alias"],
                    seed=seed,
                )
                seed2score[seed] = future.result()

            # ~~~ Log the score (if not executing silently) ~~~
            if not cfg.get("silent", False):
                score = seed2score[seed]
                if isinstance(score, dict):
                    score = np.mean(list(score.values()))
                # Below supoorts nested dicts, but when would we expect to have buckets of buckets?
                #     is_subdict = False
                #     for k, v in score.items():
                #         if isinstance(v, dict):
                #             is_subdict = True
                #     if is_subdict == False:
                #         score = np.mean(list(score.values()))
                #     else:
                #         score_list = []
                #         for k, v in score.items():
                #             score_list.append(np.mean(list(v.values())))
                #         score = np.mean(score_list)
                log.info(f"Score for seed {seed}: {score * 100:.2f}%.")

            # ~~~ Add the score to the list of score that will be used to compute the confidence interval ~~~
            run_scores_for_ci.append(seed2score[seed])
            evaluation_output_instances_queue.put(evaluation_output)

    # ~~~ Update the cache of precomputed results if results for more runs were computed ~~~
    if len(results.get("bootstrap_runs_scores", {})) < len(seed2score):
        results["bootstrap_runs_scores"] = seed2score

    return run_scores_for_ci


def _check_if_bootstrap_runs_are_already_computed(seed2score, starting_seed, bootstrap_n):
    for seed in range(starting_seed, starting_seed + bootstrap_n):
        if seed not in seed2score:
            return False

    return True


def read_precomputed_bootstrap_run_score(seed2score, seed, silent=False):
    if seed in seed2score:
        if not silent:
            log.info(f"Score for seed {seed} was already computed.")
        return seed2score[seed]
    elif str(seed) in seed2score:
        if not silent:
            log.info(f"Score for seed {seed} was already computed.")
        return seed2score[str(seed)]

    return None


def run_calculate_metrics(cfg: DictConfig) -> Dict[str, Dict[str, Union[str, float, List[float]]]]:
    """Contains the code for calculating metrics based on evaluation outputs.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Dict[str, Dict[str, Union[str, float, List[float]]]]: Dictionary containing the results of the evaluation.
    """
    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Initialize the loggers
    log.info("Instantiating loggers...")
    loggers: List[LightningLoggerBase] = general_helpers.instantiate_loggers(cfg.get("logger"))
    if loggers:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(cfg, None, loggers)

    assert cfg.output_dir is not None, "Path to the directory in which the predictions will be written must be given"
    cfg.output_dir = os.path.relpath(cfg.output_dir)
    log.info(f"Output directory: {os.path.join(cfg.work_dir, cfg.output_dir)}")

    api = wandb.Api()
    run = api.run(cfg.wandb_run_path)

    er_wandb_config, er_hydra_config, evaluation_dir = evaluation_helpers.sync_experiment_data(
        cfg.wandb_run_path,
        work_dir=cfg.work_dir,
        sync_predictions=False,
        sync_evaluation_output=True,
        replace_evaluation_output=cfg.replace_evaluation_output,
        sync_results=True,
        replace_results=True,
    )
    problems_dataset = evaluation_helpers.get_dataset_used_in_run(er_hydra_config, cfg.split_to_evaluate, cfg.data_dir)
    evaluation_output = EvaluationOutput(evaluation_dir, problems_dataset=problems_dataset)
    results = Results(evaluation_dir)
    if cfg.complete_override:
        results.data = {}

    evaluation_outputs_instances_queue = None
    metrics = hydra.utils.instantiate(cfg.metrics, _recursive_=True)

    log.info(f"Calculating metrics...")
    for metric_name, metric in metrics.items():
        if not cfg.override and metric.id in results.data and results.data[metric.id] != {}:
            log.info(f"Skipped -- {metric_name} -- as it is already present in the results json.")
        else:
            results.data[metric.id] = {"alias": metric_name, "score": metric.compute(evaluation_output)}
            score = results.get_score(metric.id, reduce_buckets_to_mean=True)
            log.info(f"[{metric.id}] Score: {score * 100:.2f}%")

        if cfg.get("bootstrap_n", None):
            bootstrap_n = cfg.bootstrap_n
            starting_seed = cfg.seed

            seed2score = results.get_bootstrap_runs_scores(metric.id)

            if not cfg.override:
                all_done = _check_if_bootstrap_runs_are_already_computed(seed2score, starting_seed, bootstrap_n)
                if all_done:
                    log.info(f"Skipped the bootstrap for {metric_name}, as scores for runs are already computed.")
                    continue

            log.info(f"Getting bootstrap samples for {metric_name}")

            if evaluation_outputs_instances_queue is None:
                evaluation_outputs_instances_queue = Queue(cfg.num_workers)
                for _ in range(cfg.num_workers):
                    evaluation_outputs_instances_queue.put(
                        EvaluationOutput(evaluation_dir, problems_dataset=problems_dataset)
                    )

            bootstrap_run_scores = get_bootstrap_run_scores(
                cfg, evaluation_outputs_instances_queue, results.data[metric.id], starting_seed, bootstrap_n
            )
            # ~~~ [Sanity check] Construct confidence intervals (CIs) from the bootstrap run scores ~~~
            if isinstance(bootstrap_run_scores[0], dict):
                bootstrap_run_scores = [
                    np.mean(list(bucket_id2score.values())) for bucket_id2score in bootstrap_run_scores
                ]

            # ~~~ Percentile based CI ~~~
            lower, mean_perc_based, upper = evaluation_helpers.get_percentile_based_ci(bootstrap_run_scores, 0.95)
            log.info(
                f"[{metric_name}] Percentile based confidence interval: "
                f"[{lower * 100:.2f}, {mean_perc_based * 100:.2f}, {upper * 100:.2f}]"
            )

            # ~~~ Standard deviation based CI ~~~
            lower, mean_std_based, upper = evaluation_helpers.get_std_based_ci(bootstrap_run_scores)
            log.info(
                f"[{metric_name}] Standard deviation based confidence interval: "
                f"[{lower * 100:.2f}, {mean_std_based * 100:.2f}, {upper * 100:.2f}]"
            )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    log.info(f"Writing the results to disk...")
    evaluation_helpers.write_results(cfg.output_dir, results.data)

    log.info(f"Uploading the results to wandb...")
    path_to_results_file = os.path.join(cfg.output_dir, "results.json")
    general_helpers.upload_file_to_wandb(cfg.output_dir, path_to_results_file)  # current run
    run.upload_file(path_to_results_file, root=cfg.output_dir)  # original run


@hydra.main(version_base="1.2", config_path="configs", config_name="metrics_calculation")
def main(hydra_config: DictConfig):
    utils.run_task(hydra_config, run_calculate_metrics)


if __name__ == "__main__":
    main()
