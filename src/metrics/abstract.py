from abc import ABC

from typing import Union
from src import utils

log = utils.get_pylogger(__name__)


class AbstractMetric(ABC):
    def __init__(self, **kwargs):
        self.id = self._get_id(**kwargs)
        self.params = kwargs

        self.bucket_id2datapoint_ids = None
        self._load_bucketing_data()

    def _load_bucketing_data(self):
        if self.params["bucketing_id"] is None:
            return

        self.bucket_id2datapoint_ids = utils.evaluation_helpers.read_bucketing_data(
            self.params["evaluation_buckets_dir"], self.params["bucketing_id"]
        )

    @staticmethod
    def _get_id(**kwargs):
        raise NotImplementedError()

    def compute(self, evaluation_output, seed=None):
        if self.bucket_id2datapoint_ids is not None:
            return self._compute_per_bucket_performance(evaluation_output, seed=seed)

        solve_rate = self._compute(evaluation_output, seed=seed)

        return solve_rate

    def _compute(self, evaluation_output, seed=None):
        # ~~ Concerning bootstrapping ~~
        original_data = evaluation_output.data

        if seed is not None:
            if len(original_data) == 1:
                log.info("Bootstrapping is enabled but the evaluation output contains only one problem.")

            evaluation_output.data = evaluation_output.get_bootstrapped_data(seed=seed)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        tests_key = "hidden_tests_results" if self.params["hidden_test_cases"] else "public_tests_results"
        score = self._compute_score(evaluation_output, tests_key)

        # ~~ Concerning bootstrapping ~~
        evaluation_output.data = original_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        return score

    def _compute_score(self, evaluation_output, tests_key):
        raise NotImplementedError()

    def _compute_per_bucket_performance(self, evaluation_output, seed):
        bucket_id2score = {}

        for bucket_id, datapoint_ids in self.bucket_id2datapoint_ids.items():
            # ~~ Concerning bucketing ~~
            original_data = evaluation_output.data
            evaluation_output.data = evaluation_output.get_filtered_data(ids_to_keep=datapoint_ids)

            if len(evaluation_output.data) == 0:
                raise ValueError(f"Bucket {bucket_id} is empty.")

            # Convenient for debugging but can hide bugs for real runs
            # if len(evaluation_output.data) == 0:
            #     evaluation_output.data = original_data
            #     continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~

            bucket_id2score[bucket_id] = self._compute(evaluation_output, seed=seed)

            # ~~ Concerning bucketing ~~
            evaluation_output.data = original_data
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~

        return bucket_id2score
