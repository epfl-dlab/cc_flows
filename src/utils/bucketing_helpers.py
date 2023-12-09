import itertools
from collections import defaultdict
from typing import List, Dict

from src import utils

from src.datasets import AbstractDataset

log = utils.get_pylogger(__name__)


def get_bucketing_file_id(dataset_name, kwargs_to_filter_on: Dict, keys_to_bucket_on: List[str], min_bucket_size: int):
    name = f"{dataset_name}__"
    if kwargs_to_filter_on is not None:
        for key, value in kwargs_to_filter_on.items():
            name += f"{key}_{value}_"
        name += "__"
    name += f"{'_'.join(keys_to_bucket_on)}"
    name += f"__min-bucket-size_{min_bucket_size}"

    return name


def bucket_datapoints(dataset: AbstractDataset, keys_to_bucket_on: List[str], return_ids_only: bool = False):
    bucket2datapoints = defaultdict(list)
    keys_to_bucket_on = sorted(keys_to_bucket_on)

    for dp in dataset:
        dp_values = [dp[key] if isinstance(dp[key], list) else [dp[key]] for key in keys_to_bucket_on]
        for val_combination in itertools.product(*dp_values):
            key = tuple(val_combination)
            dps = bucket2datapoints[key]
            if return_ids_only:
                dps.append(dp["id"])
            else:
                dps.append(dp)
            bucket2datapoints[key] = dps

    return dict(bucket2datapoints)
