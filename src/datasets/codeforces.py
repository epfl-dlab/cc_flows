import gzip
import json
import os

import jinja2

import jsonlines
from tqdm import tqdm

import src.utils as utils
import src.utils.evaluation_helpers
from src.datasets.abstract import AbstractDataset
from src.datasets.schema import assert_entry_format_codeforces

if __name__ == "__main__":
    log = utils.get_pylogger(__name__, stdout=True)
else:
    log = utils.get_pylogger(__name__)


class CodeforcesDataset(AbstractDataset):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

        self.io_example_formatter = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(
            self.params["io_example_template"]
        )
        self.explanation_formatter = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(
            self.params["explanation_template"]
        )

        self.ids_to_keep = None
        self.bucket_counter = {}
        self.bucket_id2datapoint_ids = None
        self.kwargs_to_filter_on = None
        self._setup_potential_filtering()

        self.data = None
        self._load_data()

        self.plans = None
        self._load_plans()

    def _load_plans(self):
        plans_id = self.params.get("plans_id", None)
        if plans_id is None:
            return

        plans_path = os.path.join(self.params["plans_dir"], f"{plans_id}.jsonl")
        plans_data = utils.general_helpers.read_jsonlines(plans_path)
        log.info(f"Loaded plans for {len(plans_data)} from {plans_path}")
        plans_data = {plan["id"]: plan for plan in plans_data}

        data_with_plans = []
        for dp in self.data:
            dp_plan = plans_data.get(dp["id"], None)
            if dp_plan is None:
                continue
            dp.update(**dp_plan)
            data_with_plans.append(dp)

        if len(data_with_plans) < len(self.data):
            log.warning(
                f"{len(self.data) - len(data_with_plans)} datapoints were discarded "
                f"as they didn't have an oracle plan."
            )

        self.data = data_with_plans

    def _setup_potential_filtering(self):
        ids_to_keep = set()

        if self.params.get("ids_to_keep", False):
            if self.params["ids_to_keep"] != "None":
                if isinstance(self.params["ids_to_keep"], str):
                    ids_to_keep = ids_to_keep.union(
                        _id.strip() for _id in self.params["ids_to_keep"].split(",") if _id.strip() != ""
                    )
                else:
                    ids_to_keep = ids_to_keep.union(set(self.params["ids_to_keep"]))

        if self.params.get("ids_to_discard", False):
            if self.params["ids_to_discard"] != "None":
                if isinstance(self.params["ids_to_discard"], str):
                    self.ids_to_discard = set(
                        _id.strip() for _id in self.params["ids_to_discard"].split(",") if _id.strip() != ""
                    )
                else:
                    self.ids_to_discard = set(self.params["ids_to_discard"])
        else:
            self.ids_to_discard = set()

        if self.params.get("ids_to_keep_file", False):
            if self.params["ids_to_keep_file"] != "None":
                with open(self.params["ids_to_keep_file"], "r") as f:
                    ids_to_keep = ids_to_keep.union(set([line.strip() for line in f]))

        if self.params.get("bucketing_id_to_filter_on", False):
            if self.params["bucketing_id_to_filter_on"] != "None":
                self.bucket_id2datapoint_ids = utils.evaluation_helpers.read_bucketing_data(
                    self.params["evaluation_buckets_dir"], self.params["bucketing_id_to_filter_on"]
                )

                for bucket_id in self.bucket_id2datapoint_ids:
                    if self.params["bucket_debug_k"]:
                        self.bucket_counter[bucket_id] = self.params["bucket_debug_k"]

                    if self.params["bucket_debug_k_first_k"]:
                        self.bucket_id2datapoint_ids[bucket_id] = self.bucket_id2datapoint_ids[bucket_id][
                            : self.params["bucket_debug_k"]
                        ]

        if len(ids_to_keep) > 0:
            self.ids_to_keep = ids_to_keep

        kwargs_to_filter_on = {}
        if self.params.get("kwargs_to_filter_on", False):
            if self.params["kwargs_to_filter_on"] != "None":
                if isinstance(self.params["kwargs_to_filter_on"], str):
                    kwargs_to_filter_on = json.loads(self.params["kwargs_to_filter_on"])
                else:
                    kwargs_to_filter_on = self.params["kwargs_to_filter_on"]

        if len(kwargs_to_filter_on) > 0:
            self.kwargs_to_filter_on = kwargs_to_filter_on

    def _load_data(self):
        path = os.path.join(
            self.params["load_dataset_params"]["data_dir"], f"{self.params['load_dataset_params']['split']}.jsonl.gz"
        )

        with gzip.open(path, "r") as f:
            num_datapoints = sum(1 for line in f)

        stream = gzip.open(path, "r")
        json_reader = jsonlines.Reader(stream)

        self.data = []

        num_problems_without_public_individual_tests = 0
        num_problems_with_non_unique_outputs = 0

        idx = -1
        for obj in tqdm(json_reader, total=num_datapoints, desc=f"Loading the data from: {path}"):
            idx += 1

            if obj["note"] == "":
                obj["note"] = None

            assert_entry_format_codeforces(obj)

            assert len(obj["hidden_tests_io"]) > 0
            assert len(obj["public_tests_io"]) > 0

            if (
                obj["public_tests_individual_io"] is None or len(obj["public_tests_individual_io"]) == 0
            ):  # the latter should not happen, but just in case
                num_problems_without_public_individual_tests += 1
                continue

            if obj["non_unique_output"]:
                # The local evaluator does not support non-unique outputs
                num_problems_with_non_unique_outputs += 1
                if self.params.get("keep_only_localeval_compatible", False):
                    continue

            obj["input_description"] = obj["input_description"].removeprefix("Input").strip()
            obj["output_description"] = obj["output_description"].removeprefix("Output").strip()

            if not self._to_keep(obj):
                continue

            self.data.append(obj)
            if self.params.get("debug", False) and len(self.data) >= self.params["debug_k"]:
                break

        stream.close()

        # Sanity check for the bucket_debug_k parameter
        if self.params["bucket_debug_k"]:
            for bucket_id, count in self.bucket_counter.items():
                if count > 0:
                    log.info(f"Bucket `{bucket_id}` has only {self.params['bucket_debug_k'] - count} datapoints")

        log.info(f"Loaded {len(self.data)} datapoints from {path}")
        log.info(
            f"Number of problems without public individual tests (filtered): "
            f"{num_problems_without_public_individual_tests}"
        )

        filter_problems_with_non_unique_outputs = self.params.get("keep_only_localeval_compatible", False) or (
            self.kwargs_to_filter_on is not None and self.kwargs_to_filter_on.get("non_unique_output", False)
        )
        log.info(
            f"Number of problems with non-unique outputs "
            f"({'filtered' if filter_problems_with_non_unique_outputs else 'not filtered'}): "
            f"{num_problems_with_non_unique_outputs}"
        )
        self.data = sorted(self.data, key=lambda x: x["contest"])

    def _to_keep(self, dp):
        if dp["id"] in self.ids_to_discard:
            return False

        if self.kwargs_to_filter_on:
            for k, v in self.kwargs_to_filter_on.items():
                if not isinstance(v, set):
                    # convert value to a set of items of the same type as dp[k]
                    if not isinstance(v, list):
                        v = [v]
                    if isinstance(dp[k], list):
                        if len(dp[k]) > 0:
                            v = {type(dp[k][0])(_v) for _v in v}
                        else:
                            v = set(v)
                    else:
                        v = {type(dp[k])(_v) for _v in v}
                    self.kwargs_to_filter_on[k] = v

                if isinstance(dp[k], list):
                    if not len(set(dp[k]).intersection(v)) > 0:
                        return False
                else:
                    if dp[k] not in v:
                        return False

        if self.ids_to_keep is not None and dp["id"] not in self.ids_to_keep:
            # filtering on ids and id is not in ids_to_keep
            return False

        if self.bucket_id2datapoint_ids is not None:
            # filtering based on a bucketing

            for bucket_id in self.bucket_id2datapoint_ids:
                if dp["id"] in self.bucket_id2datapoint_ids[bucket_id]:
                    # id is in the bucket_id bucket

                    if self.params["bucket_debug_k"] is None:
                        # bucket_debug_k is not used
                        return True
                    elif self.bucket_counter[bucket_id] > 0:
                        # bucket_debug_k is used and there are still datapoints to be sampled from this bucket
                        self.bucket_counter[bucket_id] -= 1
                        return True
                    else:
                        # bucket_debug_k is used and there are no more datapoints to be sampled from this bucket
                        return False

            # id is not in any bucket
            return False

        return True

    def __getitem__(self, idx):
        dp = self.data[idx]

        io_examples = []

        idx = 1
        for x, y in dp["public_tests_io"]:
            _input = "\n".join(x)
            kwargs = {"idx": idx, "input": _input, "output": y}
            io_examples.append(self.io_example_formatter.render(**kwargs))
            idx += 1

        formatted_io_examples = self.params["io_example_separator"].join(io_examples)

        if dp["note"] is None:
            dp["io_examples_and_explanation"] = formatted_io_examples
        else:
            formatted_note = self.explanation_formatter.render(note=dp["note"])
            dp["io_examples_and_explanation"] = "\n\n".join([formatted_io_examples, formatted_note])

        return dp

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    from pprint import pprint
    from omegaconf import OmegaConf

    data_dir = "data"

    path_to_config = "configs/dataset/codeforces.yaml"

    with open(path_to_config, "r") as f:
        cfg = OmegaConf.load(f)
    cfg["data_dir"] = data_dir
    cfg = OmegaConf.to_container(cfg, resolve=True)

    dataset = CodeforcesDataset(**cfg)
    print(len(dataset))
    pprint(dataset[0])
