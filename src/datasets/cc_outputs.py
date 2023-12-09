from typing import Dict

from aiflows.datasets import OutputsDataset
import src.utils as utils


if __name__ == "__main__":
    log = utils.get_pylogger(__name__, stdout=True)
else:
    log = utils.get_pylogger(__name__)


class CompetitiveCodingOutputsDataset(OutputsDataset):
    @staticmethod
    def get_prediction(inference_output: Dict):
        output_data = inference_output["data"]["output_data"]
        return output_data["code"]

    @staticmethod
    def get_plan(inference_output: Dict):
        output_data = inference_output["data"]["output_data"]
        return output_data["plan"]


if __name__ == "__main__":
    data_dir = ""
    seed = 123
