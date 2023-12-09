from src import utils
from collections import defaultdict

log = utils.get_pylogger(__name__)


class TemporalMetric:
    def __init__(
        self,
        **kwargs,
    ):
        """
        metric: AbstractMetric
        """
        self.id = self._get_id(**kwargs)
        self.params = kwargs

        self.metric = kwargs["metric"]
        del kwargs["metric"]

        assert self.metric.params["bucketing_id"] is not None, "A bucketing, mapping dates to problem ids is required."

    @staticmethod
    def _get_id(metric, **kwargs):
        return f"temporal_{metric.id}"

    def compute(self, evaluation_output, seed=None):
        return self.metric.compute(evaluation_output=evaluation_output, seed=seed)
