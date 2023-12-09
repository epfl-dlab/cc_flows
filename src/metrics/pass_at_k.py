import numpy as np
from src import utils
from src.metrics.abstract import AbstractMetric

log = utils.get_pylogger(__name__)


class PassAtK(AbstractMetric):
    def __init__(self, **kwargs):
        """
        code_evaluator_id: str,
        hidden_test_cases: bool,
        bucketing_id: Union[str, None],
        evaluation_buckets_dir: Union[str, None],
        k: int
        """
        super().__init__(**kwargs)

    @staticmethod
    def _get_id(code_evaluator_id, hidden_test_cases, bucketing_id, k, **kwargs):
        name = f"{code_evaluator_id}_pass_at_{k}"

        if hidden_test_cases:
            name += "_hidden"
        else:
            name += "_public"

        if bucketing_id is not None:
            name += f"_{bucketing_id}"

        return name

    @staticmethod
    def _estimator(n, c, k):
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def _compute_score(self, evaluation_output, tests_key):
        result = []

        for problem_eval_output in evaluation_output.data:
            eval_outputs = problem_eval_output[self.params["code_evaluator_id"]]

            total_sol_num = 0
            pass_sol_num = 0

            for candidate_sol_eval_output in eval_outputs:
                assert (
                    not self.params["code_evaluator_id"] == "online_judge"
                ) or "evaluation_status" in candidate_sol_eval_output, "Online judges must have evaluation_status"

                if (
                    "evaluation_status" in candidate_sol_eval_output
                    and candidate_sol_eval_output["evaluation_status"] != "completed"
                ):
                    log.error(
                        f"Problem {problem_eval_output['id']} has a candidate solution for which "
                        f"the evaluation status is `{candidate_sol_eval_output['evaluation_status']}` "
                        f"rather than completed."
                    )
                    continue

                test_statuses = [test["status"] for test in candidate_sol_eval_output[tests_key]]

                if len(test_statuses) == 0:
                    log.error(f"Problem {problem_eval_output['id']} has a candidate solution with no tests!")
                    continue

                pass_sol_num += np.all(test_statuses)
                total_sol_num += 1

            if total_sol_num == 1:
                log.warning("Calculating PassAtK with a single candidate solution.")
            elif total_sol_num == 0:
                log.error(f"Problem {problem_eval_output['id']} has no candidate solutions with completed evaluations.")
                continue

            if total_sol_num == self.params["k"]:
                log.warning(f"Calculating PassAtK with k = n = {total_sol_num}.")

            cur_res = self._estimator(n=total_sol_num, c=pass_sol_num, k=self.params["k"])
            result.append(cur_res)

        if len(result) == 0:
            raise ValueError("There are no problems with completed evaluations.")

        return np.mean(result)
