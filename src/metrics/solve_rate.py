import numpy as np
from src import utils
from src.metrics.abstract import AbstractMetric

log = utils.get_pylogger(__name__)


class SolveRate(AbstractMetric):
    def __init__(self, **kwargs):
        """
        code_evaluator_id: str,
        hidden_test_cases: bool,
        bucketing_id: Union[str, None],
        evaluation_buckets_dir: Union[str, None],
        test_level: bool
        """
        super().__init__(**kwargs)

    @staticmethod
    def _get_id(code_evaluator_id, test_level, hidden_test_cases, bucketing_id, **kwargs):
        if test_level:
            name = f"{code_evaluator_id}_test_pass_rate"
        else:
            name = f"{code_evaluator_id}_problem_solve_rate"

        if hidden_test_cases:
            name += "_hidden"
        else:
            name += "_public"

        if bucketing_id is not None:
            name += f"_{bucketing_id}"

        return name

    def _compute_score(self, evaluation_output, tests_key):
        result_solve_rate = []
        result_test_pass_rate = []

        for problem_eval_output in evaluation_output.data:
            eval_outputs = problem_eval_output[self.params["code_evaluator_id"]]

            psr = []
            tpr = []

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

                if self.params["test_level"]:
                    if "test_pass_rate" in candidate_sol_eval_output[tests_key][0]:
                        # We are relying on the test pass rate that was scrapped from an online judge
                        assert len(candidate_sol_eval_output[tests_key]) == 1
                        tpr.append(candidate_sol_eval_output[tests_key][0]["test_pass_rate"])
                        continue

                # Collect the status of each test
                test_statuses = [test["status"] for test in candidate_sol_eval_output[tests_key]]

                assert (
                    len(test_statuses) > 0
                ), f"Problem {problem_eval_output['id']} has a candidate solution with no tests!"

                # Compute the problem solve rate for the candidate solution
                psr.append(int(np.all(test_statuses)))

                # Compute the test pass rate for the candidate solution
                tpr.append(float(sum(test_statuses)) / len(test_statuses))

            if len(psr) == 0:
                log.error(f"Problem {problem_eval_output['id']} has no candidate solutions with completed evaluations.")
                continue

            result_solve_rate.append(np.mean(psr))
            result_test_pass_rate.append(np.mean(tpr))

        if self.params["test_level"]:
            return np.mean(result_test_pass_rate)

        return np.mean(result_solve_rate)
