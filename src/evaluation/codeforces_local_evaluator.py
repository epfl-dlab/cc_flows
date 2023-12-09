from typing import Dict

from .testing_utils_codeforces import evaluate_solution_for_problem
from src import utils

log = utils.get_pylogger(__name__)


class CodeforcesLocalEvaluator:
    name = "local_evaluator"

    def __init__(self, eval_helper_params, num_workers=1, debug=False):
        self.num_workers = num_workers
        self.debug = debug
        self.eval_helper_params = eval_helper_params
        self.eval_helper_params["debug"] = debug

    def evaluate_problem(self, problem_data, pred_data) -> Dict:
        """
        Required input fields:
        problem_data:
            - id: id of the problem in our dataset
            - hidden_tests_io: Optional(list of tuples): hidden test ceases, each of which is an input-output tuple
            - public_tests_io: Optional(list of tuples): public test ceases, each of which is an input-output tuple
        pred_data:
            - id: id of the problem in our dataset
            - candidate_solutions: list of candidate solutions for the problem
        See the readme for the output format of this function.
        """
        assert pred_data["id"] == problem_data["id"]

        if self.debug:
            log.info(f"Number of solutions: {len(pred_data['candidate_solutions'])}")

        evaluation_results_per_candidate_solutions = []

        for solution in pred_data["candidate_solutions"]:
            evaluation_results_per_candidate_solutions.append(
                self.evaluate_solution(
                    candidate_solution=solution,
                    hidden_tests_io=problem_data["hidden_tests_io"],
                    public_tests_io=problem_data["public_tests_io"],
                )
            )

        complete_evaluation_output = {
            "id": pred_data["id"],
            self.name: evaluation_results_per_candidate_solutions,
        }
        return complete_evaluation_output

    def evaluate_solution(self, candidate_solution, hidden_tests_io, public_tests_io):
        return evaluate_solution_for_problem(
            candidate_solution, hidden_tests_io, public_tests_io, **self.eval_helper_params
        )

    def evaluate_dataset(self, problems_dataset, predictions_dataset, existing_evaluation_output=[], override=False):
        id2problem_data = {problem["id"]: problem for problem in problems_dataset.data}
        id2pred_data = {
            pred["id"]: {
                "id": pred["id"],
                "candidate_solutions": [
                    predictions_dataset.get_prediction(output) for output in pred["inference_outputs"]
                ],
            }
            for pred in predictions_dataset
        }
        id2eval_output_data = {eval_output["id"]: eval_output for eval_output in existing_evaluation_output}

        for _id in id2pred_data:
            if _id in id2eval_output_data and self.name in id2eval_output_data[_id] and not override:
                log.info(f"Skipping evaluation for problem {_id} as it already exists.")
                continue
            eval_output = id2eval_output_data.get(_id, {})
            eval_output.update(self.evaluate_problem(id2problem_data[_id], id2pred_data[_id]))
            id2eval_output_data[_id] = eval_output

        evaluation_outputs = list(id2eval_output_data.values())
        evaluation_outputs.sort(key=lambda x: x["id"])
        return evaluation_outputs
