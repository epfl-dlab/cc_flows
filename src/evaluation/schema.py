def id2data(name, problems_dataset, predictions_dataset, evaluation_output):
    # create mappings from id to data
    # the evaluation_output is initialized to a valid default, if it is empty
    id2problem_data = {problem["id"]: problem for problem in problems_dataset}
    id2pred_data = {pred["id"]: pred for pred in predictions_dataset}
    initialized_eval_output = initialize_and_assert_schema(name, id2pred_data, evaluation_output)
    id2eval_data = {data["id"]: data for data in initialized_eval_output}

    # gather all the ids for which we want to evaluate the predictions
    # the ids in the eval dataset and in the prediction dataset must be identical sets
    # and we need to have problem_data for all of these ids
    pred_ids = [_id for _id in id2pred_data]
    eval_ids = [_id for _id in id2eval_data]
    problem_ids = [_id for _id in id2problem_data]
    assert set(pred_ids) == set(eval_ids)
    for _id in pred_ids:
        assert _id in problem_ids

    return pred_ids, id2problem_data, id2pred_data, id2eval_data


def initialize_and_assert_schema(name, id2pred_data, evaluation_output):
    # initialize evaluation output, if it does not exist
    if len(evaluation_output) == 0:
        for _id, pred_data in id2pred_data.items():
            num_candidates = len(pred_data["inference_outputs"])
            initial_eval_output = {
                name: [{"evaluation_status": "failed submission"} for _ in range(num_candidates)],
                "id": _id,
            }
            evaluation_output.append(initial_eval_output)

    # assert that the existing output follows the schema in the README
    for eval_output in evaluation_output:
        _id = eval_output["id"]
        pred_data = id2pred_data[_id]
        num_candidates = len(pred_data["inference_outputs"])

        if not name in eval_output:
            eval_output[name] = [{"evaluation_status": "failed submission"} for _ in range(num_candidates)]

        assert len(eval_output[name]) == num_candidates

        for existing_eval in eval_output[name]:
            assert existing_eval["evaluation_status"] in [
                "failed submission",
                "failed collection",
                "submitted",
                "completed",
            ]

            if existing_eval["evaluation_status"] == "failed submission":
                assert "submission_url" not in existing_eval

            if existing_eval["evaluation_status"] == "submitted":
                assert "submission_url" in existing_eval

            if existing_eval["evaluation_status"] == "failed collection":
                assert "compilation_status" not in existing_eval
                assert "compilation_error_message" not in existing_eval

            if existing_eval["evaluation_status"] == "completed":
                assert "compilation_status" in existing_eval
                assert "compilation_error_message" in existing_eval
                assert "hidden_tests_results" in existing_eval

    return evaluation_output
