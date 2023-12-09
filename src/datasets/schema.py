from typing import List, Tuple, Dict


def assert_test_format_codeforces(tests: List[Tuple[List[str], str]]):
    assert isinstance(tests, list) or tests is None
    if tests is None:
        return
    for test in tests:
        assert isinstance(test, list)
        assert len(test) == 2
        inputs, outputs = test
        assert isinstance(inputs, list)
        assert isinstance(outputs, str)
        for input in inputs:
            assert isinstance(input, str)


def assert_entry_format_codeforces(obj: Dict):
    # each data point must follow the same schema
    assert isinstance(obj["id"], str)  # contest + problem_name = id, will not change when formatting changes
    assert isinstance(obj["id_hash"], str)  # hashsum of all entries, any change to obj will change this
    assert isinstance(obj["contest"], int)
    assert isinstance(obj["problem_name"], str)
    assert isinstance(obj["problem_url"], str)
    assert isinstance(obj["solution_url"], str)

    assert isinstance(obj["header"], str)
    assert isinstance(obj["problem_description"], str)
    assert isinstance(obj["input_description"], str)
    assert isinstance(obj["output_description"], str)
    assert isinstance(obj["note"], str) or obj["note"] is None

    assert isinstance(obj["difficulty"], int)
    assert isinstance(obj["tags"], list)
    assert isinstance(obj["working_solution"], str)  # can be empty

    assert_test_format_codeforces(obj["public_tests_io"])
    assert_test_format_codeforces(obj["public_tests_individual_io"])
    assert_test_format_codeforces(obj["hidden_tests_io"])


def assert_entry_format_leetcode(obj: Dict):
    # each data point must follow the same schema
    assert isinstance(obj["id"], str)  # contest + problem_name = id, will not change when formatting changes
    assert isinstance(obj["id_hash"], str)  # hashsum of all entries, any change to obj will change this
    assert isinstance(obj["index"], int)
    assert isinstance(obj["problem_name"], str)
    assert isinstance(obj["problem_url"], str)

    assert isinstance(obj["problem_description"], str)
    assert isinstance(obj["constraints"], str)
    assert isinstance(obj["python_stub"], str)
    assert isinstance(obj["difficulty"], str) and obj["difficulty"] in {"easy", "medium", "hard"}
