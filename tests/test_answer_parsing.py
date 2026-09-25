import pytest

from answer_parsing import extract_answer_letter


@pytest.mark.parametrize("response,expected", [
    ("A", "A"), ("  b\n", "B"), ("(C)", "C"),
    ("Answer: D.", "D"), ("The correct answer is B.", "B"),
    ("Final answer: (A)", "A"), ("Choice C", "C"),
    (r"The result is \boxed{c}", "C"),
    ("Some reasoning about B.\nD", "D"),
    ("<think>Answer: A\n</think>\nB", "B"),
    ("<think>Answer: A", "Z"),
    (r"\boxed{A} revised to \boxed{D}", "D"),
    (r"\boxed{A} revised to \boxed{E}", "Z"),
    ("I cannot answer this question.", "Z"),
    ("The evidence is inconclusive.", "Z"),
    ("A or B", "Z"), ("Answer: C or D", "Z"),
    ("AB", "Z"), ("E", "Z"), ("", "Z"), (None, "Z"),
])
def test_explicit_answers_and_nonanswers(response, expected):
    assert extract_answer_letter(response) == expected


def test_arc_can_accept_fifth_choice():
    assert extract_answer_letter("Answer: E", num_choices=5) == "E"
    assert extract_answer_letter("C", num_choices=2) == "Z"


@pytest.mark.parametrize("count", [0, 1, 26, 3.5, True])
def test_invalid_choice_count(count):
    with pytest.raises(ValueError, match="num_choices"):
        extract_answer_letter("A", count)
