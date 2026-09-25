"""Parse explicit multiple-choice answers without guessing from prose."""

import re


def extract_answer_letter(content: str, num_choices: int = 4) -> str:
    """Return an explicit choice, or ``Z`` when no valid answer is present.

    Accepted forms are a standalone letter (optionally parenthesized), a final
    answer/option/choice line, or a LaTeX boxed letter. Reasoning inside think
    blocks is ignored. The last boxed answer takes precedence, otherwise the
    last explicit answer line wins. Prose, refusals and ambiguous choices are
    unparseable, not guessed answers. ``Z`` is reserved as the failure marker.
    """
    if not isinstance(num_choices, int) or isinstance(num_choices, bool) or not 2 <= num_choices <= 25:
        raise ValueError("num_choices must be an integer between 2 and 25")
    if not isinstance(content, str):
        return "Z"
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL | re.IGNORECASE)
    choices = set(chr(65 + i) for i in range(num_choices))
    boxed = re.findall(r"\\boxed\{\s*([^{}]*)\s*\}", cleaned, flags=re.IGNORECASE)
    if boxed:
        answer = boxed[-1].strip().upper()
        return answer if answer in choices else "Z"
    pattern = re.compile(
        r"(?:(?:the\s+)?(?:correct\s+|final\s+)?(?:answer|option|choice)\s*(?:is\s+|:\s*)?)?"
        r"(?:\(([A-Z])\)|([A-Z]))[.!]?",
        flags=re.IGNORECASE,
    )
    for line in reversed(cleaned.splitlines()):
        match = pattern.fullmatch(line.strip())
        if match:
            answer = (match.group(1) or match.group(2)).upper()
            return answer if answer in choices else "Z"
    return "Z"
