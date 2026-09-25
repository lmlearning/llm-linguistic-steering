# Linguistic Steering in Large Language Models

[![Tests](https://github.com/lmlearning/llm-linguistic-steering/actions/workflows/tests.yml/badge.svg)](https://github.com/lmlearning/llm-linguistic-steering/actions/workflows/tests.yml)

**Measure how adjective choices influence model answers.** This research collection estimates adjective contributions with sampled coalitions, then analyzes direction, magnitude, volatility and correlations across questions.

Accompanying study: [Investigating Linguistic Steering: An Analysis of Adjectival Effects Across Large Language Model Architectures](https://openreview.net/forum?id=xN7NYpQeBm). Existing interactive exports cover [GPT-4o mini](4o-mini_dashboard.html), [o3](o3_dashboard.html) and [Phi-3](phi-3_dashboard.html); download an HTML file and open it in a browser to explore it.

## Generate a report without an API key

Use Python 3.11 in an activated virtual environment, from the repository root:

```bash
python -m pip install -r requirements-analysis.txt pytest
python analyze.py examples/demo_results.json --output-dir output/demo
python -m pytest -q tests
```

This creates a summary JSON, a ranked Markdown table and six PNG charts in `output/demo/`. The [included fixture](examples/demo_results.json) is explicitly synthetic and demonstrates the report pipeline; its values are not model measurements or paper results.

## From experiments to evidence

| Stage | Entry points |
| --- | --- |
| Estimate adjective effects | [estimate_importance.py](estimate_importance.py) for MMLU; [estimate_importance_arc.py](estimate_importance_arc.py) for ARC. |
| Parse model answers | [answer_parsing.py](answer_parsing.py), shared across estimators. |
| Build an analysis report | [analyze.py](analyze.py). |
| Compare experiments | [analyze_cross_benchmark.py](analyze_cross_benchmark.py), [analyze_variance.py](analyze_variance.py), [meta_analysis.py](meta_analysis.py). |
| Explore stored outputs | The three HTML dashboards linked above. |

Generation uses provider SDKs and external datasets and may incur API charges; the quick start only analyzes a local fixture. The estimation scripts expose their provider, sampling, seed and output options through their argument parsers. Dependencies for analysis are isolated from generation so viewing results does not require model-provider credentials.

## Analysis input contract

A results file contains:

- `parameters`: experiment metadata, including the model and sampling configuration.
- `adjective_mapping`: an ordered mapping of adjective names.
- `aggregated_results`: one entry per adjective with `mean_shapley`, `mean_abs_shapley` and `std_shapley`.
- Optional `per_question_results`: vectors in the **key order of adjective_mapping**. Keys such as `q_1_philosophy` carry the subject after the first two underscore-separated fields.

The fixture is a complete example. Custom vocabularies are supported; optional reference annotations are included only when present. Without question-level data the four aggregate plots are generated. Invalid or missing input causes a nonzero CLI exit.

## Answer parsing and result provenance

The parser accepts standalone letters, explicit answer/choice lines and LaTeX boxed answers, ignores `<think>` blocks, and uses `Z` for unparseable or ambiguous responses. ARC passes the question's choice count through to the parser, including five-choice questions. Ordinary letters inside prose or refusals never become guessed answers.

This is a change from the earlier permissive extraction protocol. Historical dashboard exports are retained. Regenerate an experiment before comparing results from different parser versions; the local demo does not revalidate published measurements.

## Validation and contribution

CI runs parser regressions, renders complete reports with custom vocabularies and executes the documented fixture command. Real model calls are outside this test suite.

For analysis bugs, supply a small JSON fixture and the failing command. For experimental results, record dataset selection, model/provider version, seed, sample counts, prompts and the parser commit so results can be compared meaningfully.

[Citation metadata](CITATION.cff) · [MIT license](LICENSE).
