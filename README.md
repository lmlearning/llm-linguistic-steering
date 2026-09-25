# Linguistic Steering in Large Language Models

Research scripts and dashboards for exploring **how linguistic choices affect language-model behaviour**, including Shapley-value attribution, cross-benchmark comparisons and variance analysis.

## Explore the results

The repository includes HTML dashboards for [GPT-4o mini](4o-mini_dashboard.html), [o3](o3_dashboard.html) and [Phi-3](phi-3_dashboard.html). Download a dashboard and open it in a browser to inspect it; GitHub's file view displays the source.

## Analysis entry point

The main [analysis script](analyze.py) takes a JSON results file:

```bash
python analyze.py /path/to/results.json
```

It imports pandas, NumPy, Matplotlib and seaborn. Review its expected input structure before supplying a results file.

## Repository guide

- [analyze_cross_benchmark.py](analyze_cross_benchmark.py): cross-benchmark analysis.
- [analyze_variance.py](analyze_variance.py): variance analysis.
- [estimate_importance.py](estimate_importance.py) and [estimate_importance_arc.py](estimate_importance_arc.py): importance estimation.
- [correlation_ranks.py](correlation_ranks.py): rank correlations.
- [generate_charts.py](generate_charts.py): chart generation.
- [meta_analysis.py](meta_analysis.py): combined analysis.

This is an experimental research collection. Individual scripts may require experiment-specific inputs and path configuration.

## Analysis dependencies and regression tests

```bash
python -m pip install -r requirements-analysis.txt pytest
python -m pytest tests
```

The analysis accepts custom adjective vocabularies: optional plot annotations and
reference adjectives are used only when present in the supplied results. Each
`per_question_results` vector must follow the order of the keys in
`adjective_mapping`. The tests render the complete analysis with a small vocabulary,
including summary JSON, Markdown and PNG outputs, with and without question-level
data. They do not run model inference or regenerate the published experiments.

## Answer extraction for new experiments

Both estimation scripts use `answer_parsing.py`. It accepts explicit letter-only,
answer/choice lines and LaTeX boxed answers, ignores `<think>` blocks and returns
the existing `Z` failure marker for prose, refusals or ambiguous responses. ARC
passes each question's choice count through to the parser, including five-choice
questions. Ordinary letters inside words are never treated as answers.

This changes the scoring protocol for newly generated results. Existing dashboard
exports are retained; regenerate an experiment before comparing results produced
with different parser versions.

## License

See [LICENSE](LICENSE).
