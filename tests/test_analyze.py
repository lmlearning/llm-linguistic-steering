import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from analyze import analyze_and_visualize_complete


@pytest.mark.parametrize("with_questions", [True, False])
def test_custom_vocabulary_generates_complete_analysis(tmp_path, monkeypatch, with_questions):
    monkeypatch.chdir(tmp_path)
    adjectives = ["curious", "concise", "bold"]
    data = {
        "parameters": {"model": "test-fixture"},
        "adjective_mapping": {adj: i for i, adj in enumerate(adjectives)},
        "aggregated_results": {
            "curious": {"mean_shapley": 0.2, "mean_abs_shapley": 0.3, "std_shapley": 0.2},
            "concise": {"mean_shapley": -0.1, "mean_abs_shapley": 0.2, "std_shapley": 0.1},
            "bold": {"mean_shapley": 0.1, "mean_abs_shapley": 0.15, "std_shapley": 0.3},
        },
    }
    if with_questions:
        data["per_question_results"] = {
            "q_1_philosophy": [0.2, -0.1, 0.3],
            "q_2_philosophy": [0.4, 0.1, -0.1],
            "q_3_college_physics": [-0.1, 0.3, 0.1],
        }
    source = tmp_path / "custom.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    # Keep the full plotting path, including real PNG rendering.
    try:
        analyze_and_visualize_complete(str(source))
        output = tmp_path / "custom_analysis"
        summary = json.loads((output / "custom_summary.json").read_text())
        assert summary["parameters"] == data["parameters"]
        assert summary["analysis_summary"]["top_10_positive_adjectives"] == ["curious", "bold", "concise"]
        table = (output / "custom_summary_table.md").read_text()
        assert all(adjective in table for adjective in adjectives)
        plots = sorted(output.glob("plot_*.png"))
        assert len(plots) == (6 if with_questions else 4)
        assert all(plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for plot in plots)
        assert plt.get_fignums() == []
    finally:
        plt.close("all")
