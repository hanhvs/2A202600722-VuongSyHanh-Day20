from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_html_report, render_markdown_report
from multi_agent_research_lab.graph.workflow import run_multi_agent


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report


def test_html_report_has_comparison_and_trace() -> None:
    state, metrics = run_benchmark(
        "multi_agent", "Explain multi-agent systems", lambda q: run_multi_agent(q, Settings(OPENAI_API_KEY=None))
    )
    html = render_html_report([(metrics, state)], query="Explain multi-agent systems")
    assert "Multi-Agent Research Benchmark" in html and "multi_agent" in html
    assert "Trace timeline" in html
    assert "researcher" in html  # route flow rendered
