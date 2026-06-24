"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_html_report, render_markdown_report
from multi_agent_research_lab.graph.workflow import (
    MultiAgentWorkflow,
    run_multi_agent,
    run_single_agent,
)
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    configure_logging(get_settings().log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline (one LLM pass)."""

    _init()
    state = run_single_agent(query)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (supervisor -> researcher/analyst/writer)."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    state = MultiAgentWorkflow().run(state)
    console.print(Panel.fit(state.final_answer or "(no answer)", title="Multi-Agent Answer"))
    console.print(f"route: {' -> '.join(state.route_history)}")
    if state.errors:
        console.print(Panel.fit("\n".join(state.errors), title="Errors", style="red"))


@app.command()
def bench(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    out: Annotated[str, typer.Option("--out", help="Report basename under reports/")] = "benchmark_report",
) -> None:
    """Benchmark single-agent vs multi-agent; write HTML (with trace flow) + markdown."""

    _init()
    single_state, single = run_benchmark("single_agent", query, run_single_agent)
    multi_state, multi = run_benchmark("multi_agent", query, run_multi_agent)

    store = LocalArtifactStore()
    md_path = store.write_text(f"{out}.md", render_markdown_report([single, multi]))
    html_path = store.write_text(
        f"{out}.html",
        render_html_report([(single, single_state), (multi, multi_state)], query),
    )
    console.print(render_markdown_report([single, multi]))
    console.print(f"wrote {md_path}\nwrote {html_path}")


if __name__ == "__main__":
    app()
