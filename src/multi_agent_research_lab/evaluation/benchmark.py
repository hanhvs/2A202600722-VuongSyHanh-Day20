"""Benchmark single-agent vs multi-agent."""

import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def _sum_meta(state: ResearchState, key: str) -> float:
    return sum(r.metadata.get(key) or 0.0 for r in state.agent_results)


def _total_cost(state: ResearchState) -> float:
    return round(_sum_meta(state, "cost_usd"), 6)


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of available sources actually cited as [n] in the final answer."""

    if not state.sources:
        return None
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", state.final_answer or "")}
    return round(len(cited & set(range(1, len(state.sources) + 1))) / len(state.sources), 2)


def _quality_proxy(state: ResearchState, coverage: float | None) -> float:
    """Automated 0-10 proxy. Replace with peer-review scores for the real rubric."""

    score = 5.0 if state.final_answer else 0.0
    score += 3.0 * (coverage or 0.0)
    score += min(2.0, len(state.final_answer or "") / 500)
    return round(min(score, 10.0), 1)


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a query through `runner` and compute latency/cost/coverage/quality."""

    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:  # a benchmark must record failures, not crash
        state = ResearchState.model_construct(errors=[f"runner crashed: {exc}"])  # type: ignore[arg-type]
        failed = True

    latency = perf_counter() - started
    failed = failed or bool(state.errors) or not state.final_answer
    coverage = _citation_coverage(state)
    cost = _total_cost(state)
    in_tok = int(_sum_meta(state, "input_tokens"))
    out_tok = int(_sum_meta(state, "output_tokens"))
    total_tok = in_tok + out_tok
    answer = state.final_answer or ""
    quality = _quality_proxy(state, coverage)

    details: dict[str, float | None] = {
        "latency_s": round(latency, 4),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": total_tok,
        "cost_usd": cost,
        "cost_per_1k_tokens": round(cost / total_tok * 1000, 6) if total_tok else None,
        "tokens_per_s": round(total_tok / latency, 1) if latency > 1e-6 else None,
        "agent_steps": len(state.agent_results),
        "supervisor_iterations": state.iteration,
        "sources_used": len(state.sources),
        "citation_coverage_pct": round(coverage * 100, 1) if coverage is not None else None,
        "answer_words": len(answer.split()),
        "answer_chars": len(answer),
        "quality_score": quality,
        "errors": len(state.errors),
        "failed": 1.0 if failed else 0.0,
        "live": 1.0
        if (state.agent_results and not failed
            and not any(r.content.startswith("[offline:") for r in state.agent_results))
        else 0.0,
    }
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=(
            f"agents={len(state.agent_results)} "
            f"citation_coverage={coverage if coverage is not None else 'n/a'} "
            f"failed={failed}"
        ),
        details=details,
    )
    return state, metrics
