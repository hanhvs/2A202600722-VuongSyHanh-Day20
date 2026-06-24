"""Optional critic agent: a cheap citation-coverage check (no LLM call)."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span


class CriticAgent(BaseAgent):
    """Flags an answer that has no citations. Bonus, not on the default route."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("critic"):
            answer = state.final_answer or ""
            cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
            covered = bool(cited) and len(state.sources) > 0
            if not covered:
                state.errors.append("critic: final answer has no source citations")
            meta = {"citations": cited, "num_sources": len(state.sources), "ok": covered}
            state.agent_results.append(
                AgentResult(agent=AgentName.CRITIC, content=f"citations={cited}", metadata=meta)
            )
            state.add_trace_event("critic", meta)
        return state
