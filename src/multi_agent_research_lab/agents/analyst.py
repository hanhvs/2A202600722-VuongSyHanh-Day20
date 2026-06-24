"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent, record_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM = (
    "You are an Analyst. Extract key claims, compare viewpoints, and flag weak evidence. "
    "Keep the [n] citations from the research notes."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("analyst"):
            resp = self.llm.complete(
                SYSTEM,
                f"Query: {state.request.query}\nResearch notes:\n{state.research_notes or ''}",
            )
            state.analysis_notes = resp.content
            record_result(state, AgentName.ANALYST, resp)
        return state
