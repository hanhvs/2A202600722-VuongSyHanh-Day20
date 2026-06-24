"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent, record_result
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

SYSTEM = (
    "You are a Writer. Synthesize a clear answer for the given audience using the analysis. "
    "Cite sources as [n] and end with a 'Sources' list."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("writer"):
            sources = "\n".join(
                f"[{i + 1}] {s.title} - {s.url}" for i, s in enumerate(state.sources)
            )
            resp = self.llm.complete(
                SYSTEM,
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n"
                f"Analysis:\n{state.analysis_notes or ''}\n\nSources:\n{sources}",
            )
            state.final_answer = resp.content
            record_result(state, AgentName.WRITER, resp)
        return state
