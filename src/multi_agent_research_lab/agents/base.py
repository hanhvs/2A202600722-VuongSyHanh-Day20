"""Base agent contract and a small shared helper for recording results."""

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""


def record_result(
    state: ResearchState,
    agent: AgentName,
    response: LLMResponse,
    extra: dict | None = None,
) -> None:
    """Append an AgentResult and a trace event with token/cost metadata."""

    meta: dict = {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
    }
    if extra:
        meta.update(extra)
    state.agent_results.append(AgentResult(agent=agent, content=response.content, metadata=meta))
    state.add_trace_event(agent.value, meta)
