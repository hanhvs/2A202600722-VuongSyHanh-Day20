"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int = 6) -> None:
        self.max_iterations = max_iterations

    def decide(self, state: ResearchState) -> str:
        """Return the next route based on which fields are still missing."""

        if state.errors:
            return "done"  # fail fast: a worker errored, stop instead of looping
        if state.iteration >= self.max_iterations:
            return "done"  # guardrail against infinite loops
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        route = self.decide(state)
        state.record_route(route)
        state.add_trace_event("supervisor", {"route": route, "iteration": state.iteration})
        return state
