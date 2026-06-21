from app.agent.graph import AgentGraphBuilder
from app.agent.state import AgentState
from app.config.settings import settings


class QAService:
    def __init__(self) -> None:
        self.graph = AgentGraphBuilder().build()

    def ask(self, query: str) -> AgentState:
        initial_state: AgentState = {
            "query": query,
            "iteration_count": 0,
            "max_iterations": settings.max_agent_iterations,
            "evidence_pool": [],
            "retrieved_docs": [],
        }
        return self.graph.invoke(initial_state)
