from app.agent.state import AgentState


def should_continue(state: AgentState) -> bool:
    if state.get("enough_evidence"):
        return False
    if state.get("should_refuse"):
        return False
    return state.get("iteration_count", 0) < state.get("max_iterations", 3)
