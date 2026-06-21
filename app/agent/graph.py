from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    analyze_query,
    finalize_response,
    judge_evidence,
    plan_retrieval,
    retrieve_evidence,
    retry_or_expand,
    synthesize_answer,
)
from app.agent.policies import should_continue
from app.agent.state import AgentState


class AgentGraphBuilder:
    def build(self):
        graph = StateGraph(AgentState)

        graph.add_node("analyze_query", analyze_query)
        graph.add_node("plan_retrieval", plan_retrieval)
        graph.add_node("retrieve_evidence", retrieve_evidence)
        graph.add_node("judge_evidence", judge_evidence)
        graph.add_node("retry_or_expand", retry_or_expand)
        graph.add_node("synthesize_answer", synthesize_answer)
        graph.add_node("finalize_response", finalize_response)

        graph.set_entry_point("analyze_query")
        graph.add_edge("analyze_query", "plan_retrieval")
        graph.add_edge("plan_retrieval", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "judge_evidence")
        graph.add_conditional_edges(
            "judge_evidence",
            lambda state: "synthesize_answer" if state.get("enough_evidence") else ("retry_or_expand" if should_continue(state) else "finalize_response"),
            {
                "synthesize_answer": "synthesize_answer",
                "retry_or_expand": "retry_or_expand",
                "finalize_response": "finalize_response",
            },
        )
        graph.add_conditional_edges(
            "retry_or_expand",
            lambda state: "retrieve_evidence" if not state.get("should_refuse") else "finalize_response",
            {
                "retrieve_evidence": "retrieve_evidence",
                "finalize_response": "finalize_response",
            },
        )
        graph.add_edge("synthesize_answer", "finalize_response")
        graph.add_edge("finalize_response", END)

        return graph.compile()
