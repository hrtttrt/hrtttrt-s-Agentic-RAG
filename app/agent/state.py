from typing import Literal, TypedDict

from app.rag.schema import RetrievalResult


QuestionType = Literal["single_hop", "multi_hop", "insufficient_info"]


class AgentState(TypedDict, total=False):
    query: str
    question_type: QuestionType
    subqueries: list[str]
    target_entities: list[str]
    query_focus: str
    current_query: str
    retrieved_docs: list[RetrievalResult]
    evidence_pool: list[RetrievalResult]
    iteration_count: int
    max_iterations: int
    enough_evidence: bool
    should_refuse: bool
    final_answer: str
    citations: list[str]
