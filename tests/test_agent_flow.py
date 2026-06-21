from app.agent.nodes import analyze_query


def test_analyze_query_multi_hop() -> None:
    state = {"query": "请比较两个方案的差异"}
    result = analyze_query(state)
    assert result["question_type"] == "multi_hop"
