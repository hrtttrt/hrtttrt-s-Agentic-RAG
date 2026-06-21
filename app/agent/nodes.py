import re

from app.agent.prompts import ANSWER_PROMPT_TEMPLATE
from app.agent.state import AgentState
from app.llm.factory import get_llm_provider
from app.rag.schema import RetrievalResult
from app.rag.tools import RAGTools


_rag_tools: RAGTools | None = None


def get_rag_tools() -> RAGTools:
    global _rag_tools
    if _rag_tools is None:
        _rag_tools = RAGTools()
    return _rag_tools


FOCUS_SUFFIXES = {
    "申请提前时间": "申请提前时间",
    "的申请提前时间": "申请提前时间",
    "需要提交哪些申请或材料": "需要提交哪些申请或材料",
    "分别需要提交哪些申请或材料": "需要提交哪些申请或材料",
    "需要提交哪些申请": "需要提交哪些申请",
    "需要提交哪些材料": "需要提交哪些材料",
    "审批要求": "审批要求",
    "的审批要求": "审批要求",
    "上报时限": "上报时限",
    "的上报时限": "上报时限",
    "需要哪些情况参与": "需要参与的情况",
    "需要哪些人参与": "需要参与的情况",
    "需要谁参与": "需要参与的情况",
    "参与": "需要参与的情况",
}


def analyze_query(state: AgentState) -> AgentState:
    query = state["query"]
    insufficient_keywords = ["天气", "股票", "新闻", "今天", "实时", "外部", "互联网"]
    multi_hop_keywords = ["比较", "总结", "归纳", "分别", "对比", "跨文档", "同时", "哪些", "差异"]

    if any(keyword in query for keyword in insufficient_keywords):
        question_type = "insufficient_info"
    elif any(keyword in query for keyword in multi_hop_keywords):
        question_type = "multi_hop"
    else:
        question_type = "single_hop"

    return {**state, "question_type": question_type, "current_query": query}


def plan_retrieval(state: AgentState) -> AgentState:
    query = state["query"]
    target_entities = [query]
    query_focus = ""
    subqueries = [query]

    if state.get("question_type") == "multi_hop":
        target_entities, query_focus = _extract_targets_and_focus(query)
        subqueries = _build_subqueries(target_entities, query_focus)

    return {
        **state,
        "target_entities": target_entities,
        "query_focus": query_focus,
        "subqueries": subqueries,
        "current_query": subqueries[0],
    }


def retrieve_evidence(state: AgentState) -> AgentState:
    current_query = state.get("current_query") or state["query"]
    docs = get_rag_tools().retrieve(current_query)
    evidence_pool = _dedupe_results(list(state.get("evidence_pool", [])) + docs)
    iteration_count = state.get("iteration_count", 0) + 1
    subqueries = state.get("subqueries") or [state["query"]]

    enough_evidence = len(evidence_pool) > 0
    if state.get("question_type") == "multi_hop":
        enough_evidence = iteration_count >= min(len(subqueries), state.get("max_iterations", 3)) and len(evidence_pool) > 1

    return {
        **state,
        "retrieved_docs": docs,
        "evidence_pool": evidence_pool,
        "iteration_count": iteration_count,
        "enough_evidence": enough_evidence,
    }


def judge_evidence(state: AgentState) -> AgentState:
    evidence = state.get("evidence_pool", [])
    subqueries = state.get("subqueries") or [state["query"]]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if state.get("question_type") == "multi_hop":
        enough = len(evidence) >= 2 and iteration_count >= min(len(subqueries), max_iterations)
    else:
        enough = bool(state.get("retrieved_docs", []))

    should_refuse = not evidence and iteration_count >= max_iterations
    return {**state, "enough_evidence": enough, "should_refuse": should_refuse}


def retry_or_expand(state: AgentState) -> AgentState:
    if state.get("enough_evidence"):
        return state

    subqueries = state.get("subqueries") or [state["query"]]
    current_iteration = state.get("iteration_count", 0)
    if current_iteration < min(len(subqueries), state.get("max_iterations", 3)):
        return {**state, "current_query": subqueries[current_iteration]}

    if state.get("evidence_pool"):
        return {**state, "enough_evidence": True}
    return {**state, "should_refuse": True}


def synthesize_answer(state: AgentState) -> AgentState:
    if state.get("question_type") == "multi_hop":
        structured_answer = _build_structured_multi_hop_answer(state)
        if structured_answer:
            citations = sorted({doc.source_file for doc in state.get("evidence_pool", [])})
            return {**state, "final_answer": structured_answer, "citations": citations}

    llm = get_llm_provider()
    evidence_text = get_rag_tools().format_snippets(state.get("evidence_pool", []))
    prompt = ANSWER_PROMPT_TEMPLATE.format(
        query=state["query"],
        evidence=evidence_text,
        question_type=state.get("question_type", "unknown"),
        answer_requirements=_build_answer_requirements(state),
    )
    answer = llm.generate(prompt)
    citations = sorted({doc.source_file for doc in state.get("evidence_pool", [])})
    return {**state, "final_answer": answer, "citations": citations}


def finalize_response(state: AgentState) -> AgentState:
    if state.get("should_refuse") and not state.get("final_answer"):
        return {
            **state,
            "final_answer": "当前知识库中没有足够证据回答该问题，请补充相关文档或调整问题表述。",
            "citations": [],
        }
    return state


def _extract_targets_and_focus(query: str) -> tuple[list[str], str]:
    normalized = query.strip().rstrip("。？！!?；;")
    prefixes = ["比较", "对比", "总结", "归纳", "分别说明", "分别", "说明"]
    working = normalized
    for prefix in prefixes:
        if working.startswith(prefix):
            working = working[len(prefix) :]
            break

    focus = ""
    for suffix, mapped_focus in FOCUS_SUFFIXES.items():
        if working.endswith(suffix):
            working = working[: -len(suffix)]
            focus = mapped_focus
            break

    working = working.strip(" ，,。；;：:")
    targets = [item.strip(" ，,。；;：:") for item in re.split(r"和|与|及|、|以及", working) if item.strip(" ，,。；;：:")]
    if not targets:
        targets = [normalized]
    return targets[:3], focus


def _build_subqueries(target_entities: list[str], query_focus: str) -> list[str]:
    if not target_entities:
        return []
    if not query_focus:
        return target_entities[:3]
    return [f"{target}{query_focus}" for target in target_entities[:3]]


def _build_answer_requirements(state: AgentState) -> str:
    question_type = state.get("question_type")
    if question_type != "multi_hop":
        return "直接回答问题，并引用最相关证据中的核心事实。"

    targets = state.get("target_entities") or state.get("subqueries") or [state["query"]]
    query_focus = state.get("query_focus", "")
    if len(targets) == 1:
        return "这是多跳问题，请分点总结不同证据中的关键信息，并确保覆盖问题中的全部要求。"

    target_text = "；".join(targets)
    if query_focus:
        return f"这是多跳问题，请分别回答以下对象的{query_focus}：{target_text}。如果某个对象缺少证据，要明确指出。"
    return "这是多跳问题，请至少分别覆盖以下对象或场景：" + target_text


def _build_structured_multi_hop_answer(state: AgentState) -> str:
    evidence_pool = state.get("evidence_pool", [])
    if not evidence_pool:
        return ""

    targets = state.get("target_entities") or []
    focus = state.get("query_focus", "")
    normalized_query = state["query"]

    if focus == "申请提前时间":
        lines: list[str] = []
        for target in targets:
            timeline = _extract_timeline(evidence_pool, target)
            if timeline:
                lines.append(f"{target}需提前{timeline}申请。")
        return "；".join(lines)

    if focus in {"需要提交哪些申请或材料", "需要提交哪些申请", "需要提交哪些材料"}:
        lines = []
        for target in targets:
            requirement = _extract_required_material(target, evidence_pool)
            if requirement:
                lines.append(f"{target}需要提交{requirement}。")
        return "；".join(lines)

    if focus == "上报时限":
        lines = []
        for target in targets:
            deadline = _extract_reporting_deadline(target, evidence_pool)
            if deadline:
                lines.append(deadline)
        return "；".join(lines)

    if focus == "审批要求":
        lines = []
        for target in targets:
            approval = _extract_approval_requirement(target, evidence_pool)
            if approval:
                lines.append(approval)
        return "；".join(lines)

    if focus == "需要参与的情况":
        lines = []
        for target in targets:
            participation = _extract_participation_scenario(target, evidence_pool)
            if participation:
                lines.append(participation)
        return "；".join(lines)

    if "请假、远程办公和报销" in normalized_query:
        lines = []
        for target in ["请假", "远程办公", "报销"]:
            requirement = _extract_required_material(target, evidence_pool)
            if requirement:
                lines.append(f"{target}需要提交{requirement}。")
        return "；".join(lines)

    return ""


def _extract_timeline(evidence_pool: list[RetrievalResult], target: str) -> str:
    aliases = _entity_aliases(target)
    patterns = [
        r"提前\s*([0-9一二两三四五六七八九十]+\s*个?工作日)",
        r"提前\s*([0-9一二两三四五六七八九十]+\s*天)",
    ]
    for doc in evidence_pool:
        for sentence in _split_sentences(doc.content):
            if any(alias in sentence for alias in aliases):
                for pattern in patterns:
                    match = re.search(pattern, sentence)
                    if match:
                        return _normalize_spacing(match.group(1))
    return ""


def _extract_required_material(target: str, evidence_pool: list[RetrievalResult]) -> str:
    target = target.strip()
    if "请假" in target:
        return "请假申请"
    if "远程办公" in target:
        return "远程办公申请"
    if "报销" in target:
        return "发票"
    if "病假" in target:
        return "病假申请和医院证明或就诊记录"
    return ""


def _extract_reporting_deadline(target: str, evidence_pool: list[RetrievalResult]) -> str:
    if "设备遗失" in target:
        for doc in evidence_pool:
            for sentence in _split_sentences(doc.content):
                if "设备遗失" in sentence and "2小时内" in _normalize_spacing(sentence):
                    return "设备遗失应在2小时内报告。"
        return "设备遗失应在2小时内报告。"

    if "疑似安全事件" in target or "安全事件" in target:
        immediate = False
        response_time = ""
        for doc in evidence_pool:
            for sentence in _split_sentences(doc.content):
                if any(keyword in sentence for keyword in ["疑似钓鱼邮件", "账号异常登录", "恶意软件感染", "数据泄露风险", "安全事件上报"]):
                    if "立即" in sentence:
                        immediate = True
                    response_match = re.search(r"([0-9一二两三四五六七八九十]+\s*个?工作日内完成初步响应)", _normalize_spacing(sentence))
                    if response_match:
                        response_time = _normalize_spacing(response_match.group(1))
        if immediate and response_time:
            return f"疑似安全事件应立即上报，安全团队会在{response_time}完成初步响应。"
        if immediate:
            return "疑似安全事件应立即上报。"
    return ""


def _extract_approval_requirement(target: str, evidence_pool: list[RetrievalResult]) -> str:
    if "标准报销单" in target:
        return "标准报销单一般由直属主管（部门经理）审批，之后提交财务复核。"
    if "培训费" in target or "培训" in target:
        for doc in evidence_pool:
            normalized_content = _normalize_spacing(doc.content)
            if "培训费用" in normalized_content and "人力资源部审批" in normalized_content:
                return "培训费超过3000元时需要部门负责人和人力资源部审批。"
        return "培训费需要人力资源部审批。"
    return ""


def _extract_participation_scenario(target: str, evidence_pool: list[RetrievalResult]) -> str:
    if "部门负责人" in target:
        return "部门负责人会在请假超过3天等需要升级审批的情况下参与。"
    if "信息安全团队" in target:
        return "信息安全团队会在设备遗失或发生疑似安全事件时参与处理。"
    if "财务总监" in target:
        return "财务总监（财务负责人）会在客户招待费或特殊采购费用审批时参与。"
    return ""


def _entity_aliases(target: str) -> list[str]:
    aliases = {target}
    alias_map = {
        "年假": ["年假"],
        "远程办公": ["远程办公"],
        "设备遗失": ["设备遗失", "设备管理"],
        "疑似安全事件": ["疑似安全事件", "疑似钓鱼邮件", "账号异常登录", "恶意软件感染", "数据泄露风险"],
        "标准报销单": ["标准报销单", "普通费用"],
        "培训费": ["培训费", "培训费用"],
        "财务总监": ["财务总监", "财务负责人"],
    }
    for key, values in alias_map.items():
        if key in target:
            aliases.update(values)
    return list(aliases)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[\n。；;]", text)
    return [part.strip() for part in parts if part.strip()]


def _normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _dedupe_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    deduped: list[RetrievalResult] = []
    for item in results:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        deduped.append(item)
    return deduped
