from pathlib import Path
import sys
import tempfile

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config.settings import settings
from app.services.ingest_service import IngestService
from app.services.qa_service import QAService
from app.ui.components import render_retrieval_results


st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Agentic RAG Demo")
st.caption("支持文档加载、切分、Embedding、向量检索、证据片段展示和 Agent 多轮检索回答。")

with st.sidebar:
    st.header("知识库构建")
    st.write(f"默认知识库目录：`{settings.knowledge_base_dir}`")

    if st.button("构建默认知识库"):
        with st.spinner("正在解析、切分并写入向量库..."):
            report = IngestService().ingest_directory_with_report(Path(settings.knowledge_base_dir))
        st.success(f"已索引 {report.indexed_chunks} 个文本块，加载 {report.loaded_documents} 个文档片段。")
        if report.skipped_files:
            st.warning("部分文件被跳过")
            st.dataframe(report.skipped_files)

    uploaded_files = st.file_uploader(
        "上传文档并加入知识库",
        type=["txt", "md", "pdf", "doc", "docx", "xlsx", "ppt", "pptx"],
        accept_multiple_files=True,
    )
    if st.button("索引上传文档") and uploaded_files:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths: list[Path] = []
            for uploaded_file in uploaded_files:
                path = Path(tmp_dir) / uploaded_file.name
                path.write_bytes(uploaded_file.getvalue())
                paths.append(path)

            with st.spinner("正在索引上传文档..."):
                report = IngestService().ingest_files(paths)
        st.success(f"已索引 {report.indexed_chunks} 个文本块，加载 {report.loaded_documents} 个文档片段。")
        if report.skipped_files:
            st.warning("部分文件被跳过")
            st.dataframe(report.skipped_files)

st.subheader("对话问答")
query = st.text_input("请输入你的问题")

col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("提问", type="primary")

if ask_clicked and query.strip():
    service = QAService()
    with st.spinner("Agent 正在分析问题、检索证据并生成答案..."):
        result = service.ask(query)

    st.subheader("回答")
    st.write(result.get("final_answer", ""))

    st.subheader("引用来源")
    st.write(result.get("citations", []))

    st.subheader("Agent 执行状态")
    st.json(
        {
            "question_type": result.get("question_type"),
            "iteration_count": result.get("iteration_count"),
            "subqueries": result.get("subqueries"),
            "should_refuse": result.get("should_refuse", False),
        }
    )

    st.subheader("检索证据片段")
    st.dataframe(render_retrieval_results(result.get("evidence_pool", [])), use_container_width=True)
