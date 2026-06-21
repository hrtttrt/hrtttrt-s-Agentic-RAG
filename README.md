# Agentic RAG

## Streamlit Cloud 部署

- 主文件路径：`app/ui/streamlit_app.py`
- Python 版本建议：`3.11`
- 依赖文件：`requirements.txt`

### 部署前准备

1. 将需要检索的知识库文档放到 `data/knowledge_base/raw/`
2. 在 Streamlit Cloud 的 `Secrets` 中填入 `.streamlit/secrets.toml.example` 对应配置
3. 建议云端使用：
   - `LLM_PROVIDER=openai_compatible`
   - `EMBEDDING_BACKEND=openai_compatible`

### 推荐 Secrets

至少配置以下字段：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL_NAME`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL_NAME`
- `EMBEDDING_DIMENSION`

如果你的聊天模型和嵌入模型来自同一兼容 OpenAI 的服务，也可以使用同一套 Key 与 Base URL。

### 启动行为

应用启动时会自动创建以下运行目录：

- `chroma_db/`
- `data/knowledge_base/raw/`
- `data/knowledge_base/processed/`
- `reports/`

进入页面后，可在侧边栏点击“构建默认知识库”完成索引。
