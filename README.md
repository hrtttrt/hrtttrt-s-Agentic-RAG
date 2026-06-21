# Agentic RAG

本仓库目前包含两部分内容：

1. `app/`：基于 Streamlit 的 Agentic RAG 应用
2. `hrtttrt.github.io/`：用于 GitHub Pages 的项目展示页（Jekyll 静态站点）

## 当前部署方式

本仓库已经配置了 GitHub Actions 工作流：

- 工作流文件：`.github/workflows/deploy-pages.yml`
- 部署目标：`hrtttrt.github.io/` 下的静态网页
- 部署平台：GitHub Pages

需要注意：

- GitHub Pages **只能部署静态网页**。
- `app/` 下的 Streamlit / Python / Chroma / Embedding 应用 **不能直接运行在 GitHub Pages 上**。
- 因此当前推荐方案是：
  - 用 GitHub Pages 展示项目介绍、架构、效果截图、使用说明；
  - 将真正的 Streamlit 应用部署到支持 Python 服务的平台，再从展示页跳转过去。

## 仓库初始化前需要注意的事项

### 1. 不要提交本地模型、向量库和缓存文件

根目录已经补充了 `.gitignore`，默认会忽略以下内容：

- `.venv/`
- `.env`
- `chroma_db/`
- `reports/`
- `data/knowledge_base/processed/`
- `models/`
- 常见大模型文件后缀，如 `*.bin`、`*.pt`、`*.pth`、`*.ckpt`、`*.safetensors`

如果你还有额外下载的模型目录，请继续补充到根目录 `.gitignore`。

### 2. 不要提交真实 API Key

根目录 `.env` 一般只用于本地开发，不应提交到 GitHub。

建议：

- 只提交 `.env.example`
- 将真实密钥保留在本地
- 如果 `.env` 中已经填入真实 key，请在推送前确认没有被纳入 git 暂存区

## 路径说明

当前项目配置主要通过 `app/config/settings.py` 和 `.env` / `.env.example` 管理。

默认路径如下：

- `VECTOR_DB_PATH=chroma_db`
- `KNOWLEDGE_BASE_DIR=data/knowledge_base/raw`
- `PROCESSED_DATA_DIR=data/knowledge_base/processed`
- `REPORTS_DIR=reports`

这些路径都是**相对仓库根目录**设计的，不依赖原始拷贝来源目录，通常可以直接迁移。

## 本地运行 Streamlit 应用

安装依赖后运行：

```bash
streamlit run app/ui/streamlit_app.py
```

## GitHub Pages 发布步骤

1. 将仓库推送到 GitHub
2. 默认分支使用 `main` 或 `master`
3. 在 GitHub 仓库设置中启用 Pages
4. Source 选择 `GitHub Actions`
5. 推送后由 `.github/workflows/deploy-pages.yml` 自动构建并发布 `hrtttrt.github.io/`

## 后续推荐

如果你希望把“可交互演示”也发到线上，建议额外部署 `app/` 到以下任一平台：

- Streamlit Community Cloud
- Hugging Face Spaces
- Render
- Railway
- Azure / AWS / GCP

然后在 GitHub Pages 展示页中加入“在线体验”按钮跳转到应用地址。
