import contextlib
import io
import logging
import math
from typing import Any

from app.config.settings import settings


class RagasEvaluator:
    def __init__(self) -> None:
        self.enabled = settings.enable_ragas_eval

    def evaluate(self, records: list[dict]) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "message": "Ragas 评测已通过 ENABLE_RAGAS_EVAL 配置关闭。",
                "sample_count": len(records),
            }

        answerable_records = [record for record in records if record.get("question_type") != "insufficient_info"]
        if not answerable_records:
            return {
                "enabled": False,
                "message": "没有可用于 Ragas 的可回答样本。",
                "sample_count": len(records),
            }

        self._patch_ragas_vertexai_compat()

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_correctness, answer_relevancy, faithfulness
        except ImportError as exc:
            return {
                "enabled": False,
                "message": f"未安装或无法导入 Ragas 运行依赖: {exc}",
                "sample_count": len(records),
            }

        try:
            llm = self._build_ragas_llm()
            embeddings = self._build_langchain_embeddings()
        except Exception as exc:
            return {
                "enabled": False,
                "message": f"Ragas 依赖初始化失败: {exc}",
                "sample_count": len(records),
            }

        rows = [self._build_row(record) for record in answerable_records]
        usable_rows = [row for row in rows if row["retrieved_contexts"]]
        if not usable_rows:
            return {
                "enabled": False,
                "message": "所有可回答样本都缺少 retrieved_contexts，无法计算 Ragas 指标。",
                "sample_count": len(records),
                "excluded_records": len(answerable_records),
            }

        dataset = Dataset.from_list(usable_rows)
        try:
            with self._silent_ragas_output():
                result = evaluate(
                    dataset=dataset,
                    metrics=[answer_relevancy, faithfulness, answer_correctness],
                    llm=llm,
                    embeddings=embeddings,
                    raise_exceptions=False,
                    show_progress=False,
                )
        except TypeError:
            try:
                with self._silent_ragas_output():
                    result = evaluate(
                        dataset=dataset,
                        metrics=[answer_relevancy, faithfulness, answer_correctness],
                        llm=llm,
                        embeddings=embeddings,
                        raise_exceptions=False,
                    )
            except Exception as exc:
                return {
                    "enabled": False,
                    "message": f"Ragas 评测执行失败: {exc}",
                    "sample_count": len(records),
                }
        except Exception as exc:
            return {
                "enabled": False,
                "message": f"Ragas 评测执行失败: {exc}",
                "sample_count": len(records),
            }

        rows_result = self._result_rows(result)
        averages = self._average_metrics(rows_result)
        return {
            "enabled": True,
            "sample_count": len(records),
            "evaluated_count": len(usable_rows),
            "excluded_records": len(records) - len(usable_rows),
            "metrics": averages,
            "details": rows_result,
        }

    @staticmethod
    @contextlib.contextmanager
    def _silent_ragas_output():
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        ragas_logger = logging.getLogger("ragas")
        previous_level = ragas_logger.level
        ragas_logger.setLevel(logging.CRITICAL)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            yield
        ragas_logger.setLevel(previous_level)

    @staticmethod
    def _patch_ragas_vertexai_compat() -> None:
        import sys
        import types

        module_name = "langchain_community.chat_models.vertexai"
        if module_name in sys.modules:
            return

        try:
            from langchain_google_vertexai import ChatVertexAI  # type: ignore
        except ImportError:
            class ChatVertexAI:  # type: ignore[no-redef]
                pass

        shim = types.ModuleType(module_name)
        shim.ChatVertexAI = ChatVertexAI
        sys.modules[module_name] = shim

    @staticmethod
    def _build_row(record: dict[str, Any]) -> dict[str, Any]:
        contexts = [str(item).strip() for item in record.get("retrieved_contexts", []) if str(item).strip()]
        return {
            "user_input": str(record.get("question", "")),
            "response": str(record.get("answer", "")),
            "retrieved_contexts": contexts,
            "reference": str(record.get("gold_answer", "")),
            "record_id": str(record.get("id", "")),
            "question_type": str(record.get("question_type", "unknown")),
        }

    @staticmethod
    def _result_rows(result: Any) -> list[dict[str, Any]]:
        if hasattr(result, "to_pandas"):
            dataframe = result.to_pandas()
            records = dataframe.to_dict(orient="records")
            return [RagasEvaluator._normalize_row(row) for row in records]
        if isinstance(result, list):
            return [RagasEvaluator._normalize_row(row) for row in result]
        if isinstance(result, dict):
            return [RagasEvaluator._normalize_row(result)]
        return []

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            if isinstance(value, float) and math.isnan(value):
                normalized[key] = None
            else:
                normalized[key] = value
        return normalized

    @staticmethod
    def _average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
        metric_names = ["answer_relevancy", "faithfulness", "answer_correctness"]
        averages: dict[str, float] = {}
        for name in metric_names:
            values: list[float] = []
            for row in rows:
                value = row.get(name)
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    values.append(float(value))
            averages[name] = sum(values) / len(values) if values else 0.0
        return averages

    @staticmethod
    def _build_ragas_llm() -> Any:
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY 不能为空。")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from langchain_community.chat_models import ChatOpenAI

        from ragas.llms.base import LangchainLLMWrapper

        base_llm = ChatOpenAI(
            model=settings.llm_model_name,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
        )
        return LangchainLLMWrapper(base_llm, bypass_n=True)

    @staticmethod
    def _build_langchain_embeddings() -> Any:
        if settings.embedding_backend == "sentence_transformers":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

        if settings.embedding_backend == "openai_compatible":
            if not settings.embedding_api_key:
                raise ValueError("EMBEDDING_API_KEY 不能为空。")
            if not settings.embedding_base_url:
                raise ValueError("EMBEDDING_BASE_URL 不能为空。")

            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                from langchain_community.embeddings import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=settings.embedding_model_name,
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
            )

        raise ValueError(f"当前 embedding_backend 不支持 Ragas: {settings.embedding_backend}")
