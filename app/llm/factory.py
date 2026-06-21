from app.config.settings import settings
from app.llm.base import BaseLLMProvider
from app.llm.openai_compatible import OpenAICompatibleLLMProvider


def get_llm_provider() -> BaseLLMProvider:
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLMProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model_name=settings.llm_model_name,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")
