from typing import Any

from openai import OpenAI

from app.llm.base import BaseLLMProvider


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    def __init__(self, base_url: str, api_key: str, model_name: str, temperature: float = 0.1) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model_name),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content or ""
