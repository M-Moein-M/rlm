from abc import ABC, abstractmethod
from typing import Any

from rlm.core.types import ModelUsageSummary, UsageSummary

# Default timeout for LM API calls (in seconds)
DEFAULT_TIMEOUT: float = 300.0


class BaseLM(ABC):
    """
    Base class for all language model routers / clients. When the RLM makes sub-calls, it currently
    does so in a model-agnostic way, so this class provides a base interface for all language models.
    """

    def __init__(self, model_name: str, timeout: float = DEFAULT_TIMEOUT, **kwargs):
        self.model_name = model_name
        self.timeout = timeout
        self.kwargs = kwargs

    @abstractmethod
    def completion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """Get cost summary for all model calls."""
        raise NotImplementedError

    @abstractmethod
    def get_last_usage(self) -> ModelUsageSummary:
        """Get the last cost summary of the model."""
        raise NotImplementedError

    def _require_text_response(
        self,
        content: Any,
        *,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> str:
        provider_name = provider or self.__class__.__name__
        resolved_model = model_name or self.model_name or "unknown"
        if content is None:
            raise ValueError(
                f"{provider_name} returned None content for model '{resolved_model}'."
            )
        if not isinstance(content, str):
            raise TypeError(
                f"{provider_name} returned non-string content for model '{resolved_model}': "
                f"{type(content).__name__}"
            )
        if content.strip() == "":
            raise ValueError(
                f"{provider_name} returned empty content for model '{resolved_model}'."
            )
        return content
