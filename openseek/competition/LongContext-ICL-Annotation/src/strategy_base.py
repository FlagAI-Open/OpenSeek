from abc import ABC, abstractmethod
from typing import Any

class BaseStrategy(ABC):
    @abstractmethod
    def predict(self, task_id: int, task_description: str, prompt_examples: list[dict[str, str]], input_text: str) -> str | None:
        pass
