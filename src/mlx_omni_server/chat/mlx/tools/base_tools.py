from abc import ABC, abstractmethod
from typing import List, Optional

from ..core_types import ToolCall


class BaseToolParser(ABC):
    start_tool_calls: str
    end_tool_calls: str

    @abstractmethod
    def parse_tools(self, text: str) -> Optional[List[ToolCall]]:
        pass
