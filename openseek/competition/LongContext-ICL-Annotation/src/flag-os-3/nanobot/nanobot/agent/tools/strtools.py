""" Tools for basic math operations """
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, TypeVar
import ast
import sys
from nanobot.agent.tools.base import Tool


class ConcatStr(Tool):
    """concatenate a list of strings"""

    @property
    def name(self) -> str:
        return "concat_strs"

    @property
    def description(self) -> str:
        return (
            """ given a list of strings and concatenate them. """
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "array": {
                    "type": "list",
                    "description": "Input array, a list of strings",
                },
            },
            "required": ["array"],
        }
    
    def concat_strings(self,arr):
        """
        concat a list of strings.

        Args:
            arr (list[str]): List of strings

        Returns:
            str: concat result
        """
        # arr = ast.literal_eval(arr)
  
        return ''.join(arr)

    async def execute(
        self,
        array: str,
        **kwargs: Any,
    ) -> str:
        try:
            return self.concat_strings(array)
        except Exception as e:
            return f"Error Processing array: {e}"

