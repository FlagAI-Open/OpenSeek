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


class Findmindiff(Tool):
    """Find Min Diff"""

    @property
    def name(self) -> str:
        return "find_min_diff"

    @property
    def description(self) -> str:
        return (
            """find the minimum absolute difference between 2 integers in the list. 
            The absolute difference is the absolute value of one integer subtracted by another. """
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
                    "description": "Input array",
                },
            },
            "required": ["array"],
        }
    
    def minimum_absolute_difference(self,arr):
        """
        Find the minimum absolute difference between any two integers in the list.

        Args:
            arr (list[int]): List of integers

        Returns:
            int: The minimum absolute difference
        """
        # arr = ast.literal_eval(arr)
        # Sort the array
        sorted_arr = sorted(arr)

        # Find minimum difference between adjacent elements
        min_diff = float('inf')
        for i in range(len(sorted_arr) - 1):
            diff = abs(sorted_arr[i] - sorted_arr[i+1])
            if diff < min_diff:
                min_diff = diff
                # Early exit if we find 0, can't get smaller than that
                if min_diff == 0:
                    break

        return str(min_diff)

    async def execute(
        self,
        array: str,
        **kwargs: Any,
    ) -> str:
        try:
            return self.minimum_absolute_difference(array)
        except Exception as e:
            return f"Error Processing array: {e}"



class CollatzConjecture(Tool):
    """Collatz Conjecture"""

    @property
    def name(self) -> str:
        return "collatz_conjecture"

    @property
    def description(self) -> str:
        return (
            """For every element in the list, if the element is even you should divide by two, 
            if the element is odd you should multiply by three then add one. """
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
                    "description": "Input array",
                },
            },
            "required": ["array"],
        }
    
    def collatz_conjecture(self,arr):
        """
            For every element in the list, if the element is even you should divide by two, 
            if the element is odd you should multiply by three then add one. 
        Args:
            arr (list): List of elements 

        Returns:
            arr (list): Result 
        """
        # arr = ast.literal_eval(arr)

        result = []
        
        for num in arr:
            if num % 2 == 0:
                new_num = num // 2
            else:
                new_num = num * 3 + 1
            result.append(new_num)
        
        return str(result)

    async def execute(
        self,
        array: str,
        **kwargs: Any,
    ) -> str:
        try:
            return self.collatz_conjecture(array)
        except Exception as e:
            return f"Error Processing array: {e}"


