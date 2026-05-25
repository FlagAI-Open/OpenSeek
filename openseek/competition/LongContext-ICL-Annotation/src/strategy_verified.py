import json
from typing import Any
from strategy_base import BaseStrategy
from llm_client import post_completion

class VerifiedProgramStrategy(BaseStrategy):
    """
    代码生成与验证策略：一次生成验证代码，多次执行应用。
    """
    PROMPT_TEMPLATE = """Task: Write a Python function for the following problem.
<case1>
  <description>Find the maximum of two numbers.</description>
  <examples>
    Input: "[5, 10]", Output: "10"
    Input: "[-1, -5]", Output: "-1"
  </examples>
  <code>def solution(input: str) -> str:
    import json
    nums = json.loads(input)
    return str(max(nums))</code>
</case1>
<case2>
  <description>Calculate the factorial of a number.</description>
  <examples>
    Input: "[3]", Output: "6"
    Input: "[5]", Output: "120"
  </examples>
  <code>def solution(input: str) -> str:
    import json
    import math
    n = json.loads(input)[0]
    return str(math.factorial(n))</code>
</case2>
<case3>
  <description>Extract the first element from a list and return it.</description>
  <examples>
    Input: "['apple', 'banana']", Output: "apple"
    Input: "[10, 20, 30]", Output: "10"
  </examples>
  <code>def solution(input: str) -> str:
    import ast
    data = ast.literal_eval(input)
    return str(data[0])</code>
</case3>
<case4>
  <description>{description}</description>
  <examples>
{examples}
  </examples>
  <code>def solution(input: str) -> str:"""

    def predict(self, task_id: int, task_description: str, prompt_examples: list[dict[str, str]], input_text: str) -> str | None:
        """
        这个方法在 Strategy 基类中被定义。
        在这里，由于模式是'生成一次执行多次'，
        此方法可以直接在 main.py 调用 execute。
        """
        raise NotImplementedError("Use 'prepare_solution' followed by 'execute' for this strategy.")

    def prepare_solution(self, task_description: str, prompt_examples: list[dict[str, str]], max_rounds: int = 5) -> str | None:
        prompt = self.PROMPT_TEMPLATE.format(
            description=task_description.strip(),
            examples=self._format_examples(prompt_examples),
        )

        for round_idx in range(1, max_rounds + 1):
            print(f"\n[VerifiedProgram] Round {round_idx}/{max_rounds} generating code...")
            try:
                # 使用统一客户端
                generated_text = post_completion(prompt, max_tokens=256)
                if not generated_text:
                    continue

                full_code = self._extract_solution_code(generated_text)
                print(f"Extracted code:\n{full_code}")

                # 验证代码
                is_valid = self._verify_solution(full_code, prompt_examples)
                if is_valid:
                    print(f"Verification passed!")
                    return full_code
            except Exception as e:
                print(f"Generation round {round_idx} failed: {e}")

        return None

    def execute(self, full_code: str, input_text: str) -> str:
        namespace: dict[str, Any] = {}
        exec(full_code, namespace, namespace)
        solution_func = namespace.get("solution")
        if not callable(solution_func):
            raise ValueError("Function 'solution' not found in generated code.")
        result = solution_func(input_text)
        return str(result).strip()

    def _format_examples(self, examples: list[dict[str, str]]) -> str:
        if not examples:
            return "    Input: \"[]\", Output: \"\""
        return "\n".join(
            f'    Input: "{ex["input"]}", Output: "{ex["expected"]}"'
            for ex in examples
        )

    def _extract_solution_code(self, generated_text: str) -> str:
        body = generated_text.replace("\r\n", "\n").strip("\n")
        if not body:
            return "    raise ValueError('empty model output')"
        
        lines = body.split("\n")
        # 如果模型输出了完整的代码块标签，截断
        if "</code>" in body:
            body = body.split("</code>", 1)[0]

        # 确保缩进正确且包含定义
        lines = body.split("\n")
        if lines[0].lstrip().startswith("def solution"):
            return body

        if lines[0] and not lines[0].startswith((" ", "\t")):
            lines = [f"    {line}" if line else "" for line in lines]
        
        return "def solution(input: str) -> str:\n" + "\n".join(lines)

    def _verify_solution(self, full_code: str, verification_cases: list[dict[str, str]]) -> bool:
        for ex in verification_cases:
            try:
                actual = self.execute(full_code, ex["input"])
                if actual != ex["expected"]:
                    return False
            except:
                return False
        return True
