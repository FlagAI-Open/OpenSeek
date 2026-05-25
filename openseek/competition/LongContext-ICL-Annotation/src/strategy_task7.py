import json
import re
from typing import List, Dict
from strategy_base import BaseStrategy
from llm_client import LLMClient

class Task7JeopardyStrategy(BaseStrategy):
    """
    任务 7 专用策略：Jeopardy 问答。
    改进点：
    1. 使用 Chat Role (System/User) 模式提高 4B 模型的指令遵循能力。
    2. 引入 CoT (Chain of Thought) 引导模型先推理再给出简洁答案。
    3. 动态筛选相似 Category 的示例。
    """

    SYSTEM_PROMPT = """You are a Jeopardy expert. Your task is to provide the best answer for a given clue within a specific category.

Instructions:
- The answer must be as concise as possible (usually 1-3 words).
- For people, prioritize the most common form (often just the last name unless specified).
- ALWAYS reason first, then provide the answer in the format 'Answer: [result]'.

Example output format:
Reasoning: [your analysis]
Answer: [brief answer]"""

    USER_TEMPLATE = """{dynamic_examples}

<case_target>
  <input>
  Category: {category}
  Clue: {clue}
  </input>
</case_target>"""

    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()

    def predict(self, task_id: int, task_description: str, prompt_examples: list[dict[str, str]], input_text: str) -> str | None:
        # 1. 解析输入
        category, clue = self._parse_input(input_text)
        
        # 2. 选取相关示例 (优先匹配 Category 关键词)
        selected_examples = self._select_relevant_examples(category, prompt_examples, k=8)
        
        # 3. 格式化示例 (增加推理过程)
        formatted_examples = self._format_examples_with_reasoning(selected_examples)
        
        # 4. 构造消息队列
        user_content = self.USER_TEMPLATE.format(
            category=category,
            clue=clue,
            dynamic_examples=formatted_examples
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # 5. 实现重试机制：当答案超过 100 字符时尝试微调提示词，最多 3 次
        max_attempts = 3
        for attempt in range(max_attempts):
            if attempt > 0:
                # 微调提示词：在 system prompt 中增加更强硬的字数限制要求
                messages[0]["content"] += "\nCRITICAL: Your final answer MUST be extremely brief. Use at most 3 words. Do NOT include explanations in the final Answer line."
            
            # 调用模型
            prediction = self.llm_client.post_chat_completion(
                messages=messages, 
                max_tokens=10000,
                stop=["</case_target>"]
            )
            
            print(f"[Task7 Attempt {attempt+1}] Original Prediction: {prediction}")
            
            if not prediction:
                continue

            # 提取 Answer 部分
            result = self._extract_result(prediction)
            
            # 如果答案长度在合理范围内 (<= 100)，则返回
            if result and len(result) <= 100:
                print(f"[Task7 Final Result] {result}")
                return result
            
            print(f"[Task7 Attempt {attempt+1}] Result too long ({len(result)} chars), retrying...")

        return None

    def _extract_result(self, prediction: str) -> str:
        # 提取预测结果的最后一行作为主要候选
        lines = [line.strip() for line in prediction.strip().split("\n") if line.strip()]
        last_line = lines[-1] if lines else ""

        # 1. 尝试从最后一行匹配 Answer: [xxx] 或 Label: [xxx] (不区分大小写)
        match = re.search(r"(?:Answer|Label|Result):\s*\*?\[?(['\"]?)(.*?)\1\]?\*?$", last_line, re.IGNORECASE)
        
        if match:
            result = match.group(2).strip()
        # 2. 如果最后一行没有关键词，但也不是以 Reasoning: 开头的，则直接将最后一行作为结果
        elif last_line and not last_line.lower().startswith("reasoning:"):
            # 如果最后一行包含 "The result is '...'" 这种带特定描述但没显式标签的
            result_match = re.search(r"(?:The|This)\s+(?:result|answer)\s+is\s+(['\"]?)(.*?)\1$", last_line, re.IGNORECASE)
            if result_match:
                result = result_match.group(2).strip()
            else:
                result = last_line
        else:
            # 3. 退而求其次，在全文中查找最后一个 Answer:
            all_matches = list(re.finditer(r"(?:Answer|Label|Result):\s*\*?\[?(['\"]?)(.*?)\1\]?\*?$", prediction, re.IGNORECASE | re.MULTILINE))
            if all_matches:
                result = all_matches[-1].group(2).strip() # 注意这里是 group(2) 以跳过引号
            else:
                # fallback: 如果还是没提取到，保留最后一行
                result = last_line if last_line else prediction.strip()

        # 4. 确保清理结果中的引号和末尾标点
        result = result.strip().strip("'\"").strip()
        result = re.sub(r"[.!?]+$", "", result).strip().lower()
        return result

    def _parse_input(self, text: str) -> tuple[str, str]:
        category = ""
        clue = ""
        cat_match = re.search(r"Category:\s*(.*)", text)
        clue_match = re.search(r"Clue:\s*(.*)", text, re.DOTALL)
        
        if cat_match:
            category = cat_match.group(1).split("\n")[0].strip()
        if clue_match:
            clue = clue_match.group(1).strip()
        
        # 如果正则没匹配到，简单切分
        if not category or not clue:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines:
                if line.startswith("Category:"):
                    category = line.replace("Category:", "").strip()
                elif line.startswith("Clue:"):
                    clue = line.replace("Clue:", "").strip()
        
        return category, clue

    def _select_relevant_examples(self, current_category: str, all_examples: List[Dict], k: int) -> List[Dict]:
        cat_words = set(re.findall(r"\w+", current_category.lower()))
        scored = []
        for ex in all_examples:
            ex_cat, _ = self._parse_input(ex['input'])
            ex_cat_words = set(re.findall(r"\w+", ex_cat.lower()))
            score = len(cat_words.intersection(ex_cat_words))
            scored.append((score, ex))
        
        # 按分数排序并取前 K 个
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k]]

    def _format_examples_with_reasoning(self, examples: List[Dict]) -> str:
        parts = []
        for i, ex in enumerate(examples, 1):
            category, clue = self._parse_input(ex['input'])
            label = ex['expected'].lower()
            # 构造简短的推理占位符，引导模型学习格式
            parts.append(f"<case{i}>\n  <input>\n  Category: {category}\n  Clue: {clue}\n  </input>\n  Reasoning: The category is {category}. The clue describes {label}.\n  Answer: {label}\n</case{i}>")
        return "\n".join(parts)
