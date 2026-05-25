import json
import re
from typing import Any, Callable, List, Dict

from llm_client import LLMClient
from strategy_base import BaseStrategy


class Task2StructuredStrategy(BaseStrategy):
    """
    任务 2 专用策略：
    使用 Chat Role (System/User) 模式配合 CoT 引导模型。
    识别句子中的所有名词或实义动词，并返回计数。
    """

    SYSTEM_PROMPT = """你是一个语法分析工具。你的任务是识别给定句子中的特定词性（名词或动词）并计算它们的数量。

指令：
- 仔细分析句子。
- 为识别出的每个单词提供推理。
- 最后，将最终数量包裹在 <answer> 标签中。

示例输出格式：
<reasoning>
[你的逐步分析过程]
</reasoning>
<answer>[最终数量]</answer>"""

    NOUN_CASE_EXAMPLES = """<case1>
  <input>Ironic picture of man and woman walking up a sidewalk under a "Wrong Way" sign</input>
  <reasoning>
  1. picture: 名词 (主语)
  2. man: 名词 (介词宾语)
  3. woman: 名词 (介词宾语)
  4. sidewalk: 名词 (介词宾语)
  5. Way: 名词定语 ("Wrong Way" 的一部分)
  6. sign: 名词 (介词宾语)
  总计：6 个名词。
  </reasoning>
  <answer>6</answer>
</case1>
<case2>
  <input>A tennis player holding a racket on the tennis court</input>
  <reasoning>
  1. tennis: 名词定语 (修饰 player)
  2. player: 名词 (主语)
  3. racket: 名词 (宾语)
  4. tennis: 名词定语 (修饰 court)
  5. court: 名词 (介词宾语)
  总计：5 个名词。
  </reasoning>
  <answer>5</answer>
</case2>
<case3>
  <input>Two sinks and some cupboards in a bathroom</input>
  <reasoning>
  1. sinks: 名词 (主语)
  2. cupboards: 名词 (主语)
  3. bathroom: 名词 (介词宾语)
  注意：'Two' 是数词。
  总计：3 个名词。
  </reasoning>
  <answer>3</answer>
</case3>"""

    VERB_CASE_EXAMPLES = """<case1>
  <input>The ladder of a jet is lowered from the side for loading passengers</input>
  <reasoning>
  1. is: 助动词 (auxiliary)
  2. lowered: 实义动词 (主要动词，被动语态)
  3. loading: 实义动词 (动名词，起动词作用)
  总计动作/状态动词：2 个 (lowered, loading)。
  </reasoning>
  <answer>2</answer>
</case1>
<case2>
  <input>A baseball player catches the ball as an opponent makes it on base</input>
  <reasoning>
  1. catches: 实义动词 (现在时)
  2. makes: 实义动词 (现在时)
  总计：2 个动词。
  </reasoning>
  <answer>2</answer>
</case2>
<case3>
  <input>Jars of food are being canned in a pot of boiling water</input>
  <reasoning>
  1. are: 助动词
  2. being: 助动词
  3. canned: 实义动词 (主要动词)
  4. boiling: 实义动词 (现在分词，表示动作)
  总计：2 个动词。
  </reasoning>
  <answer>2</answer>
</case3>"""

    def __init__(self):
        super().__init__()
        self.llm_client = LLMClient()

    def predict(
        self,
        task_id: int,
        task_description: str,
        prompt_examples: list[dict[str, str]],
        input_text: str,
    ) -> str | None:
        try:
            # 1. 提取句子和目标
            parsed = self._expected_parse_output(input_text)
            sentence = parsed.get("sentence", "").strip()
            target = parsed.get("target", "").lower()

            if not sentence:
                return "0"

            # 2. 构造消息
            if "noun" in target:
                examples = self.NOUN_CASE_EXAMPLES
                instruction = "识别并计算所有名词（包括名词定语）。"
            else:
                examples = self.VERB_CASE_EXAMPLES
                instruction = "识别并计算所有实义动词和动作分词（最终计数不包括像 'is' 这样的纯助动词）。"

            user_content = f"{examples}\n\n任务：{instruction}\n目标句子：{sentence}\n\n<reasoning>\n"

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            # 3. 调用模型
            prediction = self.llm_client.post_chat_completion(
                messages=messages,
                max_tokens=5000,
                temperature=0.0,
                stop=["</case>"]
            )

            print(f"[Task2 Original Prediction] {prediction}")

            if prediction:
                # 尝试从 <answer> 标签中提取
                match = re.search(r"<answer>\s*(\d+)\s*</answer>", prediction, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1)
                
                # 如果没有标签，回退到原有逻辑：查找最后一个数字
                match = re.search(r"(?:Answer|Total|result):\s*\*?\[?(\d+)\]?\*?$", prediction.strip(), re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1)
                
                # 查找全文最后一个数字
                nums = re.findall(r"\d+", prediction)
                if nums:
                    return nums[-1]
            
            return "0"

        except Exception as e:
            print(f"[ERROR] Task2 predict failed: {e}")
            return "0"

    def _expected_parse_output(self, input_text: str) -> dict[str, str]:
        # 尝试从常见格式提取句子和目标
        # 模式1: Sentence: '...'. Count the number of ...
        prefix = "Sentence: '"
        suffix = "'. Count the number of "
        
        # 模式2: {sentence} Count the number of {target}s...
        # 任务2的输入通常包含明确的指令，这里做一个更通用的提取
        
        target = "noun" if "noun" in input_text.lower() else "verb"
        
        if prefix in input_text and suffix in input_text:
            try:
                sentence_start = input_text.index(prefix) + len(prefix)
                sentence_end = input_text.rfind(suffix)
                sentence = input_text[sentence_start:sentence_end]
                return {"sentence": sentence, "target": target}
            except Exception:
                pass
        
        # 降级方案：如果无法通过特定标记提取，尝试清洗可能的指令后缀
        sentence = input_text
        for pattern in [
            r"\.?\s*Count the number of.*",
            r"\.?\s*How many.*",
        ]:
            sentence = re.sub(pattern, "", sentence, flags=re.IGNORECASE)
            
        # 清洗引号
        sentence = sentence.strip().strip("'").strip('"')
        
        return {"sentence": sentence, "target": target}

    def _xml_escape(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
