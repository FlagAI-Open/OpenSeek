import json
import re
from typing import List, Dict
from strategy_base import BaseStrategy
from llm_client import post_completion

class Task5SadnessStrategy(BaseStrategy):
    """
    任务 5 专用策略：推文情感（悲伤）检测。
    采用动态逻辑分析（维度三）与 XML 标签诱导输出。
    """

    PROMPT_TEMPLATE = """Task: In this task you are given a tweet. You must judge whether the author of the tweet is sad or not. Label the instances as "Sad" or "Not sad" based on your judgment. You can get help from hashtags and emojis, but you should not judge only based on them, and should pay attention to tweet's text as well.

Solve the following cases based on the instruction above.

{dynamic_examples}

<case_target>
  <input>{input_text}</input>
  <output>"""

    def predict(self, task_id: int, task_description: str, prompt_examples: list[dict[str, str]], input_text: str) -> str | None:
        # 1. 分析当前输入推文的情感逻辑（寻找情感诱因）
        input_analysis = self._analyze_tweet_logic(input_text)
        print(f"[Input Analysis] {input_analysis}")
        
        # 2. 动态筛选示例 (基于逻辑关键词进行匹配)
        selected_examples = self._select_relevant_examples(input_text, input_analysis.split(), prompt_examples, k=17)
        
        # 3. 格式化示例为包含逻辑引导的 XML 块
        formatted_examples = self._format_examples_with_logic(selected_examples)
        
        # 4. 构造最终判断 Prompt
        prompt = self.PROMPT_TEMPLATE.format(
            dynamic_examples=formatted_examples,
            input_text=input_text.strip()
        )
        
        # 5. 调用模型完成标注
        prediction = post_completion(prompt, stop=["</output>", "</case_target>"])
        
        if prediction:
            prediction_lower = prediction.lower()
            if "not sad" in prediction_lower:
                return "Not sad"
            elif "sad" in prediction_lower:
                return "Sad"
        
        return "Sad"

    def _analyze_tweet_logic(self, text: str) -> str:
        """
        利用 Qwen3-4B 分析推文的情感逻辑（关键词+深层理由）。
        """
        analysis_prompt = f"""Identify the core reason for the emotion in this tweet (e.g., loss, irony, excitement, frustration, jealous). 
Tweet: {text}
Core Reason:"""
        try:
            res = post_completion(analysis_prompt, max_tokens=20, stop=["\n"])
            return res.strip() if res else ""
        except:
            return ""

    def _format_examples_with_logic(self, examples: List[Dict]) -> str:
        """
        格式化示例。在 output 中加入思维链引导。
        """
        parts = []
        for i, ex in enumerate(examples, 1):
            inp = ex['input'].strip()
            out = ex['expected'].strip()
            parts.append(f"<case{i}>\n  <input>{inp}</input>\n  <output>Analysis: Based on keywords and tone, the state is {out.lower()}. Label: {out}</output>\n</case{i}>")
        return "\n".join(parts)

    def _select_relevant_examples(self, input_text: str, analysis_keywords: List[str], all_examples: List[Dict], k: int) -> List[Dict]:
        """
        根据逻辑分析提取的关键词进行匹配。
        """
        input_lower = input_text.lower()
        scored_examples = []
        
        # 清洗关键词
        clean_keywords = [w.strip(",.?!").lower() for w in analysis_keywords if len(w) > 2]
        
        for ex in all_examples:
            ex_input = ex['input'].lower()

            # 💡 防止将当前待预测题目作为示例
            if ex_input == input_lower:
                continue 

            ex_label = ex['expected'].lower()
            score = 0
            
            # 1. 匹配分析出的核心理由词
            for word in clean_keywords:
                if word in ex_input:
                    score += 5
            
            # 2. 文本包含度匹配
            for word in input_lower.split()[:5]:
                if word in ex_input:
                    score += 1

            scored_examples.append((score, ex))
        
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # 确保正负例平衡
        top_k = [ex for score, ex in scored_examples[:k]]
        labels = [ex['expected'].lower() for ex in top_k]
        
        if "sad" not in labels or "not sad" not in labels:
            extra_sad = [ex for s, ex in scored_examples[k:] if ex['expected'].lower() == "sad"]
            extra_not_sad = [ex for s, ex in scored_examples[k:] if ex['expected'].lower() == "not sad"]
            if "sad" not in labels and extra_sad:
                top_k[-1] = extra_sad[0]
            elif "not sad" not in labels and extra_not_sad:
                top_k[-1] = extra_not_sad[0]
                
        return top_k
