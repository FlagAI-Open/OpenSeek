import re
from typing import List, Dict
from strategy_base import BaseStrategy
from llm_client import post_completion

TASK6_GENRE_DESCRIPTIONS = {
    "face-to-face": "conversations or dialogues between people (casual spoken language)",
    "government": "information released from public government websites (policy, reports, official language)",
    "letters": "written work for philanthropic fundraising (letter/appeal style)",
    "9/11": "information pertaining to the September 11, 2001 terrorist attacks",
    "slate": "cultural topics from Slate magazine (news analysis, reviews, opinion pieces)",
    "telephone": "telephonic dialogue (phone conversations, informal speech with fillers like 'um', 'uh')",
    "travel": "information in travel guides (place descriptions, tourist attractions, practical tips)",
    "verbatim": "short posts regarding linguistics (language discussion, verbatim quotes)",
    "oup": "non-fiction academic works on textile industry and child development",
    "fiction": "popular works of fiction (narrative storytelling, characters, plot)",
}


class Task6GenreStrategy(BaseStrategy):
    """
    任务 6 专用策略：MNLI 句子领域（Genre）一致性判断。
    挑战：输入中包含一个参考领域 (Ref Genre)，需要判断 S1 和 S2 是否都符合此领域。
    """

    PROMPT_TEMPLATE = """Task: In this task, you're given two sentences, sentence 1 and sentence 2, and the genre they belong to. Your job is to determine if the two sentences belong to the same genre or not. Indicate your answer with Y and N respectively.

Genres available include: face-to-face, government, letters, 9/11, slate, telephone, travel, verbatim, oup, fiction.

Instruction:
- Y: Sentence 1 and Sentence 2 both belong to the SAME genre (the indicated genre).
- N: Sentence 1 and Sentence 2 do NOT belong to the same genre.

Solve the following cases based on the instruction above.

{dynamic_examples}

    <case_target>
  <input>
  Sentence 1: {s1}
  Sentence 2: {s2}
  Genre: {ref_genre}
  </input>
  <output>"""
    GENRE_DESCRIPTIONS = TASK6_GENRE_DESCRIPTIONS
    LEXICAL_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "am", "be", "been", "being",
        "to", "of", "and", "or", "in", "on", "for", "that", "this", "it", "its",
        "i", "you", "he", "she", "they", "we", "do", "did", "does", "have", "has",
        "had", "with", "at", "by", "from", "as", "but", "if", "my", "your", "his",
        "her", "their", "our", "me", "him", "them", "too", "not", "there", "here",
        "than", "then", "so", "because", "into", "out", "up", "down", "about",
        "would", "could", "should", "can", "will", "just", "actually", "like",
        "mean", "some", "something",
    }

    def predict(self, task_id: int, task_description: str, prompt_examples: list[dict[str, str]], input_text: str) -> str | None:
        # 1. 解析输入
        s1, s2, ref_genre = self._parse_sentences(input_text)
        ref_genre = ref_genre.lower()
        
        # 2. 分别识别 S1 和 S2 的潜在领域 (多候选模式)
        genres1 = self._detect_potential_genres(s1)
        genres2 = self._detect_potential_genres(s2)
        print(f"[Genre Check] S1: {list(genres1)} | S2: {list(genres2)} | Ref: {ref_genre}")
        
        # 3. 核心决策逻辑：
        # 如果两句话的潜在领域列表中都包含了参考领域 (Ref)，则是强 Y 信号
        if ref_genre in genres1 and ref_genre in genres2:
            print(f"  [Rule] Multi-match found ({ref_genre} in both) -> Y")
            return "Y"
            
        # 如果任何一句话的潜在领域列表完全确定，且明确排除了 Ref，则是强 N 信号
        if genres1 and "unknown" not in genres1 and ref_genre not in genres1:
            print(f"  [Rule] S1 mismatch (Excluded {ref_genre}) -> N")
            return "N"
        if genres2 and "unknown" not in genres2 and ref_genre not in genres2:
            print(f"  [Rule] S2 mismatch (Excluded {ref_genre}) -> N")
            return "N"

        # 4. 对 detector 不确定的样本，使用 LLM 复核 + 句间词汇一致性约束，避免对泛化过强
        overlap = self._lexical_overlap_count(s1, s2)
        print(f"  [Review] lexical_overlap={overlap}")

        if self._is_unknown_only(genres1) and self._is_unknown_only(genres2):
            pair_match = self._review_pair_against_reference(s1, s2, ref_genre)
            print(f"  [Review] pair_match={pair_match}")
            return "Y" if pair_match and overlap >= 1 else "N"

        if self._is_unknown_only(genres1):
            s1_match = self._review_sentence_against_reference(s1, ref_genre)
            print(f"  [Review] S1_match={s1_match}")
            return "Y" if s1_match and overlap >= 1 else "N"

        if self._is_unknown_only(genres2):
            s2_match = self._review_sentence_against_reference(s2, ref_genre)
            print(f"  [Review] S2_match={s2_match}")
            return "Y" if s2_match and overlap >= 1 else "N"

        return "Y"

    def _detect_potential_genres(self, sentence: str) -> set[str]:
        """
        请求识别句子可能的所有潜在领域。允许返回多个候选以降低误杀。
        """
        detect_prompt = f"""Task: Identify all possible genres for the given sentence. 
A sentence can belong to multiple categories due to context ambiguity.

List of valid genres: face-to-face, government, letters, 9/11, slate, telephone, travel, verbatim, oup, fiction.

Instruction:
1. Analyze the language style and content.
2. Output all genres that could reasonably apply, separated by commas.
3. If no specific genre fits, output 'unknown'.

Examples:
- "The treasury released the new tax guidelines." -> government
- "I think we should go now." -> face-to-face, telephone
- "The hotel is near the beach." -> travel, fiction

Sentence: {sentence}
Genres:"""
        
        try:
            res = post_completion(detect_prompt, max_tokens=30, stop=["\n", ".", "Output:", "Sentence:"])
            if res:
                res_clean = res.strip().lower()
                valid_genres = ["face-to-face", "government", "letters", "9/11", "slate", "telephone", "travel", "verbatim", "oup", "fiction"]
                detected = set()
                for g in valid_genres:
                    if g in res_clean:
                        detected.add(g)
                return detected if detected else {"unknown"}
            return {"unknown"}
        except Exception:
            return {"unknown"}

    def _is_unknown_only(self, genres: set[str]) -> bool:
        return len(genres) == 1 and "unknown" in genres

    def _review_sentence_against_reference(self, sentence: str, ref_genre: str) -> bool:
        description = self.GENRE_DESCRIPTIONS.get(ref_genre, ref_genre)
        prompt = f"""Task: Decide whether the sentence plausibly belongs to the genre "{ref_genre}".
Judge by source style and content rather than exact keywords.
A short paraphrase can still fit the same genre.

Genre description: {description}
Sentence: {sentence}

Return exactly one line: Answer: Y or Answer: N"""
        prediction = post_completion(prompt, max_tokens=50)
        return self._extract_binary_answer(prediction) == "Y"

    def _review_pair_against_reference(self, s1: str, s2: str, ref_genre: str) -> bool:
        description = self.GENRE_DESCRIPTIONS.get(ref_genre, ref_genre)
        prompt = f"""Reference genre: {ref_genre}
Genre description: {description}
Sentence 1: {s1}
Sentence 2: {s2}

Question: do BOTH sentences plausibly come from this genre?
Do not reject a sentence only because it is shorter or paraphrased.
Return exactly one line: Answer: Y or Answer: N"""
        prediction = post_completion(prompt, max_tokens=70)
        return self._extract_binary_answer(prediction) == "Y"

    def _extract_binary_answer(self, prediction: str) -> str:
        if not prediction:
            return "N"

        prediction_up = prediction.upper()
        match = re.search(r"ANSWER\s*[:：]\s*([YN])", prediction_up)
        if match:
            return match.group(1)

        lines = [line.strip().upper() for line in prediction.splitlines() if line.strip()]
        for line in reversed(lines):
            if line in ("Y", "N"):
                return line

        return "Y" if prediction_up.rfind("Y") > prediction_up.rfind("N") else "N"

    def _normalized_tokens(self, text: str) -> set[str]:
        raw_tokens = re.findall(r"[a-z0-9']+", text.lower())
        normalized = set()

        for token in raw_tokens:
            if token in self.LEXICAL_STOPWORDS or len(token) <= 2:
                continue

            for suffix in ("ing", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) > len(suffix) + 2:
                    token = token[:-len(suffix)]
                    break

            normalized.add(token)

        return normalized

    def _lexical_overlap_count(self, s1: str, s2: str) -> int:
        return len(self._normalized_tokens(s1) & self._normalized_tokens(s2))

    def _parse_sentences(self, input_text: str) -> tuple[str, str, str]:
        """
        从任务输入中提取 Sentence 1, Sentence 2 和 给定的参考领域 (Reference Genre)。
        """
        s1, s2, ref_genre = "", "", ""
        try:
            # 模式: "Sentence 1: ... Sentence 2: ... Genre: ..."
            if "Sentence 1:" in input_text and "Sentence 2:" in input_text:
                parts1 = input_text.split("Sentence 1:")
                main_chunk = parts1[1]
                
                parts2 = main_chunk.split("Sentence 2:")
                s1 = parts2[0].strip()
                
                remain_chunk = parts2[1]
                if "Genre:" in remain_chunk:
                    parts3 = remain_chunk.split("Genre:")
                    s2 = parts3[0].strip()
                    ref_genre = parts3[1].strip().strip(".")
                else:
                    s2 = remain_chunk.strip()
        except:
            pass
        return s1, s2, ref_genre

    def _select_relevant_examples(self, query_genres: List[str], all_examples: List[Dict], k: int) -> List[Dict]:
        scored_examples = []
        for ex in all_examples:
            ex_input = ex['input'].lower()
            ex_label = ex['expected'].upper()
            score = 0
            
            # 命中参考领域加分
            for g in query_genres:
                if g and g.lower() in ex_input:
                    score += 10
            
            # 加上基本的文本相关度
            scored_examples.append((score, ex))
            
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for s, ex in scored_examples[:k]]

    def _format_examples(self, examples: List[Dict]) -> str:
        parts = []
        for i, ex in enumerate(examples, 1):
            # 将原始输入重新格式化为易读的结构化输入
            inp = ex['input'].strip()
            # 从 example input 中提取 ref_genre 以便对齐 case_target
            s1, s2, rg = self._parse_sentences(inp)
            out = ex['expected'].strip()
            parts.append(f"<case{i}>\n  <input>\n  Sentence 1: {s1}\n  Sentence 2: {s2}\n  Genre: {rg}\n  </input>\n  <output>{out}</output>\n</case{i}>")
        return "\n".join(parts)
