"""
POS 序列结构检索器
基于 spaCy 词性标注序列的句式结构相似度，用于 Task 2 (count_nouns_verbs)

核心思想：人学语法靠看同类句型的例子，不是看同类话题的例子。
BM25 找的是"说同一件事"的句子，而 POS 检索找的是"有同样语法结构"的句子。
"""
import re


class POSRetriever:
    """基于 POS 标注序列的句式结构检索器

    与 BM25Retriever 接口一致：retrieve_top_k(query, top_k) -> list[dict]
    """

    def __init__(self, examples):
        """
        Args:
            examples: ICL 示例列表，每个元素包含 'input' 和 'output'
        """
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

        self.examples = examples
        self._precompute()

    @staticmethod
    def _extract_sentence(input_text):
        """从 'Sentence: '...' Count...' 格式中提取句子"""
        m = re.search(r"Sentence:\s*'(.+?)'", input_text)
        return m.group(1) if m else input_text

    @staticmethod
    def _is_noun_task(input_text):
        return 'noun' in input_text.lower()

    def _pos_ngrams(self, doc):
        """从 spaCy doc 提取 POS bigrams 和 trigrams 集合"""
        pos_seq = [token.pos_ for token in doc]

        bigrams = set()
        for i in range(len(pos_seq) - 1):
            bigrams.add((pos_seq[i], pos_seq[i + 1]))

        trigrams = set()
        for i in range(len(pos_seq) - 2):
            trigrams.add((pos_seq[i], pos_seq[i + 1], pos_seq[i + 2]))

        return bigrams, trigrams, len(pos_seq)

    def _precompute(self):
        """预计算所有示例的 POS 特征（只在初始化时跑一次）"""
        print(f"  🔤 预计算 {len(self.examples)} 个示例的 POS 结构特征...")

        sentences = [self._extract_sentence(ex['input']) for ex in self.examples]
        docs = list(self.nlp.pipe(sentences, batch_size=256))

        # 按 noun/verb 类型分池
        self.noun_pool = []  # [(bigrams, trigrams, length, example), ...]
        self.verb_pool = []

        for ex, doc in zip(self.examples, docs):
            bigrams, trigrams, length = self._pos_ngrams(doc)
            entry = (bigrams, trigrams, length, ex)

            if self._is_noun_task(ex['input']):
                self.noun_pool.append(entry)
            else:
                self.verb_pool.append(entry)

        print(f"  ✅ POS 索引完成: noun={len(self.noun_pool)}, verb={len(self.verb_pool)}")

    def retrieve_top_k(self, test_input, top_k=20):
        """检索与 test_input 句式结构最相似的 top_k 个示例

        Args:
            test_input: 测试样本的完整 input 文本
            top_k: 返回的示例数量

        Returns:
            list[dict]: 按结构相似度降序排列的示例列表
        """
        is_noun = self._is_noun_task(test_input)
        pool = self.noun_pool if is_noun else self.verb_pool

        test_sentence = self._extract_sentence(test_input)
        test_doc = self.nlp(test_sentence)
        test_bi, test_tri, test_len = self._pos_ngrams(test_doc)

        scored = []
        for bigrams, trigrams, length, ex in pool:
            # Bigram Jaccard
            bi_union = len(test_bi | bigrams)
            bi_sim = len(test_bi & bigrams) / bi_union if bi_union else 0

            # Trigram Jaccard
            tri_union = len(test_tri | trigrams)
            tri_sim = len(test_tri & trigrams) / tri_union if tri_union else 0

            # Length similarity
            len_sim = 1 - abs(test_len - length) / max(test_len, length, 1)

            score = 0.4 * bi_sim + 0.4 * tri_sim + 0.2 * len_sim
            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]
