""" Tools for counting nouns and verbs in English sentences using spaCy """
from __future__ import annotations

from typing import Any
from nanobot.agent.tools.base import Tool

# Global spaCy model instance, lazy loading
_nlp = None

def _load_spacy_model():
    """Load spaCy pre-trained model lazily, only load once globally"""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


class CountNounsVerbs(Tool):
    """Count the number of nouns and/or verbs in a given English sentence"""

    @property
    def name(self) -> str:
        return "count_nouns_verbs"

    @property
    def description(self) -> str:
        return (
            """Count the number of nouns (including common nouns and proper nouns)
            and/or verbs (content verbs, exclude auxiliary verbs) in an English sentence.
            Choose count type: 'noun' for nouns only, 'verb' for verbs only.
            """
        )

    @property
    def read_only(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sentence": {
                    "type": "string",
                    "description": "The English sentence to count nouns and verbs from.",
                },
                "count_type": {
                    "type": "string",
                    "description": "Type of words to count: 'noun' = only nouns, 'verb' = only verbs",
                    "enum": ["noun", "verb"],
                }
            },
            "required": ["sentence"],
        }

    def count_pos(self, sentence: str, count_type: str = "all") -> str:
        """
        Count nouns and/or verbs in the sentence using spaCy.

        Args:
            sentence (str): Input English sentence
            count_type (str): What to count: 'noun', 'verb', or 'all'

        Returns:
            str: Count result as string
        """
        if not sentence.strip():
            return "Error: Input sentence cannot be empty"

        try:
            nlp = _load_spacy_model()
        except Exception as e:
            return f"Error loading spaCy model: {str(e)}"

        doc = nlp(sentence)

        noun_count = 0
        verb_count = 0
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):  # 普通名词 + 专有名词
                noun_count += 1
            elif token.pos_ == "VERB":  # 实义动词，排除助动词is/are/was等
                verb_count += 1

        if count_type == "noun":
            return str(noun_count)
        elif count_type == "verb":
            return str(verb_count)
        else:  # all
            return f"Nouns: {noun_count}, Verbs: {verb_count}, Total: {noun_count + verb_count}"

    async def execute(
        self,
        sentence: str,
        count_type: str = "noun",
        **kwargs: Any,
    ) -> str:
        try:
            return self.count_pos(sentence, count_type)
        except Exception as e:
            return f"Error counting parts of speech: {str(e)}"
