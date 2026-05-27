import unittest

from src.ai_lab.adapters.official_reader import SampleRecord
from src.ai_lab.retrieval.example_selector import select_examples


def _record(text: str, task_type: str = "classification") -> SampleRecord:
    return SampleRecord(
        sample_id="s1",
        task_id=6,
        task_name="mnli_same_genre_classification",
        task_type=task_type,
        instruction="Choose the correct label for the sentence pair.",
        text=text,
        label_space=["entailment", "neutral", "contradiction"],
    )


class ExampleSelectorTests(unittest.TestCase):
    def test_lexical_topk_prefers_similar_examples(self) -> None:
        examples = [
            {"input": "A cooking recipe about onions and soup.", "output": ["neutral"]},
            {"input": "The soccer match ended after two late goals.", "output": ["entailment"]},
            {"input": "A football team scored a goal in the final minute.", "output": ["contradiction"]},
        ]

        selected = select_examples(
            _record("The football match had a late goal."),
            examples,
            {"selector": "lexical_topk", "num_examples": 1},
        )

        self.assertEqual("A football team scored a goal in the final minute.", selected[0]["input"])

    def test_balanced_similarity_preserves_minority_labels(self) -> None:
        examples = [
            {"input": "Football goal after a long match.", "output": ["entailment"]},
            {"input": "Another football team scored a goal.", "output": ["entailment"]},
            {"input": "Cooking soup in a kitchen.", "output": ["neutral"]},
            {"input": "The sentence directly denies the premise.", "output": ["contradiction"]},
            {"input": "More football and goal details.", "output": ["entailment"]},
        ]

        selected = select_examples(
            _record("Football team scored a goal."),
            examples,
            {
                "selector": "balanced_similarity",
                "num_examples": 4,
                "similarity_quota": 2,
                "label_balance_min_per_label": 1,
            },
        )
        labels = [example["output"][0] for example in selected]

        self.assertIn("neutral", labels)
        self.assertIn("contradiction", labels)
        self.assertGreaterEqual(labels.count("entailment"), 1)

    def test_generation_task_uses_similarity_without_label_balance(self) -> None:
        examples = [
            {"input": "Greek mythology clue", "output": ["Zeus"]},
            {"input": "American presidents clue", "output": ["Lincoln"]},
            {"input": "Roman mythology clue", "output": ["Jupiter"]},
        ]

        selected = select_examples(
            _record("Roman god clue", task_type="generation"),
            examples,
            {"selector": "balanced_similarity", "num_examples": 1},
        )

        self.assertEqual("Roman mythology clue", selected[0]["input"])


if __name__ == "__main__":
    unittest.main()
