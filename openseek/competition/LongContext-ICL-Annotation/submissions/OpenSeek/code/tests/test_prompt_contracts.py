import unittest

from src.ai_lab.output_parser import canonicalize_label
from src.ai_lab.adapters.official_reader import SampleRecord
from src.ai_lab.decision import finalize_prediction
from src.ai_lab.pipeline import _deterministic_prediction, _filter_registry, _load_protocol_set, _maybe_run_adjudication
from src.ai_lab.protocols import load_protocol, render_protocol
from src.ai_lab.runbook import build_label_description


class _CandidateABackend:
    def generate_raw(self, prompt: str, task_type: str | None = None) -> str:
        return '{"winner": "Candidate A", "positive_for_a": [], "positive_for_b": [], "decision_basis": "test"}'


class PromptContractTests(unittest.TestCase):
    def test_task_specific_answer_requirements(self) -> None:
        self.assertIn(
            "one base-10 integer",
            build_label_description(["1", "2"], "classification", "count_nouns_verbs"),
        )
        self.assertIn(
            "Python-style list of integers",
            build_label_description(["[1, 2, 3]"], "classification", "collatz_conjecture"),
        )
        self.assertIn(
            "final concatenated output",
            build_label_description(["abc"], "classification", "conala_concat_strings"),
        )
        sadness = build_label_description([], "classification", "semeval_2018_task1_tweet_sadness_detection")
        self.assertIn("Sad", sadness)
        self.assertIn("Not sad", sadness)
        self.assertIn(
            "short Jeopardy answer phrase",
            build_label_description([], "generation", "jeopardy_answer_generation_all"),
        )

    def test_general_protocol_uses_answer_requirements_not_candidate_answers(self) -> None:
        protocol = load_protocol("prompts/protocol_a.yaml")
        prompt = render_protocol(
            protocol["template"],
            {
                "task_definition": "Answer the question.",
                "task_type": "generation",
                "label_desc": build_label_description([], "generation", "jeopardy_answer_generation_all"),
                "examples_block": "# clue\n<label>example</label>",
                "context": "current clue",
                "label_a": "",
                "label_b": "",
            },
        )
        self.assertIn("[Answer Requirements]", prompt)
        self.assertNotIn("[Candidate Answers]", prompt)
        self.assertIn("not candidate answers", prompt)

    def test_sadness_canonicalization_prefers_closed_labels(self) -> None:
        self.assertEqual(
            "Not sad",
            canonicalize_label(
                "The tweet is not sad.",
                task_type="classification",
                task_name="semeval_2018_task1_tweet_sadness_detection",
                label_space=[],
            ),
        )
        self.assertEqual(
            "Sad",
            canonicalize_label(
                "@SizweM01 and it's kinda depressing hey!!!",
                task_type="classification",
                task_name="semeval_2018_task1_tweet_sadness_detection",
                label_space=[],
            ),
        )
        self.assertEqual(
            "Not sad",
            canonicalize_label(
                "Okay, let's see. The tweet is a quote from Ernest Hemingway.",
                task_type="classification",
                task_name="semeval_2018_task1_tweet_sadness_detection",
                label_space=[],
            ),
        )
        self.assertEqual(
            "Sad",
            canonicalize_label(
                "Final answer: Sad",
                task_type="classification",
                task_name="semeval_2018_task1_tweet_sadness_detection",
                label_space=[],
            ),
        )

    def test_count_task_falls_back_to_integer_from_source_text(self) -> None:
        self.assertEqual(
            "2",
            canonicalize_label(
                "Answer:",
                task_type="classification",
                task_name="count_nouns_verbs",
                label_space=[],
                source_text="Sentence: 'Jars of food are being canned in a pot of boiling water'. Count the number of verbs in this sentence.",
            ),
        )
        self.assertEqual(
            "3",
            canonicalize_label(
                "Two birds are perching on top of tree branches",
                task_type="classification",
                task_name="count_nouns_verbs",
                label_space=[],
                source_text="Sentence: 'Two birds are perching on top of tree branches'. Count the number of nouns in this sentence.",
            ),
        )

    def test_closest_integers_falls_back_to_source_min_difference(self) -> None:
        self.assertEqual(
            "1",
            canonicalize_label(
                "31",
                task_type="classification",
                task_name="closest_integers",
                label_space=[],
                source_text="[-84, 79, -59, -31, -62, -52, 78]",
            ),
        )

    def test_collatz_falls_back_to_source_text_list(self) -> None:
        self.assertEqual(
            "[274, 190, 74]",
            canonicalize_label(
                "Answer:",
                task_type="classification",
                task_name="collatz_conjecture",
                label_space=[],
                source_text="[91, 380, 148]",
            ),
        )
        self.assertEqual(
            "[36, 88, 148]",
            canonicalize_label(
                "[36, 88, 148]",
                task_type="classification",
                task_name="collatz_conjecture",
                label_space=[],
                source_text="[72, 29, 49]",
            ),
        )

    def test_concat_task_falls_back_to_joined_source_list(self) -> None:
        self.assertEqual(
            "fkbuttonedearefIasW",
            canonicalize_label(
                "The task is to concatenate the strings in the list.",
                task_type="classification",
                task_name="conala_concat_strings",
                label_space=[],
                source_text="['f', 'k', 'buttoned', 'e', 'are', 'f', 'I', 'as', 'W']",
            ),
        )

    def test_jeopardy_answer_canonicalization_removes_question_preamble(self) -> None:
        self.assertEqual(
            "the simpsons",
            canonicalize_label(
                "Final answer: What is The Simpsons?",
                task_type="generation",
                task_name="jeopardy_answer_generation_all",
                label_space=[],
            ),
        )
        self.assertEqual(
            "lord cornwallis",
            canonicalize_label(
                "<label>Who is Lord Cornwallis?</label>",
                task_type="generation",
                task_name="jeopardy_answer_generation_all",
                label_space=[],
            ),
        )
        self.assertEqual(
            "salman rushdie",
            canonicalize_label(
                "Salman Rushdie</label>",
                task_type="generation",
                task_name="jeopardy_answer_generation_all",
                label_space=[],
            ),
        )
        self.assertEqual(
            "venice film festival",
            canonicalize_label(
                "<answer>Venice Film Festival</answer>",
                task_type="generation",
                task_name="jeopardy_answer_generation_all",
                label_space=[],
            ),
        )
        self.assertEqual(
            "riverdale",
            canonicalize_label(
                '{"choice": "B", "answer": "Riverdale", "confidence": 99}',
                task_type="generation",
                task_name="jeopardy_answer_generation_all",
                label_space=[],
            ),
        )

    def test_pipeline_deterministic_prediction_for_safe_tasks(self) -> None:
        closest = SampleRecord(
            sample_id="s1",
            task_id=1,
            task_name="closest_integers",
            task_type="classification",
            instruction="",
            text="[71, -93, 63, -41, -18, 18]",
            label_space=[],
        )
        self.assertEqual("8", _deterministic_prediction(closest))

        collatz = SampleRecord(
            sample_id="s3",
            task_id=3,
            task_name="collatz_conjecture",
            task_type="classification",
            instruction="",
            text="[91, 63, 148, 8, 6]",
            label_space=[],
        )
        self.assertEqual("[274, 190, 74, 4, 3]", _deterministic_prediction(collatz))

        concat = SampleRecord(
            sample_id="s4",
            task_id=4,
            task_name="conala_concat_strings",
            task_type="classification",
            instruction="",
            text="['f', 'k', 'buttoned', 'e', 'are', 'f', 'I', 'as', 'W']",
            label_space=[],
        )
        self.assertEqual("fkbuttonedearefIasW", _deterministic_prediction(concat))

    def test_pipeline_runtime_task_id_filter(self) -> None:
        registry = [
            {"task_id": 1, "task_name": "a"},
            {"task_id": 2, "task_name": "b"},
            {"task_id": 7, "task_name": "c"},
        ]
        filtered = _filter_registry(registry, {"runtime": {"task_ids": [2, "7"]}})
        self.assertEqual([2, 7], [entry["task_id"] for entry in filtered])

    def test_task_name_specific_protocol_override(self) -> None:
        protocols = _load_protocol_set(
            {
                "protocol_a_path": "prompts/protocol_a.yaml",
                "protocol_b_path": "prompts/protocol_b.yaml",
                "protocol_c_light_path": "prompts/protocol_c_light.yaml",
                "jeopardy_minimal_path": "prompts/jeopardy_minimal.yaml",
                "protocols_by_task_name": {
                    "jeopardy_answer_generation_all": ["jeopardy_minimal_path"],
                },
            },
            "generation",
            "jeopardy_answer_generation_all",
        )
        self.assertEqual(["jeopardy_minimal"], [protocol["name"] for protocol in protocols])

    def test_voting_ignores_empty_labels_when_nonempty_exists(self) -> None:
        final = finalize_prediction(
            [
                {"label": "", "valid": False, "confidence": 0},
                {"label": "", "valid": False, "confidence": 0},
                {"label": "salman rushdie", "valid": True, "confidence": 60},
            ],
            {"score": 0.4},
        )
        self.assertEqual("salman rushdie", final["prediction"])

    def test_adjudication_candidate_placeholder_maps_to_answer_text(self) -> None:
        record = SampleRecord(
            sample_id="s1",
            task_id=7,
            task_name="jeopardy_answer_generation_all",
            task_type="generation",
            instruction="answer the clue",
            text="current clue",
            label_space=[],
        )
        predictions = _maybe_run_adjudication(
            record=record,
            config={"prompt": {"protocol_c_path": "prompts/protocol_c.yaml"}},
            backend=_CandidateABackend(),
            model_backend="transformers_local",
            fallback_prediction="fallback",
            prompt_values={
                "task_definition": record.instruction,
                "task_type": record.task_type,
                "label_desc": build_label_description([], "generation", record.task_name),
                "examples_block": "",
                "context": record.text,
            },
            predictions=[
                {"label": "duke university", "valid": True, "confidence": 60},
                {"label": "university of north carolina", "valid": True, "confidence": 50},
            ],
        )
        self.assertEqual("duke university", predictions[-1]["label"])


if __name__ == "__main__":
    unittest.main()
