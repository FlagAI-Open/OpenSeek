from flagos_icl.long_context import compress_text, split_chunks
from flagos_icl.parser import parse_prediction
from flagos_icl.retrieval import LexicalRetriever
from flagos_icl.data import Record
from flagos_icl.deterministic import closest_integers, collatz, concat_strings, extract_answer_tag
from scripts.validate_submission_zip import EXPECTED_COUNTS, validate_zip

import json
import zipfile


def test_parse_prediction_json() -> None:
    result = parse_prediction('{"label":"technology","confidence":0.9,"rationale":"matched"}', "unknown")
    assert result["label"] == "technology"
    assert result["confidence"] == 0.9


def test_parse_prediction_fallback() -> None:
    result = parse_prediction("not json", "unknown")
    assert result["label"] == "unknown"


def test_retriever_selects_similar_record() -> None:
    examples = [
        Record("1", "budget policy government", "policy", {}),
        Record("2", "model inference transformer", "technology", {}),
    ]
    selected = LexicalRetriever(examples).select("large model inference", 1)
    assert selected[0].label == "technology"


def test_long_context_compression_respects_budget() -> None:
    text = "\n".join(f"paragraph {index} " + "x" * 100 for index in range(100))
    compressed = compress_text(text, max_chars=1000, chunk_chars=200)
    assert len(compressed) <= 1100
    assert split_chunks("a\nb\nc", 10) == ["a\nb\nc"]


def test_deterministic_exact_tasks() -> None:
    assert closest_integers("[59, 26, -96, -30]") == "33"
    assert collatz("[72, 29, 49]") == "[36, 88, 148]"
    assert concat_strings("['p', 'that.', 'o']") == "pthat.o"
    assert extract_answer_tag("reason\n<answer>Sad</answer>") == "Sad"


def test_validate_submission_zip_accepts_expected_shape(tmp_path) -> None:
    archive_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, count in EXPECTED_COUNTS.items():
            lines = [
                json.dumps({"test_sample_id": f"{name}-{index}", "prediction": "x"})
                for index in range(count)
            ]
            archive.writestr(name, "\n".join(lines) + "\n")

    counts, errors = validate_zip(archive_path)

    assert errors == []
    assert counts == EXPECTED_COUNTS


def test_validate_submission_zip_rejects_bom_and_extra_fields(tmp_path) -> None:
    archive_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, count in EXPECTED_COUNTS.items():
            records = [
                {"test_sample_id": f"{name}-{index}", "prediction": "x"}
                for index in range(count)
            ]
            if name == "openseek-1-v1.jsonl":
                records[0]["confidence"] = 1.0
                payload = "\ufeff" + "\n".join(json.dumps(record) for record in records) + "\n"
            else:
                payload = "\n".join(json.dumps(record) for record in records) + "\n"
            archive.writestr(name, payload)

    _, errors = validate_zip(archive_path)

    assert any("UTF-8 BOM" in error for error in errors)
    assert any("extra fields: confidence" in error for error in errors)
