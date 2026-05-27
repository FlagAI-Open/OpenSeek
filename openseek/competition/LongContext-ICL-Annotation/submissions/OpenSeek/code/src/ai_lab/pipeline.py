import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from src.ai_lab.adapters import SampleRecord, load_task_dataset
from src.ai_lab.config import load_yaml
from src.ai_lab.data import save_jsonl
from src.ai_lab.datasets import load_registry
from src.ai_lab.decision import (
    build_judge_prompt_values,
    compute_confidence,
    finalize_prediction,
    select_top2_labels,
)
from src.ai_lab.inference import build_backend
from src.ai_lab.output_parser import canonicalize_label, parse_protocol_output
from src.ai_lab.protocols import load_protocol, render_protocol
from src.ai_lab.retrieval import build_query, chunk_text, reorder_front_back, retrieve_top_chunks, select_examples
from src.ai_lab.runbook import build_examples_block, build_label_description
from src.ai_lab.submit import validate_prediction_dir
from src.ai_lab.utils.io import ensure_dir
from src.ai_lab.utils.logging import build_logger


def _submission_file_name(task_id: int, version: int = 1) -> str:
    return f"openseek-{task_id}-v{version}.jsonl"


def _filter_registry(registry: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_ids = config.get("runtime", {}).get("task_ids")
    if not task_ids:
        return registry
    selected = {int(task_id) for task_id in task_ids}
    return [entry for entry in registry if int(entry["task_id"]) in selected]


def _load_protocol_set(prompt_cfg: Dict[str, Any], task_type: str, task_name: str = "") -> List[Dict[str, Any]]:
    protocols_by_task_name = prompt_cfg.get("protocols_by_task_name") or {}
    if task_name in protocols_by_task_name:
        loaded: List[Dict[str, Any]] = []
        for item in protocols_by_task_name[task_name]:
            path = prompt_cfg[item] if item in prompt_cfg else item
            loaded.append(load_protocol(path))
        return loaded

    if task_type == "code_generation":
        return [load_protocol(prompt_cfg["code_generation_protocol_path"])]

    first_pass_protocols = prompt_cfg.get("first_pass_protocols")
    if first_pass_protocols:
        loaded: List[Dict[str, Any]] = []
        for item in first_pass_protocols:
            path = prompt_cfg[item] if item in prompt_cfg else item
            loaded.append(load_protocol(path))
        return loaded

    return [
        load_protocol(prompt_cfg["protocol_a_path"]),
        load_protocol(prompt_cfg["protocol_b_path"]),
        load_protocol(prompt_cfg["protocol_c_light_path"]),
    ]


def _heuristic_protocol_prediction(protocol_name: str, fallback_prediction: str) -> str:
    if protocol_name == "protocol_b":
        return json.dumps(
            {
                "candidates": [
                    {
                        "label": fallback_prediction,
                        "status": "support",
                        "evidence": "heuristic fallback",
                    }
                ],
                "final_label": fallback_prediction,
                "confidence": 60,
            },
            ensure_ascii=False,
        )
    if protocol_name == "protocol_c":
        return json.dumps(
            {
                "positive_for_a": [fallback_prediction],
                "positive_for_b": [fallback_prediction],
                "winner": fallback_prediction,
                "decision_basis": "heuristic fallback",
            },
            ensure_ascii=False,
        )
    if protocol_name == "protocol_c_light":
        return json.dumps(
            {
                "label": fallback_prediction,
                "confidence": 58,
                "evidence": ["heuristic fallback"],
            },
            ensure_ascii=False,
        )
    if protocol_name == "code_generation":
        return fallback_prediction
    return json.dumps(
        {
            "label": fallback_prediction,
            "confidence": 60,
            "evidence": ["heuristic fallback"],
            "reason": "heuristic fallback",
        },
        ensure_ascii=False,
    )


def _build_context(record: SampleRecord, cfg: Dict[str, Any]) -> str:
    if record.task_type in {"classification", "generation"} and len(record.text) <= int(
        cfg["icl"]["small_text_threshold_chars"]
    ):
        return record.text

    chunks = chunk_text(
        record.text,
        chunk_size=int(cfg["icl"]["chunk_size_chars"]),
        overlap=int(cfg["icl"]["chunk_overlap_chars"]),
    )
    query = build_query(record.instruction, record.label_space, record.text)
    retrieved = retrieve_top_chunks(
        chunks,
        query=query,
        top_k=int(cfg["icl"]["retrieval_top_k"]),
    )
    ordered = reorder_front_back(retrieved)
    return "\n\n".join(str(chunk["text"]) for chunk in ordered)


def _first_example_prediction(examples: List[Dict[str, Any]]) -> str:
    if not examples:
        return ""
    output = examples[0].get("output", "")
    if isinstance(output, list):
        return str(output[0]) if output else ""
    return str(output)


def _build_prompt_values(
    record: SampleRecord,
    context: str,
    examples_block: str,
    label_desc: str,
) -> Dict[str, Any]:
    return {
        "task_definition": record.instruction,
        "task_type": record.task_type,
        "label_desc": label_desc,
        "examples_block": examples_block,
        "context": context,
        "label_a": "",
        "label_b": "",
    }


def _deterministic_prediction(record: SampleRecord) -> str | None:
    if record.task_name not in {"closest_integers", "collatz_conjecture", "conala_concat_strings"}:
        return None
    value = canonicalize_label(
        "__deterministic__",
        task_type=record.task_type,
        task_name=record.task_name,
        label_space=record.label_space,
        source_text=record.text,
    )
    return value if value.strip() else None


def _run_first_pass(
    record: SampleRecord,
    protocols: List[Dict[str, Any]],
    prompt_values: Dict[str, Any],
    backend: Any,
    model_backend: str,
    fallback_prediction: str,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for protocol in protocols:
        prompt = render_protocol(protocol["template"], prompt_values)
        if model_backend == "heuristic":
            raw_text = _heuristic_protocol_prediction(protocol["name"], fallback_prediction)
        else:
            raw_text = backend.generate_raw(prompt, task_type=record.task_type)
        parsed = parse_protocol_output(raw_text, record.task_type)
        parsed["label"] = canonicalize_label(
            str(parsed.get("label", "")),
            task_type=record.task_type,
            task_name=record.task_name,
            label_space=record.label_space,
            source_text=record.text,
        )
        parsed["answer"] = parsed["label"]
        parsed["valid"] = bool(str(parsed["label"]).strip())
        parsed["protocol"] = protocol["name"]
        parsed["prompt_chars"] = len(prompt)
        predictions.append(parsed)
    return predictions


def _maybe_run_variant(
    record: SampleRecord,
    config: Dict[str, Any],
    prompt_values: Dict[str, Any],
    backend: Any,
    model_backend: str,
    fallback_prediction: str,
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    variant = load_protocol(config["prompt"]["protocol_a_variant_path"])
    prompt = render_protocol(variant["template"], prompt_values)
    if model_backend == "heuristic":
        raw_text = _heuristic_protocol_prediction("protocol_a", fallback_prediction)
    else:
        raw_text = backend.generate_raw(prompt, task_type=record.task_type)
    parsed = parse_protocol_output(raw_text, record.task_type)
    parsed["label"] = canonicalize_label(
        str(parsed.get("label", "")),
        task_type=record.task_type,
        task_name=record.task_name,
        label_space=record.label_space,
        source_text=record.text,
    )
    parsed["answer"] = parsed["label"]
    parsed["valid"] = bool(str(parsed["label"]).strip())
    parsed["protocol"] = variant["name"]
    parsed["prompt_chars"] = len(prompt)
    predictions.append(parsed)
    return predictions


def _maybe_run_adjudication(
    record: SampleRecord,
    config: Dict[str, Any],
    backend: Any,
    model_backend: str,
    fallback_prediction: str,
    prompt_values: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    protocol = load_protocol(config["prompt"]["protocol_c_path"])
    label_a, label_b = select_top2_labels(predictions)
    judge_values = build_judge_prompt_values(
        task_definition=record.instruction,
        task_type=record.task_type,
        label_desc=prompt_values["label_desc"],
        examples_block=prompt_values["examples_block"],
        context=prompt_values["context"],
        label_a=label_a or fallback_prediction,
        label_b=label_b or fallback_prediction,
    )
    prompt = render_protocol(protocol["template"], judge_values)
    if model_backend == "heuristic":
        raw_text = _heuristic_protocol_prediction("protocol_c", fallback_prediction)
    else:
        raw_text = backend.generate_raw(prompt, task_type=record.task_type)
    parsed = parse_protocol_output(raw_text, record.task_type)
    raw_label = str(parsed.get("label", "")).strip().lower().rstrip(".:")
    if raw_label in {"candidate a", "a"}:
        parsed["label"] = label_a or fallback_prediction
    elif raw_label in {"candidate b", "b"}:
        parsed["label"] = label_b or fallback_prediction
    parsed["label"] = canonicalize_label(
        str(parsed.get("label", "")),
        task_type=record.task_type,
        task_name=record.task_name,
        label_space=record.label_space,
        source_text=record.text,
    )
    parsed["answer"] = parsed["label"]
    parsed["valid"] = bool(str(parsed["label"]).strip())
    parsed["protocol"] = protocol["name"]
    parsed["prompt_chars"] = len(prompt)
    predictions.append(parsed)
    return predictions


def run_prediction(config_path: str) -> None:
    config = load_yaml(config_path)
    registry = _filter_registry(load_registry(config["data"]["registry_path"]), config)
    logger = build_logger(config["runtime"]["log_dir"])
    prediction_root = ensure_dir(config["data"]["prediction_root"])
    model_backend = config["model"].get("backend", "heuristic")
    backend = build_backend(config["model"])
    save_every = int(config["runtime"].get("save_every_samples", 10))

    for dataset_cfg in registry:
        dataset = load_task_dataset(config["data"]["official_data_dir"], dataset_cfg)
        examples = dataset.examples
        fallback_prediction = _first_example_prediction(examples[: int(config["icl"]["num_examples"])])
        label_desc = build_label_description(
            dataset.test_samples[0].label_space if dataset.test_samples else [],
            dataset.task_type,
            dataset.task_name,
        )
        protocols = _load_protocol_set(config["prompt"], dataset.task_type, dataset.task_name)
        outputs: List[Dict[str, Any]] = []
        max_samples = config["runtime"].get("max_samples_per_task")
        sample_records = dataset.test_samples[: int(max_samples)] if max_samples else dataset.test_samples
        output_path = prediction_root / _submission_file_name(dataset.task_id)

        for index, record in enumerate(sample_records, start=1):
            deterministic = _deterministic_prediction(record)
            if deterministic is not None:
                outputs.append(
                    {
                        "test_sample_id": record.sample_id,
                        "prediction": deterministic,
                        "meta": {
                            "task_id": record.task_id,
                            "task_name": record.task_name,
                            "task_type": record.task_type,
                            "confidence": 1.0,
                            "strategy": "deterministic",
                            "evidence": [],
                        },
                    }
                )
                if index % save_every == 0 or index == len(sample_records):
                    save_jsonl(str(output_path), outputs)
                    logger.info(
                        "Task %s progress: %s/%s predictions saved to %s",
                        dataset.task_id,
                        index,
                        len(sample_records),
                        output_path,
                    )
                continue

            context = _build_context(record, config)
            selected_examples = select_examples(record, examples, config["icl"])
            examples_block = build_examples_block(selected_examples, dataset.task_type)
            prompt_values = _build_prompt_values(
                record=record,
                context=context,
                examples_block=examples_block,
                label_desc=label_desc,
            )
            predictions = _run_first_pass(
                record=record,
                protocols=protocols,
                prompt_values=prompt_values,
                backend=backend,
                model_backend=model_backend,
                fallback_prediction=fallback_prediction,
            )
            confidence = compute_confidence(predictions)

            if (
                dataset.task_type != "code_generation"
                and confidence["score"] < float(config["decision"]["variant_threshold"])
            ):
                predictions = _maybe_run_variant(
                    record=record,
                    config=config,
                    prompt_values=prompt_values,
                    backend=backend,
                    model_backend=model_backend,
                    fallback_prediction=fallback_prediction,
                    predictions=predictions,
                )
                confidence = compute_confidence(predictions)

            if (
                dataset.task_type != "code_generation"
                and confidence["score"] < float(config["decision"]["adjudication_threshold"])
            ):
                predictions = _maybe_run_adjudication(
                    record=record,
                    config=config,
                    backend=backend,
                    model_backend=model_backend,
                    fallback_prediction=fallback_prediction,
                    prompt_values=prompt_values,
                    predictions=predictions,
                )
                confidence = compute_confidence(predictions)

            final = finalize_prediction(predictions, confidence)
            outputs.append(
                {
                    "test_sample_id": record.sample_id,
                    "prediction": final["prediction"],
                    "meta": {
                        "task_id": record.task_id,
                        "task_name": record.task_name,
                        "task_type": record.task_type,
                        "confidence": round(float(final["confidence"]), 4),
                        "strategy": final["strategy"],
                        "evidence": final["evidence"],
                    },
                }
            )

            if index % save_every == 0 or index == len(sample_records):
                save_jsonl(str(output_path), outputs)
                logger.info(
                    "Task %s progress: %s/%s predictions saved to %s",
                    dataset.task_id,
                    index,
                    len(sample_records),
                    output_path,
                )

        save_jsonl(str(output_path), outputs)
        logger.info(
            "Saved %s predictions for task %s to %s",
            len(outputs),
            dataset.task_id,
            output_path,
        )


def run_evaluation(config_path: str) -> None:
    config = load_yaml(config_path)
    registry = _filter_registry(load_registry(config["data"]["registry_path"]), config)
    logger = build_logger(config["runtime"]["log_dir"])
    report = validate_prediction_dir(
        prediction_root=config["data"]["prediction_root"],
        registry=registry,
        official_data_dir=config["data"]["official_data_dir"],
    )
    eval_path = Path(config["data"]["prediction_root"]) / "submission_validation.json"
    eval_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for task_name, status in report.items():
        logger.info("%s validation: %s", task_name, status)
    logger.info("Wrote submission validation report to %s", eval_path)


def run_packaging(config_path: str) -> None:
    config = load_yaml(config_path)
    logger = build_logger(config["runtime"]["log_dir"])
    prediction_root = Path(config["data"]["prediction_root"])
    zip_path = Path(config["submission"]["zip_name"])

    required_files = config["submission"].get("required_prediction_files", [])
    missing = [name for name in required_files if not (prediction_root / name).exists()]
    if missing:
        logger.warning("Packaging with missing prediction files: %s", ", ".join(missing))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(prediction_root.glob("*.jsonl")):
            zf.write(file_path, arcname=file_path.name)

    logger.info("Created submission package at %s", zip_path)
