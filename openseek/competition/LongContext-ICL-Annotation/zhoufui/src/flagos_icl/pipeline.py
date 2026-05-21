from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .config import AppConfig
from .data import Record
from .model import ModelClient
from .parser import parse_prediction
from .prompting import build_messages
from .retrieval import LexicalRetriever


def vote(predictions: list[dict[str, Any]], abstain_label: str) -> dict[str, Any]:
    labels = [str(item["label"]) for item in predictions if item.get("label")]
    if not labels:
        return {"label": abstain_label, "confidence": 0.0, "rationale": "no valid predictions"}
    counts = Counter(labels)
    confidence_sum: dict[str, float] = defaultdict(float)
    for prediction in predictions:
        confidence_sum[str(prediction["label"])] += float(prediction.get("confidence", 0.0))
    winner = max(counts, key=lambda label: (counts[label], confidence_sum[label]))
    avg_confidence = confidence_sum[winner] / max(1, counts[winner])
    rationales = [item.get("rationale", "") for item in predictions if item.get("label") == winner]
    return {"label": winner, "confidence": round(avg_confidence, 4), "rationale": rationales[0] if rationales else ""}


class AnnotationPipeline:
    def __init__(self, config: AppConfig, train_records: list[Record]) -> None:
        self.config = config
        self.retriever = LexicalRetriever([record for record in train_records if record.label is not None])
        self.model = ModelClient(config.model)

    def annotate_one(self, record: Record) -> dict[str, Any]:
        pipeline = self.config.pipeline
        examples = self.retriever.select(record.text, pipeline.k_examples)
        predictions: list[dict[str, Any]] = []
        for _ in range(max(1, pipeline.vote_rounds)):
            messages = build_messages(self.config, record, examples)
            raw = self.model.complete(messages)
            predictions.append(parse_prediction(raw, pipeline.abstain_label))
        final = vote(predictions, pipeline.abstain_label)
        return {
            pipeline.id_field: record.record_id,
            pipeline.label_field: final["label"],
            "confidence": final["confidence"],
            "rationale": final["rationale"],
        }

    def annotate(self, records: list[Record]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total = len(records)
        for index, record in enumerate(records, start=1):
            print(f"Annotating {index}/{total}: {record.record_id}")
            rows.append(self.annotate_one(record))
        return rows
