from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests
from tqdm import tqdm

from method import postprocess_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "openseek-2_count_nouns_verbs.json"
TASK_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 noun/verb count candidate router.")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="候选文件，格式 name=path。第一个候选会作为保守回退。",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:2026")
    parser.add_argument("--model-name", type=str, default="Qwen3-4B")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--prompt-style",
        choices=("plain", "conservative", "calibrated"),
        default="conservative",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_candidate_specs(specs: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    seen_names = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"候选参数必须是 name=path 格式: {spec}")
        name, path_text = spec.split("=", 1)
        name = re.sub(r"[^a-zA-Z0-9_:-]+", "_", name.strip()) or f"candidate_{len(parsed) + 1}"
        if name in seen_names:
            raise ValueError(f"候选名重复: {name}")
        path = resolve_path(path_text.strip())
        if not path.exists():
            raise FileNotFoundError(f"候选文件不存在: {path}")
        parsed.append((name, path))
        seen_names.add(name)
    return parsed


def normalize_row_id(row: Dict[str, Any]) -> str:
    for key in ("test_sample_id", "id", "sample_id"):
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    raise KeyError(f"未找到样本 id 字段，可用键={list(row.keys())}")


def normalize_count(value: Any) -> str:
    if value is None:
        return ""
    cleaned = postprocess_prediction(str(value), TASK_ID)
    return "" if cleaned is None else str(cleaned).strip()


def load_prediction_file(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[normalize_row_id(row)] = normalize_count(row.get("prediction"))
    return rows


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_reference(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    loaded = load_prediction_file(path)
    return {row_id: pred for row_id, pred in loaded.items() if pred}


def chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def candidate_letter(index: int) -> str:
    return chr(ord("A") + index)


def source_label(source: str, index: int) -> str:
    if index == 0 or source.lower() in {"current", "base", "baseline"}:
        return "trusted current"
    if source.lower() in {"balanced", "bucket", "bucket_balanced"}:
        return "label-balanced alternative"
    return source


def parse_task_target(sample_input: str) -> str:
    lower = sample_input.lower()
    if "count the number of nouns" in lower:
        return "nouns"
    if "count the number of verbs" in lower:
        return "verbs"
    return "unknown"


CALIBRATION_EXAMPLES = [
    (
        "Sentence: 'Ironic picture of man and woman walking up a sidewalk under a \"Wrong Way\" sign'. Count the number of nouns in this sentence.",
        "6",
    ),
    (
        "Sentence: 'a gentleman in pajamas taking a selfie with his camera'. Count the number of nouns in this sentence.",
        "4",
    ),
    (
        "Sentence: 'A little girl with an broken arm posing near a restroom sink and toilet'. Count the number of nouns in this sentence.",
        "5",
    ),
    (
        "Sentence: 'A tennis player holding a racket on the tennis court'. Count the number of nouns in this sentence.",
        "5",
    ),
    (
        "Sentence: 'Two pieces of pizza with lasagna toppings on a plate'. Count the number of verbs in this sentence.",
        "0",
    ),
    (
        "Sentence: 'A elephant that is standing on a floor at a bowling alley'. Count the number of verbs in this sentence.",
        "1",
    ),
    (
        "Sentence: 'Jars of food are being canned in a pot of boiling water'. Count the number of verbs in this sentence.",
        "2",
    ),
    (
        "Sentence: 'A baseball player catches the ball as an opponent makes it on base'. Count the number of verbs in this sentence.",
        "2",
    ),
]


def calibration_block() -> str:
    lines = []
    for idx, (sample_input, answer) in enumerate(CALIBRATION_EXAMPLES, start=1):
        lines.append(f"{idx}. {sample_input}\n   Correct count: {answer}")
    return "\n".join(lines)


def build_router_prompt(
    sample_input: str,
    candidates: Sequence[Dict[str, str]],
    prompt_style: str,
) -> str:
    candidate_lines = []
    for idx, item in enumerate(candidates):
        candidate_lines.append(
            f"{candidate_letter(idx)}. [{source_label(item['source'], idx)}] {item['answer']}"
        )
    calibration_text = ""
    if prompt_style == "plain":
        decision_rules = (
            "1. Recount the requested part of speech in the sentence.\n"
            "2. Choose the candidate whose integer count is correct.\n"
            "3. If uncertain, choose A.\n"
        )
    elif prompt_style == "calibrated":
        calibration_text = f"[Caption Counting Calibration]\n{calibration_block()}\n\n"
        decision_rules = (
            "1. Follow the caption-counting conventions illustrated by the calibration examples, not generic grammar alone.\n"
            "2. First identify the requested target: nouns or verbs. Ignore the other part of speech.\n"
            "3. For nouns, count noun words naming people, visible objects, places, substances, body parts, and concrete things. Count coordinated noun words separately. Count repeated noun words separately when they appear as words in the caption.\n"
            "4. Do not count articles, determiners, pronouns, numbers, prepositions, ordinary color words, or action participles as nouns.\n"
            "5. For verbs, count clear predicate/action/state verb words in the caption, including finite verbs, copulas/auxiliaries when they carry the predicate, and verbal participles used as actions. Do not count noun words that merely resemble verbs.\n"
            "6. Candidate A is the current strongest prediction. Choose B/C/D only when your recount clearly supports that integer under the calibration examples.\n"
            "7. If two candidate counts remain plausible, choose A.\n"
        )
    else:
        decision_rules = (
            "1. First identify whether the request asks for nouns or verbs, then recount only that target.\n"
            "2. Candidate A is the current strongest prediction. Choose another candidate only when your recount clearly supports it.\n"
            "3. For nouns, count noun words that name visible objects, people, places, substances, or concrete things. Count repeated noun words separately. Do not count articles, determiners, numbers, ordinary adjectives, pronouns, or prepositions.\n"
            "4. For verbs, count finite verbs, auxiliaries/copulas, and main participles that function as actions or states. Do not count noun words such as players, shop, sale, front, or container names just because they look verb-like.\n"
            "5. Caption fragments may omit a finite verb; in that case count only clear action/state verb forms.\n"
            "6. If both candidates look plausible, choose A.\n"
        )
    return (
        "You are resolving candidate integer answers for one image-caption part-of-speech counting task.\n"
        "Return exactly one capital letter from the candidate list. Do not write the number. Do not explain.\n\n"
        f"{calibration_text}"
        "[Sample]\n"
        f"{sample_input.strip()}\n\n"
        "[Candidate Counts]\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "[Decision Rules]\n"
        f"{decision_rules}\n"
        "Letter only:\n"
    )


def extract_chat_content(message: Any) -> str:
    if not isinstance(message, dict):
        return str(message).strip()
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def request_router(sample_input: str, candidates: Sequence[Dict[str, str]], args: argparse.Namespace) -> Tuple[str, str]:
    prompt = build_router_prompt(sample_input, candidates, args.prompt_style)
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(chat_url(args.api_base), json=payload, timeout=args.timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return "", prompt
    return extract_chat_content(choices[0].get("message", {})), prompt


def choose_from_response(raw_text: str, candidates: Sequence[Dict[str, str]]) -> int:
    text = raw_text.strip().upper()
    for idx in range(len(candidates)):
        letter = candidate_letter(idx)
        if re.search(rf"\b{letter}\b", text):
            return idx
    count = normalize_count(raw_text)
    if count:
        for idx, item in enumerate(candidates):
            if item["answer"] == count:
                return idx
    return 0


def build_unique_candidates(row_id: str, candidate_predictions: Sequence[Tuple[str, Dict[str, str]]]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    seen_answers = set()
    for name, rows in candidate_predictions:
        answer = rows.get(row_id, "")
        if not answer or answer in seen_answers:
            continue
        seen_answers.add(answer)
        candidates.append({"source": name, "answer": answer})
    return candidates


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(resolve_path(args.dataset_path))
    test_samples = list(dataset.get("test_samples", []))
    if args.max_samples > 0:
        test_samples = test_samples[: args.max_samples]

    candidate_specs = parse_candidate_specs(args.candidate)
    candidate_predictions = [(name, load_prediction_file(path)) for name, path in candidate_specs]

    output_file = output_dir / "openseek-2-v1.jsonl"
    decisions_file = output_dir / "task2_router_decisions.jsonl"
    summary_file = output_dir / "task2_router_summary.json"

    def route_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        row_id = str(sample.get("id"))
        sample_input = str(sample.get("input", ""))
        candidates = build_unique_candidates(row_id, candidate_predictions)
        if not candidates:
            candidates = [{"source": candidate_specs[0][0], "answer": ""}]

        raw_response = ""
        selected_idx = 0
        prompt = ""
        if len(candidates) > 1:
            try:
                raw_response, prompt = request_router(sample_input, candidates, args)
                selected_idx = choose_from_response(raw_response, candidates)
            except Exception as exc:
                raw_response = f"<router_error>{type(exc).__name__}: {exc}"
                selected_idx = 0

        selected = candidates[selected_idx]
        return {
            "test_sample_id": row_id,
            "prediction": selected["answer"],
            "_decision": {
                "id": row_id,
                "target": parse_task_target(sample_input),
                "input": sample_input,
                "candidates": candidates,
                "selected_index": selected_idx,
                "selected_source": selected["source"],
                "selected_answer": selected["answer"],
                "raw_response": raw_response,
                "prompt": prompt,
            },
        }

    rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(route_one, sample): idx for idx, sample in enumerate(test_samples)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Task2 router", unit="sample"):
            idx = futures[future]
            row = future.result()
            rows.append({"idx": idx, "row": row})

    rows.sort(key=lambda item: item["idx"])
    with output_file.open("w", encoding="utf-8") as writer, decisions_file.open("w", encoding="utf-8") as decision_writer:
        for item in rows:
            row = item["row"]
            writer.write(json.dumps({"test_sample_id": row["test_sample_id"], "prediction": row["prediction"]}, ensure_ascii=False) + "\n")
            decisions.append(row["_decision"])
            decision_writer.write(json.dumps(row["_decision"], ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "output_file": str(output_file),
        "decisions_file": str(decisions_file),
        "samples": len(rows),
        "candidate_sources": [name for name, _ in candidate_specs],
        "routed_samples": sum(1 for item in decisions if len(item["candidates"]) > 1),
        "selected_by_source": {},
    }
    source_counts: Dict[str, int] = {}
    for item in decisions:
        source = item["selected_source"]
        source_counts[source] = source_counts.get(source, 0) + 1
    summary["selected_by_source"] = dict(sorted(source_counts.items()))

    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
