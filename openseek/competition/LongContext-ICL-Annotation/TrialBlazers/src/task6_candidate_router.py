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
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "openseek-6_mnli_same_genre_classification.json"
TASK_ID = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task6 same-genre candidate router.")
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
    return parser.parse_args()


CALIBRATION_EXAMPLES = [
    (
        "Sentence 1: I do not have the energy to remedy these deficiencies now. Sentence 2: I don't have the strength to fix these problems now. Genre: slate.",
        "Y",
    ),
    (
        "Sentence 1: and i think it's frightening to them to see the roles switching and i think i think this reaction comes more out of fear now my husband is Sentence 2: I think it scares them to notice the roles switching. Genre: telephone.",
        "Y",
    ),
    (
        "Sentence 1: However, if it is obtained in order to engage the patient in treatment, the information is protected under the above federal regulations that require the express, written permission of the patient before it can be shared with others. Sentence 2: Ecology of Fear, according to the columnists, contained errors but Los City of Quartz didn't. Genre: slate.",
        "N",
    ),
    (
        "Sentence 1: were you have you i take it you haven't spent any time in the military Sentence 2: Jon said there is nothing else we can do. Genre: telephone.",
        "N",
    ),
    (
        "Sentence 1: To the west of Naoussa is Kolymbithres, a growing resort whose beaches are surrounded by strange rock features that are folded by immense natural forces and eroded by the wind. Sentence 2: Sangria is suited for hot environments. Genre: travel.",
        "N",
    ),
]


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


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    cleaned = postprocess_prediction(str(value), TASK_ID)
    if cleaned is None:
        return ""
    text = str(cleaned).strip().upper()
    if text in {"Y", "N"}:
        return text
    if text.startswith("Y"):
        return "Y"
    if text.startswith("N"):
        return "N"
    return ""


def load_prediction_file(path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[normalize_row_id(row)] = normalize_label(row.get("prediction"))
    return rows


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chat_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/chat/completions"


def candidate_letter(index: int) -> str:
    return chr(ord("A") + index)


def source_label(source: str, index: int) -> str:
    if index == 0 or source.lower() in {"current", "base", "baseline"}:
        return "trusted current"
    if source.lower() in {"genrefirst", "genre_first"}:
        return "genre-first alternative"
    return source


def calibration_block() -> str:
    lines = []
    for idx, (sample_input, answer) in enumerate(CALIBRATION_EXAMPLES, start=1):
        lines.append(f"{idx}. {sample_input}\n   Correct label: {answer}")
    return "\n".join(lines)


def build_router_prompt(sample_input: str, candidates: Sequence[Dict[str, str]]) -> str:
    candidate_lines = []
    for idx, item in enumerate(candidates):
        candidate_lines.append(
            f"{candidate_letter(idx)}. [{source_label(item['source'], idx)}] {item['answer']}"
        )
    return (
        "You are resolving candidate Y/N labels for one MNLI-style same-genre classification item.\n"
        "Return exactly one capital letter from the candidate list. Do not write Y/N. Do not explain.\n\n"
        "[Calibration]\n"
        f"{calibration_block()}\n\n"
        "[Sample]\n"
        f"{sample_input.strip()}\n\n"
        "[Candidate Labels]\n"
        f"{chr(10).join(candidate_lines)}\n\n"
        "[Decision Rules]\n"
        "1. Candidate A is the current strongest prediction. Choose another candidate only when the sample clearly supports it.\n"
        "2. Y means sentence 2 plausibly belongs to the stated genre and can serve as a hypothesis, paraphrase, consequence, contradiction, or same-source continuation related to sentence 1.\n"
        "3. N means sentence 2 is from a different genre/source, shifts to an unrelated concrete anchor, or only shares generic words without the benchmark-style relation.\n"
        "4. Contradiction, negation, simplification, or a changed detail can still be Y when sentence 2 is clearly about the same sentence-1 situation.\n"
        "5. If candidate B says N while A says Y, choose B only for a clear genre/source/anchor switch, not for ordinary contradiction.\n"
        "6. If candidate B says Y while A says N, choose B only when sentence 2 has strong evidence for the stated genre or a clear same-source hypothesis relation.\n"
        "7. If uncertain, choose A.\n\n"
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
    prompt = build_router_prompt(sample_input, candidates)
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
    label = normalize_label(raw_text)
    if label:
        for idx, item in enumerate(candidates):
            if item["answer"] == label:
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


def parse_genre(sample_input: str) -> str:
    match = re.search(r"Genre:\s*([^.]+)\.?\s*$", sample_input, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip().lower() if match else "unknown"


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

    output_file = output_dir / "openseek-6-v1.jsonl"
    decisions_file = output_dir / "task6_router_decisions.jsonl"
    summary_file = output_dir / "task6_router_summary.json"

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
                "genre": parse_genre(sample_input),
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="Task6 router", unit="sample"):
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
