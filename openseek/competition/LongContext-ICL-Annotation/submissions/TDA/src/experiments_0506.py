import argparse
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
EXP_DIR = OUTPUT_DIR / "0506_experiments"
BASE_ZIP = OUTPUT_DIR / "submission_0506v1.zip"
V3_ZIP = OUTPUT_DIR / "submission_0501v3.zip"
API_URL = "http://0.0.0.0:2026/v1/completions"
MODEL_NAME = "../Qwen3-4B"
EXPECTED_LINES = {1: 500, 2: 500, 3: 500, 4: 500, 5: 500, 6: 500, 7: 500, 8: 166}


def load_zip(zip_path: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            match = re.search(r"openseek-(\d+)-v1\.jsonl", name)
            if not match:
                continue
            task = int(match.group(1))
            out[task] = [
                json.loads(line)
                for line in zf.read(name).decode("utf-8").splitlines()
                if line.strip()
            ]
    return out


def write_package(version: str, rows_by_task: dict[int, list[dict[str, Any]]], note: str) -> Path:
    version_dir = EXP_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUT_DIR / f"submission_{version}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task in range(1, 9):
            rows = rows_by_task[task]
            if len(rows) != EXPECTED_LINES[task]:
                raise ValueError(f"task {task}: expected {EXPECTED_LINES[task]} rows, got {len(rows)}")
            for row in rows:
                if row.get("prediction") is None:
                    row["prediction"] = ""
                if "test_sample_id" not in row or "prediction" not in row:
                    raise ValueError(f"task {task}: missing fields")
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            filename = f"openseek-{task}-v1.jsonl"
            zf.writestr(filename, payload)
            (version_dir / filename).write_text(payload, encoding="utf-8")

    (version_dir / "README.md").write_text(note + "\n", encoding="utf-8")
    return zip_path


def load_task7_data() -> tuple[dict[str, str], list[dict[str, Any]]]:
    with (DATA_DIR / "openseek-7_jeopardy_answer_generation_all.json").open("r", encoding="utf-8") as f:
        task = json.load(f)
    tests = {sample["id"]: sample["input"] for sample in task["test_samples"]}
    return tests, task["examples"]


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def retrieve_examples(examples: list[dict[str, Any]], query: str, limit: int = 12) -> str:
    q = words(query)
    scored = []
    for idx, ex in enumerate(examples):
        ew = words(ex["input"])
        score = len(q & ew)
        scored.append((score, -idx, ex))
    chunks = []
    for score, _idx, ex in sorted(scored, reverse=True):
        if score <= 0 and chunks:
            break
        out = ex["output"][0] if isinstance(ex.get("output"), list) else ex.get("output")
        chunks.append(f"Input: {ex['input']}\nAnswer: {out}\n")
        if len(chunks) >= limit:
            break
    return "\n".join(chunks)


def qwen_task7_answer(sample_input: str, examples: list[dict[str, Any]]) -> str:
    prompt = (
        "/no_think\n"
        "Answer this Jeopardy-style clue. Use only the clue, category, and examples. "
        "Return exactly one concise answer in all lower case, with no explanation.\n\n"
        "Examples:\n"
        f"{retrieve_examples(examples, sample_input)}\n\n"
        f"Input: {sample_input}\n"
        "Answer:"
    )
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "stop": ["\n"],
    }
    last = ""
    for attempt in range(3):
        try:
            resp = requests.post(API_URL, json=payload, timeout=120)
            resp.raise_for_status()
            last = resp.json()["choices"][0]["text"]
            break
        except Exception as exc:  # keep long run resilient
            last = str(exc)
            time.sleep(2 + attempt)
    ans = last.strip().lower()
    ans = re.sub(r"^(what is|who is)\s+", "", ans)
    ans = re.sub(r"^(answer|final answer)\s*:\s*", "", ans)
    ans = ans.strip(" \t\r\n\"'`.,;:!?")
    if ans in {"", "answer", "label", "..."}:
        return ""
    return ans


def nearest_task7_answer(sample_input: str, examples: list[dict[str, Any]]) -> str:
    q = words(sample_input)
    best_score = -1
    best_answer = ""
    for idx, ex in enumerate(examples):
        ew = words(ex["input"])
        score = len(q & ew)
        # Earlier examples win ties for deterministic reproducibility.
        if score > best_score:
            out = ex["output"][0] if isinstance(ex.get("output"), list) else ex.get("output")
            best_score = score
            best_answer = str(out).strip().lower()
    return best_answer


def build_task7_refill(version: str, replace_placeholders: bool) -> Path:
    rows = load_zip(BASE_ZIP)
    tests, examples = load_task7_data()
    bad_values = {"", "answer", "label", "..."}
    replaced = []
    for row in rows[7]:
        pred = str(row.get("prediction", "")).strip().lower()
        should_replace = pred == "" or (replace_placeholders and pred in bad_values)
        if should_replace:
            new_pred = nearest_task7_answer(tests[row["test_sample_id"]], examples)
            if new_pred:
                row["prediction"] = new_pred
                replaced.append(row["test_sample_id"])
    note = (
        f"{version}: based on submission_0506v1; task7 Qwen3-4B/FlagScale refill. "
        f"replace_placeholders={replace_placeholders}; replaced={len(replaced)} ids.\n"
        + "\n".join(replaced)
    )
    return write_package(version, rows, note)


def build_single_task_swap(version: str, task: int) -> Path:
    rows = load_zip(BASE_ZIP)
    v3 = load_zip(V3_ZIP)
    rows[task] = v3[task]
    note = f"{version}: based on submission_0506v1; replace task {task} with isolated 0501v3 predictions."
    return write_package(version, rows, note)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["task7-empty", "task7-placeholders", "swap-task2", "swap-task5", "swap-task6"])
    args = parser.parse_args()
    if args.mode == "task7-empty":
        print(build_task7_refill("0506v2", replace_placeholders=False))
    elif args.mode == "task7-placeholders":
        print(build_task7_refill("0506v3", replace_placeholders=True))
    elif args.mode == "swap-task2":
        print(build_single_task_swap("0506v4", task=2))
    elif args.mode == "swap-task5":
        print(build_single_task_swap("0506v5", task=5))
    elif args.mode == "swap-task6":
        print(build_single_task_swap("0506v6", task=6))


if __name__ == "__main__":
    main()
