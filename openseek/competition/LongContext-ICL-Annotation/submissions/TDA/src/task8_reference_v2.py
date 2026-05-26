import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

from competition_optimize import load_task, parse_wrapper_signature


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
SYSTEM_DIR = OUTPUT_DIR / "0513_systematic"
API_URL = "http://0.0.0.0:2026/v1/completions"
MODEL_NAME = "../Qwen3-4B"


def call_qwen(prompt: str, max_tokens: int = 1200) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(API_URL, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as exc:  # noqa: BLE001 - long-running generation should retry.
            last_error = exc
            time.sleep(2 + attempt)
    return f"ERROR: {last_error}"


def clean_instruction(text: str) -> str:
    match = re.search(r"Wrapper Entry Information:(.*?)(?:After generation|$)", text, flags=re.S)
    return match.group(1).strip() if match else text


def extract_code(raw: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fenced:
        return fenced.group(1).strip()
    tagged = re.search(r"BEGIN_CODE\s*(.*?)\s*END_CODE", raw, flags=re.S | re.I)
    if tagged:
        return tagged.group(1).strip()
    start = raw.find("from __future__")
    if start == -1:
        start = raw.find("import torch")
    if start == -1:
        start = raw.find("def ")
    if start != -1:
        return raw[start:].strip()
    return raw.strip()


def build_prompt(instruction: str, error: str | None = None, previous: str | None = None) -> str:
    feedback = ""
    if error:
        feedback = (
            "\nYour previous answer was rejected because it was not valid executable Python:\n"
            f"{error}\n"
            "Return corrected code only.\n"
        )
    if previous:
        feedback += f"\nRejected previous answer:\n{previous[:1200]}\n"
    return (
        "You are writing a PyTorch reference wrapper for an operator generation benchmark.\n"
        "Return ONLY executable Python source code between BEGIN_CODE and END_CODE.\n"
        "No explanation, no markdown, no prose outside the tags.\n"
        "Rules:\n"
        "1. Use the exact function name and parameter signature from the wrapper entry.\n"
        "2. Use only torch and torch.nn.functional as F.\n"
        "3. Implement the math/operation described by the instruction, simply and correctly.\n"
        "4. If an out parameter exists, copy the result into out and return out.\n"
        "5. Include imports: from __future__ import annotations; import torch; import torch.nn.functional as F.\n"
        f"{feedback}\n"
        "Wrapper entry and task description:\n"
        f"{instruction}\n\n"
        "BEGIN_CODE\n"
    )


def generate_one(sample_id: str, instruction: str, attempts: int = 3) -> dict[str, Any]:
    previous = None
    error = None
    raw = ""
    code = ""
    for attempt in range(attempts):
        raw = call_qwen(build_prompt(instruction, error=error, previous=previous))
        code = extract_code(raw)
        try:
            compile(code, sample_id, "exec")
            return {"id": sample_id, "compile_ok": True, "attempts": attempt + 1, "error": "", "code": code, "raw": raw[:500]}
        except Exception as exc:  # noqa: BLE001 - feed compile error back.
            error = f"{type(exc).__name__}: {exc}"
            previous = code
    return {"id": sample_id, "compile_ok": False, "attempts": attempts, "error": error or "unknown", "code": code, "raw": raw[:500]}


def main() -> None:
    priority = json.loads((SYSTEM_DIR / "task8_qwen_priority.json").read_text(encoding="utf-8"))
    samples = {sample["id"]: sample for sample in load_task(8)["test_samples"]}
    outputs = []
    for index, (sample_id, name, _msg) in enumerate(priority, 1):
        parsed = parse_wrapper_signature(samples[sample_id]["input"])
        if not parsed:
            outputs.append({"id": sample_id, "name": name, "compile_ok": False, "error": "signature_parse_failed"})
            continue
        result = generate_one(sample_id, clean_instruction(samples[sample_id]["input"]))
        result["name"] = name
        outputs.append(result)
        print(index, "/", len(priority), sample_id, name, "compile", result["compile_ok"], "attempts", result.get("attempts"), result.get("error", ""), flush=True)
    out_path = SYSTEM_DIR / "task8_qwen_reference_candidates_v2.json"
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
