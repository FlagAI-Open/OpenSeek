import json
import re
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
PREFIX = "from __future__ import annotations\nimport torch\nimport torch.nn.functional as F\n\n\n"


def call_qwen(prompt: str, max_tokens: int = 900) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stop": ["END_CODE", "```"],
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(API_URL, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 + attempt)
    return f"ERROR: {last_error}"


def clean_instruction(text: str) -> str:
    match = re.search(r"Wrapper Entry Information:(.*?)(?:After generation|$)", text, flags=re.S)
    return match.group(1).strip() if match else text


def strip_to_code(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^BEGIN_CODE\s*", "", text, flags=re.I).strip()
    start = text.find("def ")
    if start == -1:
        return text
    text = text[start:]
    # Remove common trailing prose if the model appended it.
    cut_markers = ["\nExplanation:", "\nThe code", "\nThis code", "\nNote:"]
    cuts = [text.find(marker) for marker in cut_markers if text.find(marker) != -1]
    if cuts:
        text = text[: min(cuts)]
    return text.strip()


def build_prompt(instruction: str, signature: str, error: str | None = None, previous: str | None = None) -> str:
    retry = ""
    if error:
        retry = (
            "\nPrevious code failed Python compile with this error:\n"
            f"{error}\n"
            "Fix the code. Do not explain.\n"
        )
    if previous:
        retry += f"\nPrevious code:\n{previous[:1600]}\n"
    return (
        "You are implementing a PyTorch reference function for a benchmark.\n"
        "You must output only the function body/source code after the imports already provided.\n"
        "Do not write analysis, comments about correctness, markdown, or prose.\n"
        "Use exactly this function signature:\n"
        f"def {signature}\n\n"
        "Implementation rules:\n"
        "- Use torch and torch.nn.functional as F.\n"
        "- Implement the described math literally and simply.\n"
        "- If there is an out parameter, copy result to out and return out.\n"
        "- End the answer with END_CODE.\n"
        f"{retry}\n"
        "Wrapper description:\n"
        f"{instruction}\n\n"
        "BEGIN_CODE\n"
        f"{PREFIX}"
    )


def compile_code(sample_id: str, code: str) -> str:
    try:
        compile(code, sample_id, "exec")
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def generate_one(sample_id: str, instruction: str, signature: str, attempts: int = 4) -> dict[str, Any]:
    error = None
    previous = None
    raw = ""
    code = ""
    for attempt in range(1, attempts + 1):
        raw = call_qwen(build_prompt(instruction, signature, error=error, previous=previous))
        body = strip_to_code(raw)
        code = PREFIX + body
        error = compile_code(sample_id, code)
        if not error:
            return {"id": sample_id, "compile_ok": True, "attempts": attempt, "error": "", "code": code, "raw": raw[:500]}
        previous = code
    return {"id": sample_id, "compile_ok": False, "attempts": attempts, "error": error, "code": code, "raw": raw[:500]}


def main() -> None:
    priority = json.loads((SYSTEM_DIR / "task8_qwen_priority.json").read_text(encoding="utf-8"))
    samples = {sample["id"]: sample for sample in load_task(8)["test_samples"]}
    outputs = []
    for index, (sample_id, name, _msg) in enumerate(priority, 1):
        parsed = parse_wrapper_signature(samples[sample_id]["input"])
        if not parsed:
            result = {"id": sample_id, "name": name, "compile_ok": False, "attempts": 0, "error": "signature_parse_failed", "code": ""}
        else:
            parsed_name, params = parsed
            signature = f"{parsed_name}({params}):"
            result = generate_one(sample_id, clean_instruction(samples[sample_id]["input"]), signature)
            result["name"] = name
        outputs.append(result)
        print(index, "/", len(priority), sample_id, name, "compile", result["compile_ok"], "attempts", result.get("attempts"), result.get("error", ""), flush=True)
    out_path = SYSTEM_DIR / "task8_qwen_reference_candidates_v3.json"
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
