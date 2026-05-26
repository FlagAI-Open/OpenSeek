#!/usr/bin/env python3
"""Live demo server for the OpenSeek task flow UI.

The server exposes the static frontend and a small /api/run endpoint that
reuses the repository's Qwen-backed task methods.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
import types
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import requests


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:2026")

sys.path.insert(0, str(SRC))
os.chdir(ROOT.parent)


def load_method_module(name: str):
    """Load method files under Python 3.9 without editing competition code."""
    module_path = SRC / f"{name}.py"
    source = module_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in source.splitlines()[:5]:
        source = "from __future__ import annotations\n" + source
    module = types.ModuleType(name)
    module.__file__ = str(module_path)
    sys.modules[name] = module
    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module


method_task1 = load_method_module("method_task1")
method_task2 = load_method_module("method_task2")
method_task3 = load_method_module("method_task3")
method_task5 = load_method_module("method_task5")
method_task6 = load_method_module("method_task6")
method_task7 = load_method_module("method_task7")


TASK_FILES = {
    1: DATA / "openseek-1_closest_integers.json",
    2: DATA / "openseek-2_count_nouns_verbs.json",
    3: DATA / "openseek-3_collatz_conjecture.json",
    5: DATA / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
    6: DATA / "openseek-6_mnli_same_genre_classification.json",
    7: DATA / "openseek-7_jeopardy_answer_generation_all.json",
}

METHODS = {
    1: method_task1,
    2: method_task2,
    3: method_task3,
    5: method_task5,
    6: method_task6,
    7: method_task7,
}

TASK_CACHE: dict[int, dict[str, Any]] = {}


def load_task(task_id: int) -> dict[str, Any]:
    if task_id not in TASK_CACHE:
        with TASK_FILES[task_id].open("r", encoding="utf-8") as f:
            TASK_CACHE[task_id] = json.load(f)
    return TASK_CACHE[task_id]


def parse_key_values(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def extract_list(raw: str) -> str:
    match = re.search(r"\[[^\]]*\]", raw, flags=re.DOTALL)
    if match:
        return re.sub(r"\s+", " ", match.group(0)).strip()
    return raw.strip()


def normalize_input(task_id: int, raw: str) -> str:
    raw = raw.strip()
    values = parse_key_values(raw)

    if task_id in {1, 3}:
        return extract_list(raw)

    if task_id == 2:
        if raw.lower().startswith("sentence:"):
            return raw
        target = values.get("target", "nouns").lower()
        if target in {"noun", "n"}:
            target = "nouns"
        if target in {"verb", "v"}:
            target = "verbs"
        sentence = values.get("sentence", raw).strip().strip("\"'")
        return f"Sentence: '{sentence}'. Count the number of {target} in this sentence."

    if task_id == 5:
        return values.get("tweet", raw).strip().strip("\"'")

    if task_id == 6:
        if raw.lower().startswith("sentence 1:"):
            return raw
        genre = values.get("genre", "travel").strip().rstrip(".")
        s1 = values.get("s1", values.get("sentence 1", "")).strip().strip("\"'")
        s2 = values.get("s2", values.get("sentence 2", "")).strip().strip("\"'")
        return f"Sentence 1: {s1} Sentence 2: {s2} Genre: {genre}."

    if task_id == 7:
        if raw.lower().startswith("category:"):
            return raw
        category = values.get("category", "").strip().strip("\"'")
        clue = values.get("clue", raw).strip().strip("\"'")
        return f"Category: {category}\nClue: {clue}"

    return raw


def qwen_status() -> dict[str, Any]:
    try:
        resp = requests.get(f"{QWEN_BASE_URL}/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        model = data.get("data", [{}])[0].get("id", "unknown")
        return {"ok": True, "model": model}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def build_input_prompt(task_id: int, text2annotate: str) -> tuple[str, dict[str, Any]]:
    task = load_task(task_id)
    module = METHODS[task_id]
    task_description = task["Definition"][0]
    examples = task.get("examples", [])[:100]
    meta: dict[str, Any] = {"normalized_input": text2annotate}

    if task_id == 2:
        prompt = module.build_prompt(task_description, text2annotate)
        examples_str = module.select_examples(examples, task_description, text2annotate)
        return prompt.replace("[[EXAMPLES]]", examples_str), meta

    if task_id == 7:
        retrieval_info = module.select_examples(examples, task_description, text2annotate)
        prompt = module.build_prompt(
            task_id,
            task_description,
            text2annotate,
            answer_type_hint=retrieval_info.get("detected_answer_type", ""),
            route_summary=retrieval_info.get("route_summary", ""),
            strategy_class=retrieval_info.get("strategy_class", ""),
        )
        meta.update(
            {
                "route": retrieval_info.get("strategy_class", ""),
                "examples": retrieval_info.get("selected_example_count", 0),
            }
        )
        return prompt.replace("[[EXAMPLES]]", retrieval_info.get("examples_str", "")), meta

    prompt = module.build_prompt(task_id, task_description, text2annotate)
    examples_str = module.select_examples(examples, task_description, text2annotate)
    return prompt.replace("[[EXAMPLES]]", examples_str), meta


def run_task(task_id: int, raw_input: str) -> dict[str, Any]:
    if task_id not in METHODS:
        raise ValueError(f"unsupported task: {task_id}")

    status = qwen_status()
    if not status["ok"]:
        raise RuntimeError(f"Qwen service is not available at {QWEN_BASE_URL}: {status['error']}")

    module = METHODS[task_id]
    text2annotate = normalize_input(task_id, raw_input)
    input_prompt, meta = build_input_prompt(task_id, text2annotate)
    meta["model"] = status.get("model", "unknown")

    if task_id == 2:
        prediction = module.annotate_nvidia(input_prompt)
        raw_output = "Task 2 method returned normalized prediction after Qwen long-prepass and short classifier."
    elif task_id == 7:
        prediction, raw_output = module.annotate_nvidia(
            input_prompt,
            task_id=task_id,
            debug=True,
            text2annotate=text2annotate,
        )
    else:
        prediction, raw_output = module.annotate_nvidia(
            input_prompt,
            task_id=task_id,
            debug=True,
            text2annotate=text2annotate,
        )

    return {
        "ok": True,
        "task": task_id,
        "model": status.get("model"),
        "normalized_input": text2annotate,
        "prediction": prediction,
        "output": format_label(task_id, prediction),
        "raw_output": raw_output,
        "checks": build_checks(task_id, text2annotate, prediction, meta),
        "trace": build_trace(task_id, text2annotate, prediction, raw_output, meta),
    }


def format_label(task_id: int, prediction: Any) -> str:
    if prediction is None:
        return "<label>None</label>"
    if task_id == 5:
        return f"<label>{prediction}</label>"
    return f"<label>{prediction}</label>"


def build_checks(task_id: int, text: str, prediction: Any, meta: dict[str, Any]) -> list[str]:
    common = [
        f"Qwen model: {meta.get('model', 'unknown')}",
        f"Normalized input: {text[:180]}",
    ]
    if task_id == 1:
        return common + ["真实调用 MinGapAudit-S12：Qwen 排序、差分、归约与 debug raw output。"]
    if task_id == 2:
        return common + ["真实调用 SyntaxAudit-R14：30k prepass 后回到短提示词性计数。"]
    if task_id == 3:
        return common + ["真实调用 ParityStep-C11：整表输出失败时会进入逐位置修复。"]
    if task_id == 5:
        return common + ["真实调用 AffectBoundary-WC5：边界规则由 Qwen 输出标签。"]
    if task_id == 6:
        return common + ["真实调用 GenreBridge-M24：S1 anchor 与 pair bridge 共同决定 Y/N。"]
    if task_id == 7:
        return common + [f"真实调用 AnchorCandidate-R4：route={meta.get('route', 'auto')}，examples={meta.get('examples', 0)}。"]
    return common


def build_trace(task_id: int, text: str, prediction: Any, raw_output: Any, meta: dict[str, Any]) -> list[str]:
    trace = [
        f"input normalized -> {text[:120]}",
        "prompt built -> existing method_task pipeline",
        f"qwen returned -> {str(prediction)}",
    ]
    raw = str(raw_output or "")
    if raw:
        first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
        if first_line:
            trace.append(f"raw trace -> {first_line[:140]}")
    if task_id == 7 and meta.get("route"):
        trace.insert(2, f"retrieval route -> {meta['route']}")
    trace.append(f"final label -> {prediction}")
    return trace


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "OpenSeekDemo/1.0"

    def translate_path(self, path: str) -> str:
        path = unquote(path.split("?", 1)[0].split("#", 1)[0])
        if path in {"", "/"}:
            path = "/demo/"
        return str(ROOT / path.lstrip("/"))

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self.send_json(qwen_status())
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body or "{}")
            task_id = int(payload.get("task"))
            raw_input = str(payload.get("input", ""))
            self.send_json(run_task(task_id, raw_input))
        except Exception as exc:
            self.send_json(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                },
                status=500,
            )

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo server: http://{args.host}:{args.port}/demo/")
    print(f"Qwen API: {QWEN_BASE_URL}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
