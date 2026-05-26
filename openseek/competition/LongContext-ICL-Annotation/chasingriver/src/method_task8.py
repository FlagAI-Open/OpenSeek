from __future__ import annotations

import json
import os
import sys
from pathlib import Path

"""Task 8 entry wired to the ca2a7b guided PyTorch-R3 pipeline.

The ca2a7b baseline implementation for Task 8 used
``task8_best_realtime.py`` with config
``pytorch_v1_r3_probe15k_then_r3_repair_semantic40``.
This flagos wrapper keeps the public
``build_prompt`` / ``select_examples`` / ``annotate_nvidia`` contract used by
``flagos/src/main_task8.py`` while delegating the actual generation loop to the
same ca2a7b implementation and config.
"""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEST_CONFIG_NAME = "pytorch_v1_r3_probe15k_then_r3_repair_semantic40"
DEFAULT_TASK_DESCRIPTION = (
    "Implementing custom algorithms or functions using Triton, and ensuring "
    "correct block masking and stride handling for memory safety."
)
PROMPT_MARKER = "__FLAGOS_TASK8_BEST__"


FLAGOS_SRC = Path(__file__).resolve().parent
if str(FLAGOS_SRC) not in sys.path:
    sys.path.insert(0, str(FLAGOS_SRC))

os.environ.setdefault("OPENSEEK_TASK8_SERVICE_URL", "http://127.0.0.1:2026/v1/completions")

import task8_best_realtime_ca2a7b as _best  # noqa: E402


BEST_CONFIG = _best.CONFIGS[BEST_CONFIG_NAME]


def _pack_payload(task_description: str, text2annotate: str) -> str:
    return PROMPT_MARKER + "\n" + json.dumps(
        {
            "task_description": task_description,
            "text2annotate": text2annotate,
            "config_name": BEST_CONFIG_NAME,
        },
        ensure_ascii=False,
    )


def _unpack_payload(input_prompt: str) -> tuple[str, str]:
    if input_prompt.startswith(PROMPT_MARKER):
        raw = input_prompt.split("\n", 1)[1] if "\n" in input_prompt else "{}"
        payload = json.loads(raw)
        return (
            payload.get("task_description") or DEFAULT_TASK_DESCRIPTION,
            payload.get("text2annotate") or "",
        )
    return DEFAULT_TASK_DESCRIPTION, input_prompt


def build_prompt(task_description: str, text2annotate: str) -> str:
    return _pack_payload(task_description, text2annotate)


def select_examples(all_examples: list[dict], task_description: str, text2annotate: str) -> str:
    del all_examples, task_description, text2annotate
    return ""


def annotate_nvidia(input_prompt: str) -> str | None:
    task_description, query = _unpack_payload(input_prompt)
    config = dict(BEST_CONFIG)
    expected_wrapper = _best._extract_wrapper_name(query)
    prediction, info = _best.iterative_code_generation(
        task_description=task_description,
        query=query,
        examples=[],
        config=config,
        expected_wrapper=expected_wrapper,
        sample_id="flagos-task8",
    )

    if config.get("final_semantic_refine"):
        prediction, _ = _best.final_semantic_refine_prediction(
            query=query,
            prediction=prediction,
            config=config,
            expected_wrapper=expected_wrapper,
            sample_id="flagos-task8",
        )

    prediction = _best._sanitize_candidate_code(prediction, config=config)
    if prediction is None:
        return _best.NULL_COMPILE_FAIL_PREDICTION
    return prediction


def annotate_ascend(input_prompt: str) -> str | None:
    return annotate_nvidia(input_prompt)
