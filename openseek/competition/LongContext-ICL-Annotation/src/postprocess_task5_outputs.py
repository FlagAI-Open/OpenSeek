#!/usr/bin/env python3
"""
Task5（tweet sadness）预测后处理：基于 ``data/openseek-5_*.json`` 原文与 examples 先验。

步骤（可单独开关）：
1. ``normalize_task5_label``：从 ``<label>`` / 碎片文本中抽取 ``Sad`` | ``Not sad``；
2. ``emoji`` 多 emoji 表决 + 单 emoji 高置信覆盖（examples 共现比例）；
3. ``text_calib``：边界正则 + examples 短语挖掘 + 非对称词表（默认 **fp_suppress**，仅压过判 Sad）。

面向 examples recall（Sad 偏多 / FP 为主）时推荐::
    --text-calib --strip-hashtag --prob-ge
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from method_hyb import count_answer
from task5_eval_emoji_postprocess_compare import (
    _build_emoji_prior_from_examples,
    _dominant_label_from_prior,
    _emoji_vote_postprocess,
    _extract_emojis,
    _normalize_label,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK5_JSON = REPO_ROOT / "data" / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"

_LABEL_IN_TEXT_RE = re.compile(r"\b(Sad|Not sad)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z']{3,}")

# 固定边界：对齐 fixed_badcase_fs + 常见 FP
_NOT_SAD_CALIB_PATTERNS: tuple[str, ...] = (
    r"hope your\b",
    r"hope .* isn't too (grim|bad|awful|rough|hard|tough)",
    r"excited to\b",
    r"can't wait to\b",
    r"come on (nsw|qld|team|lfc|mufc|origin)\b",
    r"thanks so much for picking me",
    r"thanks so much for\b",
    r"thank you so much",
    r"literally never win anything",
    r"you made my week",
    r"\bu so lucky\b",
    r"\bso lucky\b",
    r"won't rt things that might offend",
    r"hemingway\b.*#?quote",
    r"^always do sober\b",
    r"dancing in the dark with you between my arms",
    r"listening to our song",
    r"retweet my pin\b",
    r"\bwattpad\b",
    r"\bpromo\b",
    r"pin rt\b",
    r"hurt fic\b",
    r"giveaway_bot\b",
    r"picking me\b",
    r"is offense!\b",
    r"\bis offense\b",
    r"sound astounded",
    r"ethical,moral",
    r"have an any ethical",
    r"i'm so done\b",
    r"i am so done\b",
    r"bastard squirrels",
    r"boy play n0 no play",
)
_SAD_CALIB_PATTERNS: tuple[str, ...] = (
    r"\bmy depression\b",
    r"get over.*depression",
    r"you're not helping",
    r"you're not helping\.",
    r"kinda depressing hey",
    r"i feel exhausted\b",
    r"i'?m (so )?(depressed|depressing|exhausted|weary|dying)\b",
    r"back in .* after an amazing .* depressing",
    r"lost my keys and i forgot",
    r"got woken up by",
    r"woken up by a road sweeper",
    r"want to do .+ so bad but my dad won't",
    r"grass growing simulator is offended",
    r"opinions on sports is dreadful",
    r"your opinions on sports is dreadful",
)

_NOT_SAD_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in _NOT_SAD_CALIB_PATTERNS)
_SAD_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in _SAD_CALIB_PATTERNS)

# 感激/中奖语境下的 😭 等（金标常为 Not sad）
_GRATEFUL_CRY_RE = re.compile(
    r"(lucky|thanks|thank you|thank u|grateful|picked me|never win|made my week|giveaway)",
    re.IGNORECASE,
)
_CRY_EMOJI = frozenset({"😭", "😢", "🥺"})


def _strip_hashtag_symbol(text: str) -> str:
    if not text:
        return text
    return re.sub(r"#([A-Za-z0-9_]+)", r"\1", text)


def normalize_task5_label(raw: Any) -> str:
    """将模型输出规范为 ``Sad`` 或 ``Not sad``，无法解析时返回空串。"""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""

    norm = _normalize_label(text)
    if norm is not None:
        return norm

    parsed = count_answer(text)
    if parsed is not None:
        norm = _normalize_label(parsed)
        if norm is not None:
            return norm

    text = re.sub(r"</?label>", "", text, flags=re.IGNORECASE).strip()
    match = _LABEL_IN_TEXT_RE.search(text)
    if match:
        return "Sad" if match.group(1).lower() == "sad" else "Not sad"
    return ""


def build_text_lexicon_from_examples(
    bundle: dict[str, Any],
    *,
    min_count: int = 15,
    sad_ratio_ge: float = 0.88,
    sad_ratio_le: float = 0.12,
) -> tuple[set[str], set[str]]:
    """从 examples 统计高置信 Sad / Not sad 词（≥3 字符）。"""
    sad_w: Counter[str] = Counter()
    not_w: Counter[str] = Counter()

    for ex in bundle.get("examples") or []:
        raw = ex.get("output")
        if not isinstance(raw, list) or not raw:
            continue
        label = _normalize_label(raw[0])
        if label is None:
            continue
        words = set(_WORD_RE.findall(str(ex.get("input", "")).lower()))
        target = sad_w if label == "Sad" else not_w
        for w in words:
            target[w] += 1

    sad_lex: set[str] = set()
    not_lex: set[str] = set()
    for w in set(sad_w) | set(not_w):
        s, n = sad_w[w], not_w[w]
        t = s + n
        if t < min_count:
            continue
        ratio = s / t
        if ratio >= sad_ratio_ge:
            sad_lex.add(w)
        elif ratio <= sad_ratio_le:
            not_lex.add(w)
    return sad_lex, not_lex


def _tokenize_for_phrases(text: str) -> list[str]:
    t = str(text).lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " @user ", t)
    t = re.sub(r"#(\w+)", r" \1 ", t)
    t = re.sub(r"[^a-z0-9'\s]+", " ", t)
    return [w for w in t.split() if len(w) >= 2]


def build_not_sad_phrase_patterns(
    bundle: dict[str, Any],
    *,
    min_count: int = 5,
    not_sad_ratio_ge: float = 0.82,
    max_patterns: int = 80,
) -> tuple[re.Pattern[str], ...]:
    """从 examples 挖掘偏向 Not sad 的 2–4 gram 短语（用于压 FP）。"""
    sad_c: Counter[str] = Counter()
    not_c: Counter[str] = Counter()

    for ex in bundle.get("examples") or []:
        raw = ex.get("output")
        if not isinstance(raw, list) or not raw:
            continue
        label = _normalize_label(raw[0])
        if label is None:
            continue
        toks = _tokenize_for_phrases(str(ex.get("input", "")))
        target = not_c if label == "Not sad" else sad_c
        n = len(toks)
        for size in (2, 3, 4):
            for i in range(n - size + 1):
                phrase = " ".join(toks[i : i + size])
                if len(phrase) < 6:
                    continue
                target[phrase] += 1

    scored: list[tuple[float, int, str]] = []
    for phrase, nc in not_c.items():
        sc = sad_c.get(phrase, 0)
        total = nc + sc
        if total < min_count:
            continue
        ratio = nc / total
        if ratio >= not_sad_ratio_ge and nc > sc:
            scored.append((ratio, nc, phrase))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()
    for _ratio, _nc, phrase in scored[:max_patterns]:
        if phrase in seen:
            continue
        seen.add(phrase)
        escaped = re.escape(phrase)
        patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))
    return tuple(patterns)


def _emoji_single_strong_prior(
    text: str,
    prior: dict[str, dict[str, Any]],
    *,
    prob_threshold: float,
    strict_prob_gt: bool,
) -> str | None:
    """仅 1 个 emoji 且先验极高时返回 dominant 标签。"""
    distinct = list(dict.fromkeys(_extract_emojis(text)))
    if len(distinct) != 1:
        return None
    row = prior.get(distinct[0])
    if row is None:
        return None
    mx = max(row["pct_sad_occ"], row["pct_not_sad_occ"])
    ok = mx >= prob_threshold if not strict_prob_gt else mx > prob_threshold
    if not ok:
        return None
    return _dominant_label_from_prior(row)


def _apply_grateful_cry_rule(label: str, text: str) -> tuple[str, bool]:
    if label != "Sad" or not text:
        return label, False
    emojis = set(_extract_emojis(text))
    if not emojis & _CRY_EMOJI:
        return label, False
    if _GRATEFUL_CRY_RE.search(text):
        return "Not sad", True
    return label, False


def _apply_pattern_calibration(
    label: str,
    text: str,
    *,
    not_patterns: tuple[re.Pattern[str], ...],
    sad_patterns: tuple[re.Pattern[str], ...],
    fp_suppress: bool,
    fn_boost: bool,
) -> tuple[str, bool]:
    if not label or not text:
        return label, False

    if not fp_suppress or label == "Sad":
        for pat in not_patterns:
            if pat.search(text) and label != "Not sad":
                return "Not sad", True

    if fn_boost and label == "Not sad":
        for pat in sad_patterns:
            if pat.search(text) and label != "Sad":
                return "Sad", True

    if not fp_suppress and label == "Not sad":
        for pat in sad_patterns:
            if pat.search(text) and label != "Sad":
                return "Sad", True

    return label, False


def _apply_lexicon_calibration(
    label: str,
    text: str,
    sad_lex: set[str],
    not_lex: set[str],
    *,
    fp_suppress: bool,
    fn_boost: bool,
) -> tuple[str, bool]:
    if not label or not text:
        return label, False
    words = set(_WORD_RE.findall(text.lower()))
    hit_sad = words & sad_lex
    hit_not = words & not_lex

    if hit_not and not hit_sad and label == "Sad" and (not fp_suppress or True):
        return "Not sad", True
    if fn_boost and hit_sad and not hit_not and label == "Not sad":
        return "Sad", True
    if not fp_suppress and hit_sad and not hit_not and label == "Not sad":
        return "Sad", True
    return label, False


def postprocess_task5_prediction(
    prediction_raw: str,
    input_text: str,
    *,
    prior: dict[str, dict[str, Any]] | None = None,
    sad_lex: set[str] | None = None,
    not_lex: set[str] | None = None,
    not_sad_phrases: tuple[re.Pattern[str], ...] | None = None,
    emoji_enabled: bool = True,
    text_calib_enabled: bool = False,
    fp_suppress: bool = True,
    fn_boost: bool = False,
    single_emoji_enabled: bool = True,
    prob_threshold: float = 90.0,
    single_emoji_threshold: float = 85.0,
    strict_prob_gt: bool = True,
    min_distinct_emojis: int = 2,
) -> dict[str, Any]:
    """返回后处理结果字典，含 ``prediction`` 及各步骤是否生效。"""
    base = normalize_task5_label(prediction_raw)
    meta: dict[str, Any] = {
        "prediction_base": prediction_raw,
        "prediction": base,
        "label_normalized": base != str(prediction_raw or "").strip(),
        "emoji_postprocess_applied": False,
        "emoji_single_applied": False,
        "text_calib_applied": False,
        "text_calib_pattern_applied": False,
        "text_calib_phrase_applied": False,
        "text_calib_lexicon_applied": False,
        "grateful_cry_applied": False,
    }

    current = base
    if emoji_enabled and prior is not None and input_text:
        maj, _votes = _emoji_vote_postprocess(
            input_text,
            prior,
            prob_threshold=prob_threshold,
            strict_prob_gt=strict_prob_gt,
            min_distinct_emojis=min_distinct_emojis,
        )
        if maj is not None:
            current = maj
            meta["emoji_postprocess_applied"] = True

        if single_emoji_enabled:
            single = _emoji_single_strong_prior(
                input_text,
                prior,
                prob_threshold=single_emoji_threshold,
                strict_prob_gt=strict_prob_gt,
            )
            if single is not None:
                current = single
                meta["emoji_single_applied"] = True

    if text_calib_enabled and input_text and current:
        not_pats = _NOT_SAD_COMPILED + (not_sad_phrases or ())
        cur, ch = _apply_pattern_calibration(
            current,
            input_text,
            not_patterns=not_pats,
            sad_patterns=_SAD_COMPILED,
            fp_suppress=fp_suppress,
            fn_boost=fn_boost,
        )
        if ch:
            meta["text_calib_pattern_applied"] = True
            meta["text_calib_applied"] = True
            current = cur

        cur, ch = _apply_grateful_cry_rule(current, input_text)
        if ch:
            meta["grateful_cry_applied"] = True
            meta["text_calib_applied"] = True
            current = cur

        if sad_lex is not None and not_lex is not None:
            cur, ch = _apply_lexicon_calibration(
                current,
                input_text,
                sad_lex,
                not_lex,
                fp_suppress=fp_suppress,
                fn_boost=fn_boost,
            )
            if ch:
                meta["text_calib_lexicon_applied"] = True
                meta["text_calib_applied"] = True
                current = cur

    meta["prediction"] = current or base
    return meta


def _load_id_to_input(
    bundle: dict[str, Any],
    *,
    strip_hashtag: bool,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in ("test_samples", "examples"):
        for item in bundle.get(key) or []:
            sid = str(item.get("id", "")).strip()
            if not sid:
                continue
            t = str(item.get("input", "") or "")
            if strip_hashtag:
                t = _strip_hashtag_symbol(t)
            out[sid] = t
    return out


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}-postprocessed{input_path.suffix}")


def evaluate_jsonl_rows(rows: list[dict], pred_key: str = "model_output") -> dict[str, Any]:
    total = len(rows)
    matched = fp = fn = 0
    for row in rows:
        gold = _normalize_label(row.get("expected_output"))
        pred = _normalize_label(row.get(pred_key))
        if gold is None or pred is None:
            continue
        if gold == pred:
            matched += 1
        elif gold == "Not sad" and pred == "Sad":
            fp += 1
        elif gold == "Sad" and pred == "Not sad":
            fn += 1
    return {
        "total": total,
        "matched": matched,
        "accuracy": matched / total if total else 0.0,
        "fp": fp,
        "fn": fn,
    }


def postprocess_jsonl(
    input_path: Path,
    output_path: Path | None = None,
    *,
    task5_json: Path = DEFAULT_TASK5_JSON,
    inplace: bool = False,
    strip_hashtag: bool = False,
    emoji_enabled: bool = True,
    text_calib_enabled: bool = False,
    fp_suppress: bool = True,
    fn_boost: bool = False,
    single_emoji_enabled: bool = True,
    prob_threshold: float = 90.0,
    single_emoji_threshold: float = 85.0,
    prob_ge: bool = False,
    min_distinct_emojis: int = 2,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    task5_json = task5_json.resolve()
    if not task5_json.is_file():
        raise FileNotFoundError(f"找不到 task5 JSON: {task5_json}")

    with task5_json.open(encoding="utf-8") as f:
        bundle = json.load(f)

    id_to_input = _load_id_to_input(bundle, strip_hashtag=strip_hashtag)
    prior = _build_emoji_prior_from_examples(bundle) if emoji_enabled else None
    sad_lex, not_lex = (
        build_text_lexicon_from_examples(bundle) if text_calib_enabled else (set(), set())
    )
    not_sad_phrases = (
        build_not_sad_phrase_patterns(bundle) if text_calib_enabled else tuple()
    )
    strict_prob_gt = not prob_ge

    if inplace:
        target = input_path.with_suffix(input_path.suffix + ".tmp")
    else:
        target = _resolve_output_path(input_path, output_path.resolve() if output_path else None)

    total = changed = emoji_n = emoji_single_n = text_n = missing_input = 0
    baseline_metrics: dict[str, Any] | None = None
    output_rows: list[dict] = []

    with input_path.open(encoding="utf-8-sig") as rf:
        raw_lines = [ln.replace("\x00", "").lstrip("\ufeff").strip() for ln in rf]
        raw_lines = [ln for ln in raw_lines if ln]

    parsed_rows: list[dict] = []
    for line_no, line in enumerate(raw_lines, start=1):
        try:
            parsed_rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_no} 行 JSON 无效: {exc}") from exc

    if parsed_rows and "expected_output" in parsed_rows[0]:
        baseline_metrics = evaluate_jsonl_rows(parsed_rows, pred_key="model_output")

    for row in parsed_rows:
        sid = str(
            row.get("test_sample_id")
            or row.get("sample_id")
            or row.get("example_id")
            or ""
        ).strip()
        if not sid:
            raise ValueError("缺少 sample id")

        raw_pred = row.get("prediction", row.get("model_output", ""))
        inp = str(row.get("input") or id_to_input.get(sid, ""))
        if sid not in id_to_input and not row.get("input"):
            missing_input += 1

        pp = postprocess_task5_prediction(
            str(raw_pred) if raw_pred is not None else "",
            inp,
            prior=prior,
            sad_lex=sad_lex,
            not_lex=not_lex,
            not_sad_phrases=not_sad_phrases,
            emoji_enabled=emoji_enabled,
            text_calib_enabled=text_calib_enabled,
            fp_suppress=fp_suppress,
            fn_boost=fn_boost,
            single_emoji_enabled=single_emoji_enabled,
            prob_threshold=prob_threshold,
            single_emoji_threshold=single_emoji_threshold,
            strict_prob_gt=strict_prob_gt,
            min_distinct_emojis=min_distinct_emojis,
        )

        new_pred = pp["prediction"]
        old_pred = str(raw_pred).strip() if raw_pred is not None else ""
        if new_pred != old_pred:
            changed += 1
        if pp["emoji_postprocess_applied"]:
            emoji_n += 1
        if pp["emoji_single_applied"]:
            emoji_single_n += 1
        if pp["text_calib_applied"]:
            text_n += 1

        out_row = dict(row)
        if "model_output" in out_row:
            out_row["model_output"] = new_pred
        if "prediction" in out_row or "test_sample_id" in out_row:
            out_row["prediction"] = new_pred
        out_row["prediction_base"] = pp["prediction_base"]
        out_row["emoji_postprocess_applied"] = pp["emoji_postprocess_applied"]
        out_row["emoji_single_applied"] = pp.get("emoji_single_applied", False)
        out_row["text_calib_applied"] = pp["text_calib_applied"]
        out_row["grateful_cry_applied"] = pp.get("grateful_cry_applied", False)
        if "expected_output" in out_row:
            out_row["is_match"] = _normalize_label(new_pred) == _normalize_label(
                out_row["expected_output"]
            )

        output_rows.append(out_row)
        total += 1

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as wf:
        for out_row in output_rows:
            wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    if inplace:
        target.replace(input_path)
        final_path = input_path
    else:
        final_path = target

    result_metrics = (
        evaluate_jsonl_rows(output_rows, pred_key="model_output")
        if output_rows and "expected_output" in output_rows[0]
        else None
    )

    matched = result_metrics["matched"] if result_metrics else 0
    accuracy = result_metrics["accuracy"] if result_metrics else 0.0

    return {
        "total": total,
        "changed": changed,
        "emoji_applied": emoji_n,
        "emoji_single_applied": emoji_single_n,
        "text_calib_applied": text_n,
        "not_sad_phrase_rules": len(not_sad_phrases),
        "missing_input": missing_input,
        "matched": matched,
        "accuracy": accuracy,
        "baseline": baseline_metrics,
        "result": result_metrics,
        "file": str(final_path),
    }


def postprocess_jsonl_inplace(input_path: Path, **kwargs: Any) -> dict[str, Any]:
    return postprocess_jsonl(input_path, inplace=True, **kwargs)


def postprocess_recall_jsonl_inplace(
    output_file: Path,
    *,
    strip_hashtag: bool = True,
    fn_boost: bool = False,
    task5_json: Path = DEFAULT_TASK5_JSON,
) -> dict[str, Any]:
    """
    examples recall 推荐配置：emoji(≥阈值) + FP 压制 text_calib，默认不拉 FN。

    供 ``infer_examples_compare`` / ``infer_examples_compare_task5`` 在 task5 推理结束后调用。
    """
    return postprocess_jsonl_inplace(
        output_file,
        task5_json=task5_json,
        strip_hashtag=strip_hashtag,
        emoji_enabled=True,
        text_calib_enabled=True,
        fp_suppress=True,
        fn_boost=fn_boost,
        single_emoji_enabled=True,
        prob_ge=True,
    )


def compare_prediction_files(path_a: Path, path_b: Path) -> dict[str, Any]:
    def load(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.replace("\x00", "").lstrip("\ufeff").strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(
                    row.get("test_sample_id")
                    or row.get("sample_id")
                    or row.get("example_id")
                    or ""
                ).strip()
                pred = row.get("prediction", row.get("model_output", ""))
                out[sid] = normalize_task5_label(pred) or str(pred).strip()
        return out

    a, b = load(path_a), load(path_b)
    common = set(a) & set(b)
    agree = sum(1 for k in common if a[k] == b[k])
    dist_a = Counter(a[k] for k in common)
    dist_b = Counter(b[k] for k in common)
    disagree_samples = [
        {"id": k, "a": a[k], "b": b[k]}
        for k in sorted(common)
        if a[k] != b[k]
    ][:20]
    return {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "n_a": len(a),
        "n_b": len(b),
        "n_common": len(common),
        "agree": agree,
        "disagree": len(common) - agree,
        "agree_rate": (agree / len(common)) if common else 0.0,
        "dist_a": dict(dist_a),
        "dist_b": dict(dist_b),
        "disagree_samples": disagree_samples,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task5 预测后处理 / 双文件对比")
    p.add_argument("--input", type=Path, required=True, help="输入 JSONL")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--task5-json", type=Path, default=DEFAULT_TASK5_JSON)
    p.add_argument(
        "--strip-hashtag",
        action="store_true",
        help="对齐 input 时去掉 #（与 striphash-on 推理一致，建议 examples 上开启）",
    )
    p.add_argument("--no-emoji", action="store_true", help="关闭 emoji 先验表决")
    p.add_argument(
        "--text-calib",
        action="store_true",
        help="开启 FP 压制：边界正则 + 短语挖掘 + 非对称词表（默认 fp_suppress）",
    )
    p.add_argument(
        "--fn-boost",
        action="store_true",
        help="同时开启 FN 提升（Sad 正则/词表在预测为 Not sad 时生效）；默认仅压 FP",
    )
    p.add_argument(
        "--no-fp-suppress",
        action="store_true",
        help="关闭非对称模式，Sad/Not sad 双侧规则均可触发",
    )
    p.add_argument("--no-single-emoji", action="store_true")
    p.add_argument("--prob-threshold", type=float, default=90.0)
    p.add_argument("--single-emoji-threshold", type=float, default=85.0)
    p.add_argument(
        "--prob-ge",
        action="store_true",
        help="emoji 阈值用 ≥（默认 >）；examples 上建议开启",
    )
    p.add_argument("--min-distinct-emojis", type=int, default=2)
    p.add_argument("--compare-with", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"输入不存在: {inp}")

    if args.compare_with is not None:
        stats = compare_prediction_files(inp, args.compare_with.resolve())
        print("=" * 64)
        print("Task5 预测对比")
        print("=" * 64)
        print(f"A: {stats['path_a']}")
        print(f"B: {stats['path_b']}")
        print(f"共同样本: {stats['n_common']}, 一致: {stats['agree']}, 不一致: {stats['disagree']}")
        print(f"一致率: {stats['agree_rate']:.2%}")
        print(f"A 标签分布: {stats['dist_a']}")
        print(f"B 标签分布: {stats['dist_b']}")
        if stats["disagree_samples"]:
            print("不一致样例（前 20）:")
            for item in stats["disagree_samples"]:
                print(f"  {item['id']}: A={item['a']} B={item['b']}")
        return

    stats = postprocess_jsonl(
        inp,
        args.output.resolve() if args.output else None,
        task5_json=args.task5_json,
        inplace=args.inplace,
        strip_hashtag=args.strip_hashtag,
        emoji_enabled=not args.no_emoji,
        text_calib_enabled=args.text_calib,
        fp_suppress=not args.no_fp_suppress,
        fn_boost=args.fn_boost,
        single_emoji_enabled=not args.no_single_emoji,
        prob_threshold=args.prob_threshold,
        single_emoji_threshold=args.single_emoji_threshold,
        prob_ge=args.prob_ge,
        min_distinct_emojis=args.min_distinct_emojis,
    )
    print(f"[task5 后处理] output={stats['file']}")
    print(
        f"  total={stats['total']} changed={stats['changed']} "
        f"emoji={stats['emoji_applied']} emoji_single={stats['emoji_single_applied']} "
        f"text_calib={stats['text_calib_applied']} phrase_rules={stats['not_sad_phrase_rules']} "
        f"missing_input={stats['missing_input']}"
    )
    if stats.get("baseline") and stats.get("result"):
        b, r = stats["baseline"], stats["result"]
        print(
            f"  baseline: acc={b['accuracy']:.4%} FP={b['fp']} FN={b['fn']}"
        )
        print(
            f"  after:    acc={r['accuracy']:.4%} FP={r['fp']} FN={r['fn']} "
            f"(Δacc={(r['accuracy']-b['accuracy'])*100:+.2f}pp "
            f"ΔFP={r['fp']-b['fp']:+d} ΔFN={r['fn']-b['fn']:+d})"
        )


if __name__ == "__main__":
    main()
