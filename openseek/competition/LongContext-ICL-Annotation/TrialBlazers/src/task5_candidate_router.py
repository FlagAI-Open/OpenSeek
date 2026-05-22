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
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "openseek-5_semeval_2018_task1_tweet_sadness_detection.json"
TASK_ID = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task5 sadness candidate router.")
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
    parser.add_argument("--include-calibration-examples", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=(
            "conservative",
            "balanced_recall",
            "balanced_recall_guarded",
            "balanced_recall_guarded_v2",
            "sad_rescue_guarded",
        ),
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


def normalize_label(value: Any) -> str:
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
            rows[normalize_row_id(row)] = normalize_label(row.get("prediction"))
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
    if source.lower() in {"aggressive", "sad_biased", "old"}:
        return "negative-affect alternative"
    return source


def build_router_prompt(
    tweet: str,
    candidates: Sequence[Dict[str, str]],
    include_calibration_examples: bool,
    prompt_style: str,
) -> str:
    candidate_lines = [
        f"{candidate_letter(idx)}. [{source_label(item['source'], idx)}] {item['answer']}"
        for idx, item in enumerate(candidates)
    ]
    calibration_block = ""
    if include_calibration_examples:
        calibration_block = (
            "[Calibration Examples]\n"
            "Tweet: Damn I lost my keys and I forgot to get the garage opener\nLabel: Sad\n"
            "Tweet: Had frustration dream that left me utterly f**king furious. Plus side: so angry couldn't sleep, wrote 1500 words. Minus side: still raging!\nLabel: Sad\n"
            "Tweet: @1720maryknoll I was #fuming Kenny.\nLabel: Sad\n"
            "Tweet: Okay I seriously don't know how this whole twitter thing works #lost\nLabel: Not sad\n"
            "Tweet: Sometimes when I feel a bit depressed, I go back and watch the @Leighgriff09 free kicks against England to make me happy\nLabel: Not sad\n"
            "Tweet: @LadyScully I didn't. We all went out and got pissed down the local instead. 😄\nLabel: Not sad\n"
            "Tweet: @UKSportsZone In other words, I don't like the result of the poll so I'm packing up my polls & taking them home while I pout. 😂😂😻 #bbn\nLabel: Not sad\n\n"
        )
    if prompt_style == "balanced_recall_guarded_v2":
        decision_rules = (
            "1. Label Sad when the tweet author directly expresses personal negative affect: sadness, disappointment, frustration, anger, being upset, loneliness, boredom, grief, physical pain, insomnia, being lost/missing something, intimidation, discouragement, stress, or clear distress.\n"
            "2. If candidate A is Not sad but another candidate is Sad, choose Sad for direct first-person distress or a direct negative situation affecting the author: lost wallet/keys/charger/spectacles, head/body hurting, #pissed, fuming, frustrated, terrible rules affecting the author, or being personally intimidated.\n"
            "3. Do not switch to Sad for isolated negative words, political insults, conspiracy imagery, metaphors, news/commentary, religious/exclamatory phrases, or unclear fragments when the author's own feeling is not stated.\n"
            "4. Strong Not sad traps: sober is not sadness by itself; hell/dark/cigars in political commentary is not the author's sadness; grim/stinks is a mild object complaint unless the author is distressed; freezing/burning may be lyric or wordplay; pout can be playful; motivational advice such as don't get discouraged is not the author's sadness; a bare phrase like blood rage is too ambiguous.\n"
            "5. If candidate A is Sad and another candidate is Not sad, keep Sad unless the tweet is clearly a sports cheer, celebration, joke, quote, or non-emotional hashtag rather than negative affect.\n"
            "6. Candidate A is strong, but do not keep A when a Sad alternative clearly captures the author's own distress.\n"
            "7. If the Sad evidence is only a word-level association or the tweet is ambiguous, choose A.\n"
        )
    elif prompt_style == "sad_rescue_guarded":
        decision_rules = (
            "1. Candidate A is a high-precision current prediction. The main job is to rescue clear Sad cases when A says Not sad; do not freely rewrite A.\n"
            "2. If candidate A is Sad, keep A by default. Switch from A=Sad to Not sad only for unmistakable quotes, lyrics, jokes, slogans, sports/political/news commentary, third-person events, insults aimed at others, or wordplay with no author distress.\n"
            "3. If candidate A is Not sad but another candidate is Sad, choose Sad for direct personal negative affect or a direct bad situation affecting the author: #sad/#sadness, depressed/depressing, gloomy, crying, upsetting, lost wallet/keys/important item, missing something, head/body pain, insomnia, discouraged, frustrated, pissed, fuming, unhappy, intimidated, or clear distress.\n"
            "4. Choose Sad for dataset-style broad negative affect when the author reacts personally, including fandom/sports disappointment, gloomy days, sad film/programme reactions, and explicit sadness hashtags.\n"
            "5. Do not rescue to Sad for traps: sober, dark/gritty, bitter as banter, a bare #lost TV/person reference, cannot frown, starbucks wanting, quotes, religious/motivational advice, political insults, generic suspicion/panic, or insults where the author is not distressed.\n"
            "6. If only one weak alternative says Sad and the evidence is just a sad-sounding word, choose A.\n"
            "7. If both labels remain plausible, choose A.\n"
        )
    elif prompt_style == "balanced_recall_guarded":
        decision_rules = (
            "1. Label Sad when the tweet author directly expresses personal negative affect: sadness, disappointment, frustration, anger, being upset, loneliness, boredom, grief, physical pain, insomnia, being lost/missing something, intimidation, discouragement, stress, or clear distress.\n"
            "2. If candidate A is Not sad but another candidate is Sad, choose Sad for direct first-person distress or a direct negative situation affecting the author: lost wallet/keys/charger, head/body hurting, #pissed, fuming, frustrated, terrible rules hurting the author, or being personally intimidated.\n"
            "3. Do not switch to Sad for isolated negative words, political insults, metaphors, news/commentary, religious/exclamatory phrases, or unclear fragments when the author's own feeling is not stated.\n"
            "4. Guarded Not sad traps: sober is not sadness by itself; hell/dark/grim/dreadful/blood/rage are not enough in political or descriptive language; freezing/burning can be lyric/wordplay; pout can be playful; encouragement such as don't get discouraged is not the author's sadness.\n"
            "5. Keep Not sad for jokes, quotes, song/literary lines, insults aimed at someone else, third-person sadness, commentary about others, and motivational advice.\n"
            "6. Candidate A is strong, but do not keep A when a Sad alternative clearly captures the author's own distress.\n"
            "7. If the Sad evidence is only a word-level association or the tweet is ambiguous, choose A.\n"
        )
    elif prompt_style == "balanced_recall":
        decision_rules = (
            "1. Label Sad when the tweet author expresses personal negative affect: sadness, disappointment, frustration, anger, being upset, loneliness, boredom, grief, physical pain, insomnia, being lost/missing something, discouragement, stress, or clear distress.\n"
            "2. If candidate A is Not sad but another candidate is Sad, choose Sad when the author's own negative feeling or negative situation is direct, even if the tweet is casual, uses lol, or is short.\n"
            "3. Strong Sad cues include I lost/missed/can't find something important, my head/body hurts, I am frustrated/fuming/pissed/unhappy/bored/alone/tired of this, I can't sleep, or something ruined the author's day.\n"
            "4. Keep Not sad for jokes, quotes, song/literary lines, insults aimed at someone else, third-person sadness, news/commentary about others, or sadness words used only as references.\n"
            "5. Words like bitter, grim, dull, dark, frown, pout, lost, sad, or depression are not enough by themselves; decide whether the author personally feels bad in this tweet.\n"
            "6. Candidate A is strong, but do not keep A when a Sad alternative better matches clear personal distress.\n"
            "7. If the tweet is playful, sarcastic, or ambiguous rather than personally distressed, choose A.\n"
        )
    else:
        decision_rules = (
            "1. Label Sad when the tweet author expresses personal negative affect: sadness, disappointment, frustration, anger, being upset, loneliness, boredom, grief, physical pain, insomnia, being lost/missing something, discouragement, stress, or clear distress.\n"
            "2. Complaints can be Sad when the author is personally annoyed, pissed, fuming, unhappy, hurt, discouraged, tired of something, or emotionally affected.\n"
            "3. Label Not sad for jokes, quotes, song/literary lines, insults aimed at someone else, third-person sadness, news/commentary about others, or a mere sadness-related word without the author's own negative feeling.\n"
            "4. Words like bitter, grim, dull, dark, frown, pout, lost, sad, or depression are not enough by themselves; decide whether the author personally feels bad in this tweet.\n"
            "5. Candidate A is the current strongest prediction. Choose another candidate only when the tweet clearly supports it.\n"
            "6. If both labels are plausible, choose A.\n"
        )
    return (
        "You are resolving candidate labels for a tweet sadness/negative-affect annotation task.\n"
        "Return exactly one capital letter from the candidate list. Do not write the label. Do not explain.\n\n"
        f"{calibration_block}"
        "[Tweet]\n"
        f"{tweet.strip()}\n\n"
        "[Candidate Labels]\n"
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


def request_router(tweet: str, candidates: Sequence[Dict[str, str]], args: argparse.Namespace) -> Tuple[str, str]:
    prompt = build_router_prompt(tweet, candidates, args.include_calibration_examples, args.prompt_style)
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

    output_file = output_dir / "openseek-5-v1.jsonl"
    decisions_file = output_dir / "task5_router_decisions.jsonl"
    summary_file = output_dir / "task5_router_summary.json"

    def route_one(sample: Dict[str, Any]) -> Dict[str, Any]:
        row_id = str(sample.get("id"))
        tweet = str(sample.get("input", ""))
        candidates = build_unique_candidates(row_id, candidate_predictions)
        if not candidates:
            candidates = [{"source": candidate_specs[0][0], "answer": ""}]

        raw_response = ""
        selected_idx = 0
        prompt = ""
        if len(candidates) > 1:
            try:
                raw_response, prompt = request_router(tweet, candidates, args)
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
                "input": tweet,
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="Task5 router", unit="sample"):
            idx = futures[future]
            rows.append({"idx": idx, "row": future.result()})

    rows.sort(key=lambda item: item["idx"])
    with output_file.open("w", encoding="utf-8") as writer, decisions_file.open("w", encoding="utf-8") as decision_writer:
        for item in rows:
            row = item["row"]
            writer.write(json.dumps({"test_sample_id": row["test_sample_id"], "prediction": row["prediction"]}, ensure_ascii=False) + "\n")
            decisions.append(row["_decision"])
            decision_writer.write(json.dumps(row["_decision"], ensure_ascii=False) + "\n")

    source_counts: Dict[str, int] = {}
    for item in decisions:
        source = item["selected_source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    summary: Dict[str, Any] = {
        "output_file": str(output_file),
        "decisions_file": str(decisions_file),
        "samples": len(rows),
        "candidate_sources": [name for name, _ in candidate_specs],
        "routed_samples": sum(1 for item in decisions if len(item["candidates"]) > 1),
        "selected_by_source": dict(sorted(source_counts.items())),
    }

    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
