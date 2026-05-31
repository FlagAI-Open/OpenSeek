import re

from method_hyb import annotate_nvidia


GENRE_DESC: dict[str, str] = {
    "face-to-face": "content related to in-person dialogue or conversation",
    "government": "public information from government sources/websites",
    "letters": "charity fundraising letter-style writing",
    "9/11": "content related to the September 11 attacks",
    "slate": "culture-topic writing from Slate magazine style",
    "telephone": "telephone conversation transcripts",
    "travel": "travel guide style content",
    "verbatim": "short linguistics-focused prose",
    "oup": "non-fiction writing about textile industry and child development",
    "fiction": "popular novel-style narrative writing",
}


def build_task6_v2_single_sentence_prompt(sentence: str, genre: str) -> str:
    """Judge whether ONE sentence matches ONE genre (Y/N)."""
    g = (genre or "").strip().lower()
    g_desc = GENRE_DESC.get(g, "unknown genre definition")
    return (
        "### Role\n"
        "You are a strict genre validator for OpenSeek task6.\n\n"
        "### Task\n"
        "Given one sentence and one candidate genre, decide whether the sentence fits that genre.\n\n"
        "### Candidate Genre\n"
        f"- genre: {genre}\n"
        f"- definition: {g_desc}\n\n"
        "### Decision Rules\n"
        "1. Use style/register/source cues first, not topic overlap.\n"
        "2. If evidence is weak or ambiguous, output N.\n"
        "3. Output only one label: Y or N.\n\n"
        "### Input Sentence\n"
        f"{sentence}\n\n"
        "### Output Format\n"
        "Return exactly one label wrapped in tags: <label>Y</label> or <label>N</label>.\n"
    )


def normalize_binary_label(text: str) -> str:
    t = " ".join((text or "").strip().split()).upper()
    if t in {"Y", "N"}:
        return t
    m = re.search(r"<LABEL>\s*([YN])\s*</LABEL>", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([YN])\b", t)
    if m:
        return m.group(1)
    return ""


def extract_task6_parts(input_text: str) -> tuple[str, str, str]:
    """Parse `Sentence 1`, `Sentence 2`, `Genre` from task6 input line."""
    s1_match = re.search(r"Sentence 1:\s*(.*?)\s*Sentence 2:", input_text, flags=re.IGNORECASE | re.DOTALL)
    s2_match = re.search(r"Sentence 2:\s*(.*?)\s*Genre:", input_text, flags=re.IGNORECASE | re.DOTALL)
    genre_match = re.search(r"Genre:\s*([^\.\n]+)\.?", input_text, flags=re.IGNORECASE)
    s1 = s1_match.group(1).strip() if s1_match else ""
    s2 = s2_match.group(1).strip() if s2_match else ""
    genre = genre_match.group(1).strip() if genre_match else ""
    return s1, s2, genre


def judge_sentence_genre(sentence: str, genre: str) -> str:
    prompt = build_task6_v2_single_sentence_prompt(sentence=sentence, genre=genre)
    raw = annotate_nvidia(prompt)
    pred = "" if raw is None else str(raw).strip()
    return normalize_binary_label(pred)


__all__ = [
    "annotate_nvidia",
    "extract_task6_parts",
    "judge_sentence_genre",
    "normalize_binary_label",
]
