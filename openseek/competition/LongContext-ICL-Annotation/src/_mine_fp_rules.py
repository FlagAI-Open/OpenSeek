import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INP = REPO / "examples/openseek-5-examples-compare-emoji-off-striphash-on.jsonl"
TASK5 = REPO / "data/openseek-5_semeval_2018_task1_tweet_sadness_detection.json"

rows = [json.loads(l) for l in INP.read_text(encoding="utf-8").splitlines() if l.strip()]
fp = [r for r in rows if r["expected_output"] == "Not sad" and r["model_output"] == "Sad"]
fn = [r for r in rows if r["expected_output"] == "Sad" and r["model_output"] == "Not sad"]
ok_sad = [r for r in rows if r["expected_output"] == "Sad" and r["model_output"] == "Sad"]
ok_not = [r for r in rows if r["expected_output"] == "Not sad" and r["model_output"] == "Not sad"]

print("FP", len(fp), "FN", len(fn))


def words(t):
    return set(re.findall(r"[a-z']{3,}", t.lower()))


# mine words over-represented in FP vs ok_sad
fp_w = Counter()
ok_sad_w = Counter()
for r in fp:
    for w in words(r["input"]):
        fp_w[w] += 1
for r in ok_sad:
    for w in words(r["input"]):
        ok_sad_w[w] += 1

candidates = []
for w, c in fp_w.items():
    if c < 8:
        continue
    rate_fp = c / (c + ok_sad_w[w])
    if rate_fp >= 0.55 and fp_w[w] >= ok_sad_w[w]:
        candidates.append((rate_fp, c, w))
candidates.sort(reverse=True)
print("FP-skew words:", candidates[:25])

# test patterns
PATS = [
    r"so lucky",
    r"never win anything",
    r"made my week",
    r"u so lucky",
    r"is offense",
    r"offense!",
    r"astounded",
    r"ethical,moral",
    r"retweet my pin",
    r"wattpad",
    r"promo",
    r"dancing in the dark",
    r"between my arms",
    r"i'm so done",
    r"so done 😡",
    r"literally never win",
    r"thanks so much",
    r"thank you so much",
    r"congrats",
    r"congratulations",
    r"love this",
    r"can't wait",
    r"excited",
    r"lol\b",
    r"lmao\b",
    r"😂",
    r"😡",
    r"💕",
    r"🙈",
    r"offended\b",  # might hurt FN
    r"is dreadful",  # gold sad sometimes
    r"opinions on sports",
    r"sound astounded",
    r"have an any ethical",
]

for p in PATS:
    rx = re.compile(p, re.I)
    hit_fp = sum(1 for r in fp if rx.search(r["input"]))
    hit_fn = sum(1 for r in fn if rx.search(r["input"]))
    hit_ok_sad = sum(1 for r in ok_sad if rx.search(r["input"]))
    if hit_fp >= 3 and hit_fp > hit_fn + hit_ok_sad:
        print(f"PAT {p!r}: fp={hit_fp} fn={hit_fn} ok_sad={hit_ok_sad}")
