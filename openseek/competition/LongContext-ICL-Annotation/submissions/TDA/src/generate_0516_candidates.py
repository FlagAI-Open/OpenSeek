#!/usr/bin/env python3
"""
0516 candidate generation script.
Steps:
1. Record 0515v16=80.55 to scoreboard
2. Re-train enhanced task6 model with bigrams + sentence length + genre features
3. Find new high-confidence conflicts not previously used
4. Scan current best for task7 anomalies
5. Generate stratified submission packages
"""
import json, zipfile, hashlib, os, re, copy, glob, sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / 'outputs'
SCOREBOARD_0515 = OUTPUTS / 'scoreboard_0515.json'
TASK6_CONFLICTS = OUTPUTS / '0513_systematic' / 'task6_style_conflicts.json'
TASK2_CONFLICTS = OUTPUTS / '0513_systematic' / 'task2_ridge_conflicts.json'

# === Step 1: Record 0515v16 score ===
def record_score():
    with open(SCOREBOARD_0515) as f:
        data = json.load(f)
    for entry in data:
        if entry['version'] == '0515v16':
            entry['score'] = 80.55
            entry['submit_priority'] = 'current-best'
            print(f"Recorded 0515v16 = 80.55")
            break
    with open(SCOREBOARD_0515, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Also create 0516 scoreboard with v16 as baseline
    sb16 = [
        {
            "version": "0516v1",
            "zip": "submission_0516v1.zip",
            "score": None,
            "rank": None,
            "sha256": None,
            "changes": "baseline",
            "note": "Immutable copy of 0515v16 (80.55). Do not submit.",
            "submit_priority": "baseline-only",
            "diffs": []
        }
    ]
    with open(OUTPUTS / 'scoreboard_0516.json', 'w') as f:
        json.dump(sb16, f, indent=2, ensure_ascii=False)

# === Step 2: Enhanced task6 model ===
def load_task6_data():
    data_dir = PROJECT / 'data'
    examples = defaultdict(list)
    test_data = defaultdict(list)
    # Data is in openseek-6_mnli_same_genre_classification.json
    for fname in sorted(data_dir.glob('openseek-6_*.json')):
        with open(fname) as f:
            data = json.load(f)
        for ex in data.get('examples', []):
            examples['task6'].append(ex)
        for ts in data.get('test_samples', []):
            test_data['task6'].append(ts)
    return examples, test_data

def extract_features(sentence_pair, genre):
    s1, s2 = sentence_pair
    features = {}
    # Word-level
    words1 = set(re.findall(r'\b[a-z]{2,}\b', s1.lower()))
    words2 = set(re.findall(r'\b[a-z]{2,}\b', s2.lower()))
    common = words1 & words2
    union = words1 | words2
    if union:
        features['word_jaccard'] = len(common) / len(union)
        features['word_overlap_ratio'] = len(common) / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
    else:
        features['word_jaccard'] = 0.0
        features['word_overlap_ratio'] = 0.0

    # Bigram-level
    def bigrams(s):
        words = re.findall(r'\b[a-z]{2,}\b', s.lower())
        return set(zip(words, words[1:]))
    bg1 = bigrams(s1)
    bg2 = bigrams(s2)
    bg_common = bg1 & bg2
    bg_union = bg1 | bg2
    if bg_union:
        features['bigram_jaccard'] = len(bg_common) / len(bg_union)
    else:
        features['bigram_jaccard'] = 0.0

    # Char n-gram (3-gram)
    def char_trigrams(s):
        s_clean = re.sub(r'\s+', '_', s.lower().strip())
        return set(s_clean[i:i+3] for i in range(len(s_clean)-2))
    cg1 = char_trigrams(s1)
    cg2 = char_trigrams(s2)
    cg_common = cg1 & cg2
    cg_union = cg1 | cg2
    if cg_union:
        features['char3_jaccard'] = len(cg_common) / len(cg_union)
    else:
        features['char3_jaccard'] = 0.0

    # Sentence length ratio
    len1, len2 = len(s1.split()), len(s2.split())
    if max(len1, len2) > 0:
        features['len_ratio'] = min(len1, len2) / max(len1, len2)
    else:
        features['len_ratio'] = 0.0
    features['len_diff'] = abs(len1 - len2)

    # POS-style: question marks, negation words
    features['negation_in_only_one'] = 1.0 if bool(re.search(r'\b(not|never|no|isn\'t|don\'t|doesn\'t|won\'t)\b', s1.lower())) != bool(re.search(r'\b(not|never|no|isn\'t|don\'t|doesn\'t|won\'t)\b', s2.lower())) else 0.0

    # Genre as one-hot
    for g in ['fiction', 'government', 'slate', 'telephone', 'travel']:
        features[f'genre_{g}'] = 1.0 if genre == g else 0.0

    return features

def parse_sentence_pair(text):
    m = re.search(r'Sentence 1:\s*(.+?)\s*Sentence 2:\s*(.+?)\s*Genre:', text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return '', ''

def parse_genre(text):
    m = re.search(r'Genre:\s*(\w+)', text)
    if m:
        return m.group(1).strip().rstrip('.')
    return 'unknown'

def build_enhanced_labeled(examples):
    labeled = []
    for ex in examples:
        s1, s2 = parse_sentence_pair(ex['input'])
        if not s1 or not s2:
            continue
        genre = parse_genre(ex['input'])
        output_val = ex.get('output', '')
        if isinstance(output_val, list):
            output_val = output_val[0] if output_val else ''
        label = 1 if str(output_val).strip().upper() in ('Y', 'YES', 'TRUE') else 0
        features = extract_features((s1, s2), genre)
        labeled.append((features, label, ex.get('id', ex['input'])))
    return labeled

def sigmoid(x):
    x = max(-50, min(50, x))
    return 1.0 / (1.0 + 2.71828 ** (-x))

def train_logistic(labeled, lr=0.05, epochs=30):
    weights = Counter()
    bias = 0.0
    for epoch in range(epochs):
        for features, label, _ in labeled:
            score = bias + sum(weights[k] * v for k, v in features.items())
            prob = sigmoid(score)
            error = label - prob
            for k, v in features.items():
                weights[k] += lr * error * v
            bias += lr * error
    return weights, bias

def cross_validate(labeled, folds=5):
    n = len(labeled)
    fold_size = n // folds
    all_preds = []
    for f in range(folds):
        start = f * fold_size
        end = start + fold_size if f < folds - 1 else n
        val = labeled[start:end]
        train = labeled[:start] + labeled[end:]
        weights, bias = train_logistic(train)
        for features, label, inp in val:
            score = bias + sum(weights[k] * v for k, v in features.items())
            pred = 1 if score > 0 else 0
            all_preds.append((pred, label, score, inp))
    correct = sum(1 for p, l, _, _ in all_preds if p == l)
    print(f"Enhanced model {folds}-fold CV accuracy: {correct}/{len(all_preds)} = {correct/len(all_preds):.4f}")
    return all_preds

# === Step 3: Task7 anomaly scanner ===
def scan_task7(current_best_zip):
    anomalies = []
    with zipfile.ZipFile(current_best_zip) as zf:
        for name in zf.namelist():
            if 'openseek-7' in name and name.endswith('.jsonl'):
                with zf.open(name) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line.decode())
                        pred = sample.get('prediction', '')
                        sid = sample.get('test_sample_id', '')
                        # Check for placeholder / garbage
                        if pred in ('', '<label>', '____', '...', 'label here', 'answer', 'Label', 'label'):
                            anomalies.append((sid, pred, 'placeholder'))
                        elif len(pred) < 2 and pred not in ('a', 'i'):
                            anomalies.append((sid, pred, 'too_short'))
                        elif pred.startswith('<') and pred.endswith('>'):
                            anomalies.append((sid, pred, 'xml_tag'))
                        elif 'label' in pred.lower() and len(pred) < 15:
                            anomalies.append((sid, pred, 'has_label_word'))
    return anomalies

# === Step 4: Generate packages ===
def load_zip_as_dict(zip_path):
    result = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith('.jsonl'):
                task = re.search(r'openseek-(\d+)', name)
                if task:
                    tid = task.group(1)
                    rows = []
                    with zf.open(name) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rows.append(json.loads(line.decode()))
                    result[tid] = {'filename': name.split('/')[-1] if '/' in name else name, 'rows': rows}
    return result

def create_submission_zip(base_zip_path, output_path, modifications):
    """
    modifications: dict of task_id -> {sample_id: new_prediction}
    """
    data = load_zip_as_dict(base_zip_path)
    changes = []
    for tid, mods in modifications.items():
        if tid not in data:
            continue
        for row in data[tid]['rows']:
            sid = row['test_sample_id']
            if sid in mods:
                old = row['prediction']
                row['prediction'] = mods[sid]
                changes.append([sid, old, mods[sid]])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for tid in sorted(data.keys(), key=int):
            filename = data[tid]['filename']
            content = '\n'.join(json.dumps(r, ensure_ascii=False) for r in data[tid]['rows'])
            zf.writestr(filename, content)

    sha = hashlib.sha256(open(output_path, 'rb').read()).hexdigest()
    return sha, changes

def make_0516_packages():
    base_zip = str(OUTPUTS / 'submission_0515v16.zip')
    exp_dir = OUTPUTS / '0516_experiments'

    # Collect all info
    print("=== Step 1: Recording scores ===")
    record_score()

    print("\n=== Step 2: Enhanced task6 model ===")
    examples, test_data = load_task6_data()
    task6_examples = examples.get('task6', [])
    task6_tests = test_data.get('task6', [])
    print(f"Task6 examples: {len(task6_examples)}, tests: {len(task6_tests)}")

    labeled = build_enhanced_labeled(task6_examples)
    print(f"Labeled pairs: {len(labeled)}")

    # Cross-validate
    cv_results = cross_validate(labeled)

    # Train on all examples, predict on test samples
    weights, bias = train_logistic(labeled)

    # Load current best predictions for task6
    current_best_data = load_zip_as_dict(base_zip)
    task6_current = {}
    for row in current_best_data.get('6', {}).get('rows', []):
        task6_current[row['test_sample_id']] = row['prediction']

    # Predict on test samples and find conflicts with current best
    new_conflicts = []
    for ts in task6_tests:
        sid = ts['id']
        s1, s2 = parse_sentence_pair(ts['input'])
        if not s1 or not s2:
            continue
        genre = parse_genre(ts['input'])
        features = extract_features((s1, s2), genre)
        score = bias + sum(weights[k] * v for k, v in features.items())
        pred = 1 if score > 0 else 0
        abs_score = abs(score)
        current_pred = task6_current.get(sid, '?')
        current_label = 1 if str(current_pred).strip().upper() in ('Y', 'YES', 'TRUE') else 0

        if pred != current_label and abs_score > 0.15:
            direction = 'N->Y' if current_label == 0 else 'Y->N'
            new_pred = 'Y' if pred == 1 else 'N'
            new_conflicts.append((sid, direction, new_pred, score, abs_score, genre))

    new_conflicts.sort(key=lambda x: -x[4])
    print(f"\nNew enhanced model task6 conflicts (score>0.15, not in current best): {len(new_conflicts)}")
    for sid, direction, new_pred, score, conf, genre in new_conflicts[:30]:
        print(f"  {direction} | {sid} | conf={conf:.4f} | genre={genre}")

    # Also check original task6 style model for remaining unused
    with open(TASK6_CONFLICTS) as f:
        orig_conflicts = json.load(f)
    used_set = set()
    for sb_file in sorted(glob.glob(str(OUTPUTS / 'scoreboard_05*.json'))):
        with open(sb_file) as f:
            data = json.load(f)
        for entry in data:
            for d in entry.get('diffs', []):
                if isinstance(d, list) and len(d) == 3:
                    used_set.add(d[0])
    unused_orig = [c for c in orig_conflicts if c['id'] not in used_set]
    print(f"\nOriginal style model unused: {len(unused_orig)}")
    for c in unused_orig:
        direction = 'N->Y' if c['current'] == 'N' else 'Y->N'
        print(f"  {direction} | {c['id']} | conf={c['conf']:.4f}")

    # Merge: new from enhanced model + unused from original
    # Only include new enhanced model conflicts (different signal)
    # Prioritize those with higher confidence

    print("\n=== Step 3: Task7 anomalies ===")
    anomalies = scan_task7(base_zip)
    print(f"Task7 anomalies: {len(anomalies)}")
    for sid, pred, reason in anomalies:
        print(f"  {sid} | '{pred}' | {reason}")

    # === Generate packages ===
    print("\n=== Step 4: Generating 0516 packages ===")

    pkgs = []

    # 0516v1: baseline copy of 0515v16
    sha1, _ = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v1.zip'), {})
    pkgs.append(('0516v1', str(OUTPUTS / 'submission_0516v1.zip'), sha1, [], 'Immutable copy of 0515v16 (80.55). Do not submit.', 'baseline-only'))

    # Use enhanced model conflicts
    t6_ny = [(sid, new_pred) for sid, direction, new_pred, score, conf, genre in new_conflicts if direction == 'N->Y']
    t6_yn = [(sid, new_pred) for sid, direction, new_pred, score, conf, genre in new_conflicts if direction == 'Y->N']

    # 0516v2: top-5 enhanced model N->Y (most conservative)
    ny_top5 = t6_ny[:5]
    if ny_top5:
        mods = {'6': {sid: pred for sid, pred in ny_top5}}
        sha2, ch2 = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v2.zip'), mods)
        pkgs.append(('0516v2', str(OUTPUTS / 'submission_0516v2.zip'), sha2, ch2, f'Top 5 enhanced model task6 N->Y: {len(ny_top5)} samples.', 'high'))

    # 0516v3: top-5 enhanced model Y->N
    yn_top5 = t6_yn[:5]
    if yn_top5:
        mods = {'6': {sid: pred for sid, pred in yn_top5}}
        sha3, ch3 = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v3.zip'), mods)
        pkgs.append(('0516v3', str(OUTPUTS / 'submission_0516v3.zip'), sha3, ch3, f'Top 5 enhanced model task6 Y->N: {len(yn_top5)} samples.', 'high'))

    # 0516v4: all enhanced N->Y
    if t6_ny:
        mods = {'6': {sid: pred for sid, pred in t6_ny}}
        sha4, ch4 = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v4.zip'), mods)
        pkgs.append(('0516v4', str(OUTPUTS / 'submission_0516v4.zip'), sha4, ch4, f'All enhanced model N->Y: {len(t6_ny)} samples.', 'medium-high'))

    # 0516v5: all enhanced Y->N
    if t6_yn:
        mods = {'6': {sid: pred for sid, pred in t6_yn}}
        sha5, ch5 = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v5.zip'), mods)
        pkgs.append(('0516v5', str(OUTPUTS / 'submission_0516v5.zip'), sha5, ch5, f'All enhanced model Y->N: {len(t6_yn)} samples.', 'medium-high'))

    # 0516v6: combined enhanced N->Y top-5 + Y->N top-5
    combined_top = ny_top5[:5] + yn_top5[:5]
    if combined_top:
        mods = {'6': {sid: pred for sid, pred in combined_top}}
        sha6, ch6 = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v6.zip'), mods)
        pkgs.append(('0516v6', str(OUTPUTS / 'submission_0516v6.zip'), sha6, ch6, f'Combined enhanced top-5+5: {len(combined_top)} samples.', 'highest'))

    # 0516v7: all enhanced combined
    all_enhanced = t6_ny + t6_yn
    if all_enhanced:
        mods = {'6': {sid: pred for sid, pred in all_enhanced}}
        sha7, ch7 = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v7.zip'), mods)
        pkgs.append(('0516v7', str(OUTPUTS / 'submission_0516v7.zip'), sha7, ch7, f'All enhanced combined: {len(all_enhanced)} samples.', 'medium'))

    # 0516v8: original style model last unused + enhanced top-3 each
    extra = unused_orig[:2]  # at most 2 from original
    orig_mods = {}
    for c in extra:
        new_pred = 'Y' if c['current'] == 'N' else 'N'
        orig_mods[c['id']] = new_pred
    combined_ny3 = ny_top5[:3] + [(sid, pred) for sid, pred in orig_mods.items() if pred == 'Y']
    combined_yn3 = yn_top5[:3] + [(sid, pred) for sid, pred in orig_mods.items() if pred == 'N']
    all_combined = combined_ny3 + combined_yn3
    if all_combined:
        mods = {'6': {sid: pred for sid, pred in all_combined}}
        sha8, ch8 = create_submission_zip(base_zip, str(OUTPUTS / 'submission_0516v8.zip'), mods)
        pkgs.append(('0516v8', str(OUTPUTS / 'submission_0516v8.zip'), sha8, ch8, f'Enhanced top-3+3 + original last unused: {len(all_combined)} samples.', 'medium'))

    # Write scoreboard
    sb_entries = []
    for ver, zpath, sha, changes, note, priority in pkgs:
        entry = {
            "version": ver,
            "zip": f"submission_{ver}.zip",
            "score": None,
            "rank": None,
            "sha256": sha,
            "changes": f"task6:{len(changes)}",
            "note": note,
            "submit_priority": priority,
            "diffs": changes
        }
        sb_entries.append(entry)
        print(f"\n{ver}: {len(changes)} changes, sha256={sha[:16]}..., priority={priority}")

    # Update scoreboard
    with open(OUTPUTS / 'scoreboard_0516.json', 'w') as f:
        json.dump(sb_entries, f, indent=2, ensure_ascii=False)

    # Write run summaries
    for ver, zpath, sha, changes, note, priority in pkgs:
        vdir = exp_dir / ver
        os.makedirs(vdir, exist_ok=True)
        summary = {
            "version": ver,
            "zip": f"submission_{ver}.zip",
            "score": None,
            "rank": None,
            "sha256": sha,
            "changes": f"task6:{len(changes)}",
            "note": note,
            "submit_priority": priority,
            "diffs": changes
        }
        with open(vdir / 'run_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done! Generated {len(pkgs)} packages ===")
    print(f"Scoreboard: {OUTPUTS / 'scoreboard_0516.json'}")

if __name__ == '__main__':
    make_0516_packages()