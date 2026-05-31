/**
 * task2 test_samples：多份预测 JSONL 按 test_sample_id 交集做多数票融合。
 *
 * 默认四路：v3 zeroshot、v3 prompt1、v5、v5b。
 *
 *   node scripts/fuse_task2_test_samples_spacy_vote.mjs
 *   node scripts/fuse_task2_test_samples_spacy_vote.mjs --tie 0 --out outputs/custom.jsonl a.jsonl b.jsonl ...
 *
 * --tie：平局时采用第几个输入文件（0-based）的预测；默认 2（第三路 v5）。
 */
import fs from "fs";
import path from "path";

const ROOT = path.resolve(import.meta.dirname, "..");

function loadJsonl(absPath) {
  const raw = fs.readFileSync(absPath, "utf8");
  const byId = new Map();
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      continue;
    }
    const id = String(row.test_sample_id ?? row.example_id ?? "").trim();
    if (id) byId.set(id, row);
  }
  return byId;
}

function normPred(x) {
  const s = String(x ?? "").trim();
  if (/^\d+$/.test(s)) return s;
  const m = s.match(/\d+/);
  return m ? m[0] : "";
}

function majVote(vals, tieIdx) {
  const norm = vals.map((v) => normPred(v));
  const counts = new Map();
  for (const v of norm) {
    if (!v) continue;
    counts.set(v, (counts.get(v) || 0) + 1);
  }
  if (counts.size === 0) return norm[tieIdx] || norm.find(Boolean) || "";
  let best = -1;
  for (const c of counts.values()) if (c > best) best = c;
  const tops = [...counts.entries()].filter(([, c]) => c === best).map(([v]) => v);
  if (tops.length === 1) return tops[0];
  const pick = norm[tieIdx] || norm.find(Boolean) || "";
  return pick;
}

function parseArgs(argv) {
  const out = { tie: 2, output: null, paths: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--tie") {
      out.tie = parseInt(argv[++i], 10) || 0;
    } else if (a === "--out") {
      out.output = argv[++i];
    } else if (a.startsWith("-")) {
      console.error("未知参数:", a);
      process.exit(1);
    } else {
      out.paths.push(a);
    }
  }
  return out;
}

function defaultPaths() {
  const d = path.join(ROOT, "outputs");
  return [
    path.join(d, "openseek-2-test_samples-task2-spacy-v3-zeroshot-predictions.jsonl"),
    path.join(d, "openseek-2-test_samples-task2-spacy-v3-prompt1-zeroshot-predictions.jsonl"),
    path.join(d, "openseek-2-test_samples-task2-spacy-v5-zeroshot-predictions.jsonl"),
    path.join(d, "openseek-2-test_samples-task2-spacy-v5b-zeroshot-predictions.jsonl"),
  ];
}

function main() {
  const { tie, output: outArg, paths: pathArgs } = parseArgs(process.argv.slice(2));
  const paths = pathArgs.length ? pathArgs : defaultPaths();
  const resolved = paths.map((p) => (path.isAbsolute(p) ? p : path.join(ROOT, p)));
  for (const p of resolved) {
    if (!fs.existsSync(p)) {
      console.error(`[fuse_task2_test_samples_spacy_vote] 文件不存在:\n  ${p}`);
      process.exit(1);
    }
  }
  const maps = resolved.map((p) => loadJsonl(p));

  let inter = null;
  for (const m of maps) {
    const s = new Set(m.keys());
    if (inter === null) inter = s;
    else {
      for (const id of inter) {
        if (!s.has(id)) inter.delete(id);
      }
    }
  }
  const ids = [...inter].sort();

  const outPath =
    outArg ||
    path.join(
      ROOT,
      "outputs",
      "openseek-2-test_samples-task2-spacy-vote4-v3-v3p1-v5-v5b-predictions.jsonl",
    );
  const absOut = path.isAbsolute(outPath) ? outPath : path.join(ROOT, outPath);

  const labels = ["v3", "v3_prompt1", "v5", "v5b"];
  const perFile = maps.map(() => 0);
  let emptyAfterVote = 0;
  let unanimous = 0;
  let split = 0;

  const lines = [];
  for (const id of ids) {
    const rows = maps.map((m) => m.get(id));
    const preds = rows.map((r) => r.prediction);
    const normed = preds.map((p) => normPred(p));
    const fused = majVote(preds, tie);
    if (!fused) emptyAfterVote++;
    const uniq = new Set(normed.filter(Boolean));
    if (uniq.size <= 1) unanimous++;
    else split++;

    lines.push(
      JSON.stringify({
        test_sample_id: id,
        prediction: fused,
        vote_sources: labels,
        vote_predictions: normed,
        vote_tie_index: tie,
      }),
    );
  }

  fs.mkdirSync(path.dirname(absOut), { recursive: true });
  fs.writeFileSync(absOut, lines.join("\n") + "\n", "utf8");

  const n = ids.length;
  const summary = {
    intersection: n,
    tie_index: tie,
    tie_source: labels[tie] ?? String(tie),
    output: absOut,
    inputs: resolved.map((p) => path.relative(ROOT, p)),
    unanimous_all_four_agree: unanimous,
    split_at_least_two_differ: split,
    fused_empty: emptyAfterVote,
  };
  console.log(JSON.stringify(summary, null, 2));
  resolved.forEach((p, i) => {
    console.log(`单路[${i}] ${labels[i] ?? i} ${path.basename(p)}: ${maps[i].size} 条`);
  });
}

main();
