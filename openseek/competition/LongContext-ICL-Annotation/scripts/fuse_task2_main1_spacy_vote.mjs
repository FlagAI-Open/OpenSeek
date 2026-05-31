/**
 * main1 task2：多份 compare JSONL 按 example_id 交集做多数票融合。
 *
 * 默认五路：spacy v3 zeroshot、v5、v5a、v5b、v5c（均须已推理生成）。
 *
 *   node scripts/fuse_task2_main1_spacy_vote.mjs
 *   node scripts/fuse_task2_main1_spacy_vote.mjs --tie 1 --out examples_main1/custom_vote.jsonl  a.jsonl b.jsonl ...
 *
 * --tie：平局时采用第几个输入文件（0-based）的预测；默认 1（第二路，即 v5）。
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
    const id = String(row.example_id ?? "").trim();
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
  const out = { tie: 1, output: null, paths: [] };
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
  const d = path.join(ROOT, "examples_main1");
  return [
    path.join(d, "openseek-2-examples-main1-compare_spacy_v3_zeroshot.jsonl"),
    path.join(d, "openseek-2-examples-main1-compare_spacy_v5_zeroshot.jsonl"),
    path.join(d, "openseek-2-examples-main1-compare_spacy_v5a_zeroshot.jsonl"),
    path.join(d, "openseek-2-examples-main1-compare_spacy_v5b_zeroshot.jsonl"),
    path.join(d, "openseek-2-examples-main1-compare_spacy_v5c_zeroshot.jsonl"),
  ];
}

function taskType(inp) {
  const s = String(inp || "").toLowerCase();
  if (s.includes("nouns")) return "nouns";
  if (s.includes("verbs")) return "verbs";
  return "other";
}

function main() {
  const { tie, output: outArg, paths: pathArgs } = parseArgs(process.argv.slice(2));
  const paths = pathArgs.length ? pathArgs : defaultPaths();
  const resolved = paths.map((p) => (path.isAbsolute(p) ? p : path.join(ROOT, p)));
  for (const p of resolved) {
    if (!fs.existsSync(p)) {
      console.error(`[fuse_task2_main1_spacy_vote] 文件不存在，请先完成推理:\n  ${p}`);
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
      "examples_main1",
      "openseek-2-examples-main1-compare_vote5_task2_v3_v5_v5abc.jsonl",
    );
  const absOut = path.isAbsolute(outPath) ? outPath : path.join(ROOT, outPath);

  let fusedOk = 0;
  const by = { nouns: [0, 0], verbs: [0, 0], other: [0, 0] };
  const perFile = maps.map(() => 0);

  const lines = [];
  for (const id of ids) {
    const rows = maps.map((m) => m.get(id));
    const base = rows[0];
    const preds = rows.map((r) => r.model_output);
    const fused = majVote(preds, tie);
    const exp = String(base.expected_output ?? "").trim();
    const ok = fused === exp;
    if (ok) fusedOk++;
    const tt = taskType(base.input);
    by[tt][0]++;
    if (ok) by[tt][1]++;
    rows.forEach((r, i) => {
      if (normPred(r.model_output) === exp) perFile[i]++;
    });
    lines.push(
      JSON.stringify({
        example_id: id,
        input: base.input,
        expected_output: exp,
        model_output: fused,
        is_match: ok,
        vote_inputs: resolved.map((p) => path.relative(ROOT, p)),
        vote_model_outputs: preds.map((p) => normPred(p)),
        vote_tie_index: tie,
      }),
    );
  }

  fs.mkdirSync(path.dirname(absOut), { recursive: true });
  fs.writeFileSync(absOut, lines.join("\n") + "\n", "utf8");

  const n = ids.length;
  console.log(JSON.stringify({ intersection: n, tie_index: tie, output: absOut, inputs: resolved }, null, 2));
  resolved.forEach((p, i) => {
    console.log(`单路[${i}] ${path.basename(p)}: ${perFile[i]}/${n} = ${((perFile[i] / n) * 100).toFixed(2)}%`);
  });
  console.log(`融合: ${fusedOk}/${n} = ${((fusedOk / n) * 100).toFixed(2)}%`);
  for (const k of ["nouns", "verbs", "other"]) {
    const [tot, ok] = by[k];
    if (tot) console.log(`  ${k}: ${ok}/${tot} = ${((ok / tot) * 100).toFixed(2)}%`);
  }
}

main();
