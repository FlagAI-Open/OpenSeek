/**
 * Majority-vote fusion of task2 predictions (examples_main1_task2_v2/v3/v4).
 * Run: node scripts/fuse_task2_v2_v3_v4_stats.mjs
 */
import fs from "fs";
import path from "path";

const ROOT = path.resolve(import.meta.dirname, "..");

function loadJsonl(filePath) {
  const raw = fs.readFileSync(filePath, "utf8");
  const byId = new Map();
  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    byId.set(row.example_id, row);
  }
  return byId;
}

function targetPos(row) {
  const t = row.target_pos;
  if (t) return String(t).toLowerCase();
  const inp = String(row.input || "").toLowerCase();
  if (inp.includes("number of nouns")) return "nouns";
  if (inp.includes("number of verbs")) return "verbs";
  return "unknown";
}

function normPred(x) {
  const s = String(x ?? "").trim();
  if (/^-?\d+$/.test(s)) return s;
  const m = s.match(/-?\d+/);
  return m ? m[0] : "";
}

function majVote(vals) {
  const c = new Map();
  for (const v of vals) c.set(v, (c.get(v) || 0) + 1);
  let best = 0;
  for (const n of c.values()) if (n > best) best = n;
  const tops = [...c.entries()].filter(([, n]) => n === best).map(([v]) => v);
  if (tops.length === 1) return tops[0];
  const nums = vals
    .filter((v) => v && /^-?\d+$/.test(String(v)))
    .map((v) => parseInt(v, 10))
    .sort((a, b) => a - b);
  if (nums.length === 3) return String(nums[1]);
  return tops[0] || "";
}

function main() {
  const paths = {
    v2: path.join(ROOT, "examples_main1_task2_v2", "openseek-2-examples-task2-standalone-compare.jsonl"),
    v3: path.join(ROOT, "examples_main1_task2_v3", "openseek-2-examples-task2-standalone-compare.jsonl"),
    v4: path.join(ROOT, "examples_main1_task2_v4", "openseek-2-examples-main1-noleak-compare.jsonl"),
  };
  const D = {};
  for (const [k, p] of Object.entries(paths)) {
    D[k] = loadJsonl(p);
  }
  const common = [...D.v2.keys()].filter((id) => D.v3.has(id) && D.v4.has(id));
  common.sort();

  console.log(JSON.stringify({ v2: D.v2.size, v3: D.v3.size, v4: D.v4.size, common: common.length }, null, 2));

  const stats = {
    nouns: { n: 0, vote: 0, single: { v2: { ok: 0, n: 0 }, v3: { ok: 0, n: 0 }, v4: { ok: 0, n: 0 } } },
    verbs: { n: 0, vote: 0, single: { v2: { ok: 0, n: 0 }, v3: { ok: 0, n: 0 }, v4: { ok: 0, n: 0 } } },
  };
  const overall = {
    v2: { ok: 0, n: 0 },
    v3: { ok: 0, n: 0 },
    v4: { ok: 0, n: 0 },
  };

  let goldMismatch = 0;
  for (const id of common) {
    const r2 = D.v2.get(id);
    const r3 = D.v3.get(id);
    const r4 = D.v4.get(id);
    const exp = String(r2.expected_output ?? "").trim();
    const exp4 = String(r4.expected_output ?? "").trim();
    if (exp !== exp4) goldMismatch++;
    const pos = targetPos(r2);
    if (pos !== "nouns" && pos !== "verbs") {
      console.warn("unknown pos", id, pos);
      continue;
    }
    const preds = [
      normPred(r2.model_output),
      normPred(r3.model_output),
      normPred(r4.model_output),
    ];
    const fused = majVote(preds);
    const okVote = fused === exp;
    stats[pos].n += 1;
    if (okVote) stats[pos].vote += 1;

    for (const [name, r] of [
      ["v2", r2],
      ["v3", r3],
      ["v4", r4],
    ]) {
      const p = normPred(r.model_output);
      const ok = p === exp;
      overall[name].n += 1;
      if (ok) overall[name].ok += 1;
      stats[pos].single[name].n += 1;
      if (ok) stats[pos].single[name].ok += 1;
    }
  }

  const nounN = stats.nouns.n;
  const verbN = stats.verbs.n;
  const voteNoun = stats.nouns.vote;
  const voteVerb = stats.verbs.vote;
  const voteTot = voteNoun + voteVerb;
  const totN = nounN + verbN;

  const out = {
    goldMismatch_between_v2_v4: goldMismatch,
    single_model: overall,
    by_pos_single: stats,
    vote_fusion: {
      nouns: { correct: voteNoun, total: nounN, acc: nounN ? voteNoun / nounN : 0 },
      verbs: { correct: voteVerb, total: verbN, acc: verbN ? voteVerb / verbN : 0 },
      overall: { correct: voteTot, total: totN, acc: totN ? voteTot / totN : 0 },
    },
  };
  console.log(JSON.stringify(out, null, 2));
}

main();
