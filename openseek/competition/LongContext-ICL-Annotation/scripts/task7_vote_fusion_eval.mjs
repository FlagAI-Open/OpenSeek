/**
 * 多份 Task7 compare JSONL 按 model_output 多数票融合，并用 V3 规则算 is_match。
 * 平票时按文件顺序优先：task7opt → v2 → v3 → v4。
 */
import fs from "fs";
import readline from "readline";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES = path.join(__dirname, "..", "examples");

const FILES = [
  "openseek-7-examples-compare-task7opt.jsonl",
  "openseek-7-examples-compare-task7opt-v2.jsonl",
  "openseek-7-examples-compare-task7opt-v3.jsonl",
  "openseek-7-examples-compare-task7opt-v4.jsonl",
];

function normalizeText(text) {
  return String(text ?? "")
    .trim()
    .split(/\s+/)
    .join(" ");
}

function normalizeTask7Answer(text) {
  return normalizeText(text).toLowerCase();
}

function normalizeTask7MatchTextBase(text) {
  let s = normalizeTask7Answer(text);
  if (!s) return "";
  s = s.normalize("NFKD").replace(/\p{M}/gu, "");
  s = s.replace(/["'`“”‘’]/g, "");
  s = s.replace(/[^a-z0-9\s/&-]/g, " ");
  s = s.replace(/\b(the|a|an)\b/g, " ");
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

function task7HyphenSpaceUnify(text) {
  if (!text) return "";
  let s = text.replace(/[\u002d\u2010\u2011\u2212\u2013\u2014]+/g, " ");
  return s.replace(/\s+/g, " ").trim();
}

function task7AmpersandConjToAnd(text) {
  if (!text || !text.includes("&")) return text;
  let s = text.replace(/\s*&\s*\/\s*or\b/gi, " or ");
  s = s.replace(/\s*&\s*/g, " and ");
  return s.replace(/\s+/g, " ").trim();
}

function normalizeTask7MatchTextV3(text) {
  let s = normalizeTask7MatchTextBase(text);
  if (!s) return "";
  s = task7AmpersandConjToAnd(s);
  return task7HyphenSpaceUnify(s);
}

function expandExpectedVariantsV3(expected) {
  const variants = new Set();
  const base = normalizeTask7MatchTextV3(expected);
  if (base) variants.add(base);
  const raw = normalizeTask7Answer(expected);
  if (!raw) return variants;
  let normalized = raw
    .replace(/&\/or/gi, " or ")
    .replace(/and\/or/gi, " or ")
    .replace(/& or/gi, " or ");
  const parts = normalized
    .split(/\bor\b|[,;]/g)
    .map((p) => p.trim().replace(/^[,;\s]+|[,;\s]+$/g, ""))
    .filter(Boolean);
  for (const p of parts) {
    const v = normalizeTask7MatchTextV3(p);
    if (v) variants.add(v);
  }
  return variants;
}

function isTask7MatchV3(expected, prediction) {
  const predNorm = normalizeTask7MatchTextV3(prediction);
  if (!predNorm) return false;
  const expectedVariants = expandExpectedVariantsV3(expected);
  if (!expectedVariants.size) return false;
  if (expectedVariants.has(predNorm)) return true;
  for (const v of expectedVariants) {
    if (predNorm.endsWith(" " + v) || v.endsWith(" " + predNorm)) return true;
  }
  return false;
}

async function loadJsonlMap(filePath) {
  const m = new Map();
  const rs = fs.createReadStream(filePath, { encoding: "utf8" });
  const rl = readline.createInterface({ input: rs, crlfDelay: Infinity });
  for await (const line of rl) {
    const t = line.trim();
    if (!t) continue;
    let row;
    try {
      row = JSON.parse(t);
    } catch {
      continue;
    }
    const id = String(row.example_id ?? "").trim();
    if (id) m.set(id, row);
  }
  return m;
}

async function main() {
  const paths = FILES.map((f) => path.join(EXAMPLES, f));
  const maps = await Promise.all(paths.map(loadJsonlMap));
  const allIds = new Set();
  for (const m of maps) for (const id of m.keys()) allIds.add(id);
  const sortedIds = [...allIds].sort();

  let missing = 0;
  let fusedCorrect = 0;
  const perFileCorrect = maps.map(() => 0);
  let ties = 0;

  for (const id of sortedIds) {
    const rows = maps.map((m) => m.get(id));
    if (rows.some((r) => !r)) {
      missing++;
      continue;
    }
    const expected = rows[0].expected_output;
    for (let i = 1; i < rows.length; i++) {
      if (rows[i].expected_output !== expected) {
        console.warn("expected_output 不一致", id, i);
      }
    }

    for (let fi = 0; fi < maps.length; fi++) {
      if (isTask7MatchV3(expected, rows[fi].model_output)) perFileCorrect[fi]++;
    }

    const preds = rows.map((r) => String(r.model_output ?? ""));
    const counts = new Map();
    for (const p of preds) counts.set(p, (counts.get(p) ?? 0) + 1);
    let maxC = 0;
    for (const c of counts.values()) if (c > maxC) maxC = c;
    const winners = [...counts.entries()].filter(([, c]) => c === maxC).map(([k]) => k);
    let bestP = winners[0];
    if (winners.length > 1) {
      ties++;
      for (const p of preds) {
        if (winners.includes(p)) {
          bestP = p;
          break;
        }
      }
    }

    if (isTask7MatchV3(expected, bestP)) fusedCorrect++;
  }

  const n = sortedIds.length - missing;
  console.log(JSON.stringify({
    total_ids: sortedIds.length,
    evaluated: n,
    missing_in_some_file: missing,
    ties_examples_max_votes_tied: ties,
    per_file_accuracy_v3: Object.fromEntries(
      FILES.map((f, i) => [f, n ? perFileCorrect[i] / n : 0])
    ),
    per_file_correct: Object.fromEntries(FILES.map((f, i) => [f, perFileCorrect[i]])),
    fusion_accuracy_v3: n ? fusedCorrect / n : 0,
    fusion_correct: fusedCorrect,
    vote_rule: "majority on exact model_output string; tie-break file order task7opt→v2→v3→v4",
  }, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
