/**
 * Task7 v4 JSONL 错题启发式分类 + Category 聚合统计。
 * 用法: node scripts/analyze_task7_v4_badcase_clusters.mjs [path/to/v4.jsonl]
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..");
const DEFAULT_JSONL = path.join(
  REPO,
  "examples",
  "openseek-7-examples-compare-task7opt-v4.jsonl"
);

/** @param {string} s */
function normalizeAnswer(s) {
  return String(s || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

/** 简化版 V3 base：去引号类、非 alnum（保留 & -）、去冠词 */
function normalizeMatchBase(s) {
  let t = normalizeAnswer(s);
  if (!t) return "";
  t = t.normalize("NFD").replace(/\p{M}/gu, "");
  t = t.replace(/["'`“”‘’]/g, "");
  t = t.replace(/[^a-z0-9\s/&-]/g, " ");
  t = t.replace(/\b(the|a|an)\b/g, " ");
  t = t.replace(/\s+/g, " ").trim();
  return t;
}

function hyphenSpaceUnify(s) {
  if (!s) return "";
  return s
    .replace(/[\u002d\u2010\u2011\u2212\u2013\u2014]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function ampersandToAnd(s) {
  if (!s || !s.includes("&")) return s;
  let x = s.replace(/\s*&\s*\/\s*or\b/gi, " or ");
  x = x.replace(/\s*&\s*/g, " and ");
  return x.replace(/\s+/g, " ").trim();
}

function normalizeMatchV3(s) {
  let t = normalizeMatchBase(s);
  if (!t) return "";
  t = ampersandToAnd(t);
  return hyphenSpaceUnify(t);
}

/** @returns {Set<string>} */
function expandExpectedVariants(expected) {
  const variants = new Set();
  const base = normalizeMatchV3(expected);
  if (base) variants.add(base);
  let raw = normalizeAnswer(expected);
  if (!raw) return variants;
  const normalized = raw
    .replace(/&\/or/gi, " or ")
    .replace(/and\/or/gi, " or ")
    .replace(/& or/gi, " or ");
  const parts = normalized.split(/\b|,|;/).map((p) => p.trim()).filter(Boolean);
  for (const part of parts) {
    const p = part.replace(/^[,;/\s]+|[,;/\s]+$/g, "").trim();
    if (!p) continue;
    const v = normalizeMatchV3(p);
    if (v) variants.add(v);
  }
  // 再按 " or " 拆（gold 常见）
  for (const piece of [...variants]) {
    for (const sub of piece.split(/\bor\b/).map((x) => x.trim()).filter(Boolean)) {
      const v = normalizeMatchV3(sub);
      if (v) variants.add(v);
    }
  }
  return variants;
}

function isTask7MatchV3(expected, prediction) {
  const predNorm = normalizeMatchV3(prediction);
  if (!predNorm) return false;
  const vars = expandExpectedVariants(expected);
  if (!vars.size) return false;
  if (vars.has(predNorm)) return true;
  for (const v of vars) {
    if (predNorm.endsWith(" " + v) || v.endsWith(" " + predNorm)) return true;
  }
  return false;
}

/** alpha 长度>1 的词 */
function contentTokens(s) {
  const m = normalizeAnswer(s).match(/[a-z0-9]+/g);
  return m ? m.filter((t) => t.length > 1) : [];
}

function tokenSets(s) {
  return new Set(contentTokens(s));
}

/** pred 的词在 clue 中出现的比例（v4 clue_echo 风格） */
function clueEchoRatio(pred, clue) {
  const pt = tokenSets(pred);
  const ct = tokenSets(clue);
  if (!pt.size || !ct.size) return 0;
  let inter = 0;
  for (const x of pt) if (ct.has(x)) inter++;
  return inter / pt.size;
}

/** pred 规整后是否像在照抄 clue 里的片段 */
function looksLikeClueSubstring(pred, clue) {
  const p = normalizeMatchBase(pred).replace(/\s+/g, " ");
  const c = normalizeMatchBase(clue).replace(/\s+/g, " ");
  if (p.length < 4 || c.length < 8) return false;
  if (c.includes(p) && p.length >= 10) return true;
  const ptoks = [...tokenSets(pred)];
  if (ptoks.length >= 2) {
    const joined = ptoks.slice(0, 5).join(" ");
    const sub = joined.length >= 8 && c.includes(joined.replace(/ /g, ""));
    // 宽松：clue 中连续包含 pred 的大部分词
    const hit = ptoks.filter((t) => c.includes(t)).length;
    if (hit / ptoks.length >= 0.85 && ptoks.length >= 3) return true;
    if (sub) return true;
  }
  return false;
}

function hasMarkupNoise(s) {
  return /<\/?[a-z][a-z0-9_-]*\b/i.test(s) || /<\/[^>]+>/.test(s) || /<[a-z][^>]{0,40}>/i.test(s);
}

function classifyFormat(pred, exp) {
  const p = String(pred || "").trim();
  if (!p) return { format: true, reasons: ["empty"] };
  if (hasMarkupNoise(p)) return { format: true, reasons: ["xml_or_bracket_markup"] };
  if (/\n/.test(p)) return { format: true, reasons: ["multiline"] };
  const expLen = normalizeAnswer(exp).length;
  if (p.length > 95 && expLen > 0 && p.length > expLen * 3)
    return { format: true, reasons: ["very_long_vs_gold"] };
  return { format: false, reasons: [] };
}

/** 字面已错，但与 gold 规整串「很像」—— 归为近别名 / 等价未覆盖 */
function nearAliasWrong(expected, pred) {
  if (isTask7MatchV3(expected, pred)) return false;
  const pn = normalizeMatchV3(pred);
  const variants = [...expandExpectedVariants(expected)];
  if (!pn || !variants.length) return false;

  /** @param {string} a */
  function diceBigrams(a) {
    if (a.length < 2) return new Map();
    const m = new Map();
    for (let i = 0; i < a.length - 1; i++) {
      const bg = a.slice(i, i + 2);
      m.set(bg, (m.get(bg) || 0) + 1);
    }
    return m;
  }
  function diceSim(a, b) {
    if (!a.length || !b.length) return 0;
    const A = diceBigrams(a);
    const B = diceBigrams(b);
    let inter = 0;
    let sum = 0;
    for (const [k, v] of A) sum += v;
    for (const [k, v] of B) sum += v;
    for (const [k, v] of A) {
      if (B.has(k)) inter += Math.min(v, B.get(k));
    }
    return (2 * inter) / sum;
  }

  for (const v of variants) {
    if (!v) continue;
    const r = pn.length ? Math.min(pn.length, v.length) / Math.max(pn.length, v.length) : 0;
    const dice = diceSim(pn.replace(/\s/g, ""), v.replace(/\s/g, ""));
    const levRatio =
      pn.length || v.length
        ? levNorm(pn.replace(/\s/g, ""), v.replace(/\s/g, ""))
        : 1;
    // 短串：要求高 dice；长短差不太大且编辑距离比例低
    if (dice >= 0.72 && r >= 0.55) return true;
    if (levRatio <= 0.28 && r >= 0.5 && Math.min(pn.length, v.length) >= 5) return true;
  }

  const pTokens = new Set(normalizeMatchV3(pred).split(/\s+/).filter(Boolean));
  for (const v of variants) {
    const vt = new Set(v.split(/\s+/).filter(Boolean));
    if (!pTokens.size || !vt.size) continue;
    const inter = [...pTokens].filter((x) => vt.has(x)).length;
    const uni = new Set([...pTokens, ...vt]).size;
    const ji = uni ? inter / uni : 0;
    if (ji >= 0.85 && vt.size <= 10) return true;
  }
  return false;
}

function levNorm(a, b) {
  if (a === b) return 0;
  const m = a.length;
  const n = b.length;
  if (!m || !n) return 1;
  const dp = Array(n + 1);
  for (let j = 0; j <= n; j++) dp[j] = j;
  for (let i = 1; i <= m; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= n; j++) {
      const tmp = dp[j];
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + cost);
      prev = tmp;
    }
  }
  return dp[n] / Math.max(m, n);
}

function parseCategoryClue(input) {
  const s = String(input || "").trim();
  const re = /^Category:\s*(.+?)\s*\n\s*Clue:\s*(.*)$/is;
  const m = s.match(re);
  if (!m) return { category: "", clue: s };
  return { category: m[1].trim(), clue: m[2].trim() };
}

/**
 * Category → 单个粗桶（自上而下首条命中）；减少「一行多桶」计数膨胀。
 */
function categoryPrimaryBucket(cat) {
  const rules = [
    [/SONG|MUSIC|LYRIC|PRINCE|TREBEK|ALBUM|TUNE|JAZZ|OPERA|BAND|SINGER|SINGLES|TRACKS/i, "music_media"],
    [/GEOGRAPHY|CONTINENTS|WORLD CAPITAL|WORLD HERITAGE|NATIONAL PARK|LATIN AMERICA|AFRICA|EUROPE|ASIA|^HI\b|MIDEAST|MIDWEST|SOUTH AMERICA|ISLAND\b|SEA\b|OCEAN\b|ATLANTIC|PACIFIC|MOUNTAINS|VOLCAN/i, "geo_macro"],
    [
      /\bU\.?\s*S\.?\s*CITIES|\bWORLD CITIES|^SKY HIGH|LAKES|RIVERS|STATES\b|HIGHWAYS|MAPPED|TRAVEL|MUSEUM|^KANSAS CITY|^NEW YORK|^LONDON|SUBWAY|^HI\b/i,
      "geo_place_detail",
    ],
    [
      /SHAKESPEARE|AUTHOR|BOOKS\b|DRAMA\b|THEATER|POTTERY|QUOTE|MOVIE|MUSEUM|OPERAS|PODCAST|SCREEN|SILENT FILM|TITLES|TROPES|SCREENPLAY|SERIALS|ANIMATION|TOLKIEN|TREKKIES|WHO SAID|TURN OF PHRASE|ANAGRAM|PALINDROME/i,
      "arts_book_film_quote",
    ],
    [/TV\b|TELEVISION|BBC|CNN|NETWORK|PROGRAM|SOAP|STAR TREK|WHOSE LINE|DOCUMENTARY|SITCOM|SURVIVOR|EPISODES/i, "tv"],
    [/POETRY|POET|VERSE|RHAPSODY IN|ODE TO/i, "poetry"],
    [/ACTOR|ACTRESS|STARS|CELEBRITY|HISTORIAN|SCULPTORS|PHOTOGRAPHY|THE ARTS|MUSEUM|^ART\b|^DANCING|COMEDIANS|SINGLES.*CELEBRITY/i, "celebrity_arts_bio"],
    [
      /\bSCIENCE\b|MEDICINE|CHEMISTRY|PHYSICS|BIOLOGY|ANATOMY|ASTRONOMY|MATH|MACHINE|TECH|PSYCHOLOGY|INVENT|ELEMENT\b|VOLTS|BOTANY|SCIENCE FACT|WEATHER|SCIENCE FACT|EXPERIMENT|^DNA\b|^RNA\b|^ATOM|^GENE/i,
      "stem",
    ],
    [
      /\bWAR\b|CIVIL WAR|^WW|WORLD WAR|MILITARY|REVOLUTION|SIEGE|TREATY|SOLDIER|ARMY|^NFL BLITZ|ADMIRAL|SHERMAN|LEE\b|HAMILTON.*DUEL|MIL\. POWER|MIL\b/i,
      "military_conflict",
    ],
    [
      /PRESIDENTIAL|FIRST LAD|ELECTION|WHITE HOUSE|SENATORS|REPRESENTATIVES|SLOGANS|S\.HALL|S\.H\b|FOUNDING FATHER|WHO'S AFRAID|UNIVERSITIES|^\d{4}s?\s|COLLEGES|SCHOOL|EDUCATION|EXPLORATION|WORLD LEADERS|SULTAN|^QUEEN|^EMPER|^KING\b|PHARAOH|AUTOCRAT/i,
      "history_polity_edu_bio",
    ],
    [/OLYMP|SPORTS|^NBA|^NFL|NHL|TENNIS|SOCCER|BASEBALL|GOLF|SURF|RACE\b|MARATHON|TOUR DE|STADIUM|ATHLETE|COACH|ESPN|SPORT/i, "sports"],
    [/AUTO\b|VEHICLES|MOTOR|MERCEDES|MERCURY|MARQUE|MERCURY.*CAR|SPEEDWAY| NASCAR/i, "autos"],
    [/FOOD|VEGETABLE|FRUIT|CHEF|WINE|^COFFEE|BEER|SODA|SUGAR|SALT|SANDWICH|SUSHI|TACO|SALAD|SUGAR|SUGAR|SUGARY/i, "food_drink"],
    [/WORD\s|WORDS\b|WORDS WITH|WORD ORIGINS|DEFINITION|LANGUAGE|LINGUIST|LINGO|PRONUNCIATION|SPELL|SPOKEN|FROM THE GERMAN|EUPHEM|SAYINGS|HOMONYMS|SILENT|TONGUES|LINGUISTS/i, "language_word"],
    [
      /\bRHYME\b|RHYMES|RHYMES WITH|HOMOPHONES|HOMOPH|^BY HALVES|^FIX THE|^FAMILIAR PHRASES|COMPLETE THE|COMPLETE THIS|COMPLETE THE|TITLES.*SPOIL|SILENT|TONGUE|TONGUE.?TW|^X'S AND|^Y'S|^Z'S|SPOONSER|ANAGRAMS|BEFORE & AFTER|MIXED|MASHUP|DOUBLE TROUBLE|SPOONSER|SPOONER/i,
      "puzzle_wordplay",
    ],
    [/BIBLE|SACRAMENT|ECCLESI|^POPE|SABBAT|GENESIS|SHEPHERDS|CHRIST|^ISLAM|BHAGAVAD|TALMUD|HANUKKAH|RELIGIONS/i, "religion"],
    [/BUSINESS|MONEY|^STOCK|S\.E\.|NYSE|ECONOM|MILLIONAIR|IPO|NASDAQ|SILICON|INDUSTRY|FACTORY|AUTOMAKER|AUTO INDUSTRY/i, "business_econ"],
    [/LEGENDS|TALE\b|LEGENDARY|SUPER\b|MARVEL|MARVEL|TITANS|ALIASES|TITLES FOR CLIVE|WHO IS CLIVE/i, "trivia_entities"],
    [/\bGAMES\b|BOARD GAMES|TREBEK|JEOPARDY|ANNUAL EVENTS|AWARDS\b|TOURNAMENT|POKER|CHESS|KING.*CHECK|DICE|RANDOM|RANDOMIZED/i, "games_events"],
    [/ANIMAL|DOG|CAT\b|HORSE|FISH|BIRD|INSECT|ZOO|WILDLIFE|REPTILE|LION|ELEPHANT|WHALE|POODLE|TERRIER|MAMMALS?|VENOM/i, "animals"],
    [/FASHION|DESIGNER|JEANS|FABRIC|TATTOOS|COSMETIC|COSMET|^MAKEUP|UNDERWEAR|FOOTWEAR|HATS|SNEAKER/i, "fashion_design"],
    [/GEOLOGY|GEM|MINERAL|ROCKS|SILVER GOLD|SILVER|MERCURY|ELEMENT NAMES/i, "earth_materials"],
  ];
  for (const [rx, tag] of rules) {
    if (rx.test(cat)) return tag;
  }
  return "other_uncategorized";
}

/** 主次分类（互斥优先级） */
function primaryErrorType(row) {
  const pred = row.model_output ?? "";
  const exp = row.expected_output ?? "";
  const { category, clue } = parseCategoryClue(row.input ?? "");

  const fmt = classifyFormat(pred, exp);
  if (fmt.format) return { type: "格式/抽取问题", reasons: fmt.reasons, category, clue };

  const echo = clueEchoRatio(pred, clue);
  const substring = looksLikeClueSubstring(pred, clue);
  const strongEcho = echo >= 0.55 && contentTokens(pred).length >= 2;
  if (strongEcho || (substring && echo >= 0.35)) {
    return { type: "抄题干/题干显性短语", reasons: [`echo_ratio≈${echo.toFixed(2)}`, substring ? "substring_like" : "token_overlap"].filter(Boolean), category, clue };
  }

  if (nearAliasWrong(exp, pred)) {
    return { type: "近别名或未覆盖等价", reasons: ["dice_edit_or_tokens_high_vs_gold"], category, clue };
  }

  return { type: "知识/实体错误或理解偏题", reasons: ["no_structural_match"], category, clue };
}

function incr(map, key) {
  map.set(key, (map.get(key) || 0) + 1);
}

function pct(a, b) {
  return b ? ((100 * a) / b).toFixed(2) + "%" : "0%";
}

const jsonlPath = process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_JSONL;
const raw = fs.readFileSync(jsonlPath, "utf8");
/** @type {any[]} */
const rows = [];
for (const line of raw.split(/\r?\n/)) {
  const t = line.trim();
  if (!t) continue;
  try {
    rows.push(JSON.parse(t));
  } catch {
    // skip
  }
}

let wrong = 0;
const byPrimary = new Map();
const byBucketAndType = new Map(); // `${bucket}\t${type}`，每桶错题条数可加总至 wrong
const byCategoryPrefix = new Map();

for (const row of rows) {
  if (row.is_match === true) continue;
  wrong++;
  const r = primaryErrorType(row);
  incr(byPrimary, r.type);

  const bucket = categoryPrimaryBucket(r.category);
  incr(byBucketAndType, `${bucket}\t${r.type}`);
  const pref = r.category.slice(0, 56).replace(/\s+/g, " ");
  incr(byCategoryPrefix, pref);
}

const total = rows.length;

console.log("=== Task7 V4 错题聚类（启发式） ===\n");
console.log(`文件: ${jsonlPath}`);
console.log(`总条数: ${total} | 错题: ${wrong} (${pct(wrong, total)}) | 正确: ${total - wrong}\n`);

console.log("--- 全局：错误类型占比（主次互斥优先级：格式 → 抄题干 → 近别名 → 知识）---");
for (const [t, n] of [...byPrimary.entries()].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${t}: ${n} (${pct(n, wrong)})`);
}

console.log("\n--- Category 关键词粗桶 × 错误类型（每条错题只归一个桶，可加总=%wrong） ---");
/** @type {Map<string, Map<string, number>>} */
const bucketToTypes = new Map();
for (const [k, n] of byBucketAndType) {
  const [bucket, ...rest] = k.split("\t");
  const errType = rest.join("\t");
  if (!bucketToTypes.has(bucket)) bucketToTypes.set(bucket, new Map());
  bucketToTypes.get(bucket).set(errType, (bucketToTypes.get(bucket).get(errType) || 0) + n);
}
const bucketOrder = [
  "other_uncategorized",
  "stem",
  "history_polity_edu_bio",
  "celebrity_arts_bio",
  "arts_book_film_quote",
  "tv",
  "poetry",
  "geo_place_detail",
  "geo_macro",
  "puzzle_wordplay",
  "language_word",
  "games_events",
  "music_media",
  "military_conflict",
  "business_econ",
  "food_drink",
  "animals",
  "sports",
  "autos",
  "religion",
  "fashion_design",
  "earth_materials",
  "trivia_entities",
];
for (const b of bucketOrder) {
  const m = bucketToTypes.get(b);
  if (!m) continue;
  const sum = [...m.values()].reduce((a, x) => a + x, 0);
  console.log(`\n[${b}] 错题 ${sum} (${pct(sum, wrong)} of wrong)`);
  for (const [t, n] of [...m.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${t}: ${n}`);
  }
}

console.log("\n--- 错题最多的 Category（前 25，按原文字符截断）---");
for (const [c, n] of [...byCategoryPrefix.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25)) {
  console.log(`  ${n}\t${c}`);
}
