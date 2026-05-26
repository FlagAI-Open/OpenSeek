---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-06 18:10:00 CST"
phase: "current-bottleneck-statistics"
status: "completed"
---

# Current Bottleneck Statistics

## 1. Current Official State

- Stable official peak remains `73.05`.
- Historical stable official base is `v27 = 73.05`.
- `v30 = 73.05` confirmed score parity for the audited-clean task8 maintenance subset.
- Latest live submit was `v33 = 73.00`, which is `-0.05` vs the current working base.

This means the project is not currently blocked by the absence of submit candidates in general. It is blocked by the absence of a candidate that still converts online after the `v27` plateau.

## 2. Official Submission Statistics

### 2.1 Overall official record

- Official attempts tracked here: `16`
  - Ledger entries through `v30`: `15`
  - Plus live readback `v33`: `1`
- Online directions:
  - Positive: `6`
  - Negative: `8`
  - Neutral: `2`

### 2.2 Post-`v27` plateau window

- Post-`v27` official attempts: `4`
  - `v28` task2 dependency broad: `71.68` (`-1.37`)
  - `v29` task2 dependency ultra-narrow: `73.03` (`-0.02`)
  - `v30` task8 maintenance clean subset: `73.05` (`0.00`)
  - `v33` task2 motion exact: `73.00` (`-0.05`)
- Post-`v27` improvement count: `0 / 4`
- Post-`v27` negative-or-neutral count: `4 / 4`

This is the clearest top-level bottleneck: after `v27`, every official attempt has either been too broad and unsafe, or clean but too low-amplitude to lift the total score.

### 2.3 By task line

#### Task2

- Official attempts: `7`
- Online positive: `4`
- Online negative: `3`
- Net summed official delta across tracked attempts: `+0.35`
- Best single gain: `+1.47` (`v18`)
- Worst single loss: `-1.37` (`v28`)
- Recent window after `v27`:
  - `0 / 3` positive
  - results: `-1.37`, `-0.02`, `-0.05`

Interpretation:

- Task2 is still the most historically trustworthy transfer line.
- But the current Task2 frontier is no longer behaving like the earlier `v18 -> v22 -> v23 -> v27` climb.
- It has shifted from "reliable gain source" to "historically trusted but currently plateaued and easy to waste submits on".

#### Task7

- Official attempts: `4`
- Online positive: `2`
- Online negative: `2`
- Net summed official delta across tracked attempts: `-0.02`
- Positive versions:
  - `v19`: `+0.15`
  - `v25`: `+0.03`
- Negative versions:
  - `v21`: `-0.10`
  - `v26`: `-0.10`

Interpretation:

- Task7 has real online upside, unlike task5/task6.
- But its transfer profile is fragile rather than robust.
- It remains the main remaining method frontier, but only if work targets the actual mechanism bottleneck instead of another rerank tweak.

#### Task8

- Official attempts: `1`
- Online positive: `0`
- Online negative: `0`
- Online neutral: `1`
- Result: `v30 = 73.05` (`0.00`)

Interpretation:

- Task8 looks maintenance-safe.
- It does not currently look like a score-lifting frontier.

#### Task5

- Official attempts: `1`
- Online positive: `0`
- Online negative: `1`
- Result: `v17 = -0.10`

#### Task6

- Official attempts: `1`
- Online positive: `0`
- Online negative: `1`
- Result: `v20 = -0.17`

Interpretation:

- Task5 and Task6 remain blocked by offline/online mismatch.
- They are not the current bottleneck only because they are already effectively disqualified as near-term submit frontiers.

## 3. Task-Level Bottlenecks

### 3.1 Task2 bottleneck: residual frontier is exhausted faster than it converts

There are now two different Task2 failure modes, and both matter:

1. Broad method changes leak scope.
2. Clean tiny patches no longer reliably buy leaderboard score.

Concrete evidence:

- `v28` broad dependency-aware task2 changed `93` test rows, with `91` outside the intended target family, and returned `71.68` (`-1.37`).
- `v29` shrank to `1` changed row and still returned `73.03` (`-0.02`).
- `v33` changed only `3` task2 rows, all `0 -> 1`, and still returned `73.00` (`-0.05`).

Residual search metrics also show that the older Task2 frontier is basically mined out:

- `verb_rel_or_have`
  - triggered adjustments: `4`
  - net gain on examples: `+1`
  - test changes: `0`
- `verb_has_this`
  - triggered adjustments: `2`
  - net gain on examples: `+2`
  - test changes: `0`
- `verb_has_been_clause`
  - triggered adjustments: `0`
  - test changes: `0`

The only surviving narrow branch that still touched test was the motion/test-touch family:

- `verb_test_touch_motion`
  - triggered adjustments: `25`
  - net gain on examples: `+18`
  - test changes: `3` rows, all `0 -> 1`
- That exact line was promoted as `v33` and failed online.

Current Task2 bottleneck is therefore not "we have no clean micro-patch". It is:

- the remaining clean micro-patches are too weak to transfer,
- while the stronger-looking method changes are too broad and unstable.

### 3.2 Task7 bottleneck: selection/conversion still blocks recall gains

Task7 has the best remaining upside, but its bottleneck is now well localized.

The older rerank diagnosis already showed the core problem:

- In the `v26` diagnosis bundle, `194` task7 test rows changed relative to `v25`.
- Only `7` rows entered the triggered diagnosis bundle.
- `6 / 7` of those were `secondary_visible_not_selected`.
- Pairwise comparison count was only `16`.
- `secondary_strong_win = 1`
- `secondary_split_win = 2`
- average secondary preference rate = `0.2083`

So the main bottleneck is not just "candidate never appears". It is:

- candidate sometimes becomes visible,
- but the judge still does not reliably select it,
- and tightening rerank thresholds does not create a stable takeover path.

The newer direct-fact projection work moved this bottleneck for the first time, but only partially:

- Author bucket baseline judge correctness: `4 / 10`
- `append_unique_projected_r3`: `5 / 10`
- Author bucket baseline judge visibility: `4 / 10`
- `append_unique_projected_r3`: `6 / 10`
- `projected_top_only` author bucket visibility: `7 / 10`
- First confirmed judge gain row:
  - `j.r.r. tolkien`
  - baseline: `john r r tolkien`
  - projected: `j.r.r. tolkien`

But the remaining hard rows show why Task7 is still blocked:

- `art fleming`: visible under direct-fact, still not converted
- `simon & schuster`: visible under direct-fact, still not converted
- `buck`: visible after surname projection, still not selected correctly

Current Task7 bottleneck is therefore:

- candidate generation is no longer the only blocker,
- but conversion quality is still too narrow and too bucket-concentrated,
- with only one confirmed judge gain so far.

### 3.3 Task5 / Task6 bottleneck: calibration failure, not idea shortage

These lines already have enough evidence to say the main issue is trust, not search coverage:

- Task5 guided looked positive offline and still returned `-0.10`.
- Task6 structured hybrid looked non-negative offline and still returned `-0.17`.

So the real bottleneck here is not "find a better prompt" or "try one more conservative patch". It is:

- local evaluation is not calibrated enough to justify another near-term submit.

### 3.4 Task8 bottleneck: amplitude ceiling

Task8 is the opposite of task5/task6:

- it is not obviously dangerous,
- but it is too low-amplitude.

`v30` proved that an audited-clean `2`-row subset can be safe. It also proved that this safety does not currently move the total score.

## 4. Ranked Bottleneck List

### Bottleneck 1: no post-`v27` line has produced real online lift

- Strongest single summary stat: `0 / 4` post-`v27` official attempts improved the score.
- This is the immediate global plateau.

### Bottleneck 2: Task2 transfer amplitude has collapsed

- Task2 still has the best historical record.
- But the current residuals are either:
  - too broad and dangerous, or
  - narrow and clean but too weak to convert.

### Bottleneck 3: Task7 still cannot reliably convert newly visible gold candidates

- This is the most important unsolved mechanism bottleneck.
- It is also the only frontier with both:
  - historical online wins,
  - and new 2026-04-01 evidence of fresh judge-positive movement.

### Bottleneck 4: offline trust for task5/task6 is too weak to justify budget

- These are blocked before method optimization even begins.

### Bottleneck 5: Task8 is operationally safe but strategically capped

- It can preserve score.
- It has not shown evidence that it can raise score.

## 5. Practical Conclusion

The current bottleneck is not simply "we need another tiny patch".

It is a three-part plateau:

1. Task2 has stopped being an easy incremental gain source.
2. Task7 has finally shown a new real signal, but only at the level of partial judge conversion inside a narrow author bucket.
3. Task5/6/8 do not currently offer a better expected value than a carefully gated Task7 continuation.

If the question is "where is the highest-value remaining blockage right now", the answer is:

- globally: post-`v27` no-conversion plateau
- method-wise: Task7 selection/conversion after candidate visibility
- execution-wise: over-trusting clean local diffs on Task2 once amplitude is already exhausted

## 6. Recommended Next Interpretation

Based on the statistics above:

- Do not treat Task2 micro-patches as the default next submit path anymore.
- Keep Task8 as maintenance-only.
- Keep Task5/6 frozen unless calibration quality improves materially.
- Treat Task7 author-bucket direct-fact projection as the only current frontier with both:
  - real historical online plausibility,
  - and newly observed mechanism-level improvement that has not yet been tested in the mainline path.

