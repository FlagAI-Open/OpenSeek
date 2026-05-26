---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-06 17:02:46 CST"
phase: "task2-motion-exact-stop"
status: "in-progress"
---

# Task2 Motion Exact Stop

## 1. Current Snapshot
- Scope: Re-open the residual Task2 verb-side frontier after the earlier dependency closeout, but only for ultra-narrow `verb_rel_or_have`-adjacent probes that could plausibly beat the current submit line without broad risk.
- Time window: 2026-04-06.
- Target outcome for this phase: Decide whether the newly isolated `motion_only exact-surface` Task2 patch is strong enough to justify a live submit on top of the current stable package, or whether Task2 should be frozen again.

## 2. Achieved Results
- Completed work:
  - Re-audited the surviving Task2 micro-variants against the real `v27`/mainline behavior with `nltk` restored locally, instead of relying only on older probe outputs.
  - Confirmed that `verb_rel_or_have` itself is effectively exhausted on the current mainline: it still changes examples a little, but does not move test rows.
  - Isolated a narrower `motion_only exact-surface` patch from the older `verb_test_touch_*` family and wired it into `src/method.py` behind a new default-off switch: `OPENSEEK_TASK2_VERB_TEST_TOUCH_MOTION_EXACT`.
  - Built a Task2-only candidate on top of `v31` as `v33`, then submitted it for live evaluation.
- Key outputs or artifacts:
  - `src/method.py`
  - `src/build_task2_motion_exact_v33_candidate.py`
  - `outputs/work_logs/task2_rule_probe_2026-04-06_rel_neighbor_vs_v27.json`
  - `outputs/work_logs/task2_probe_delta_audit_2026-04-06_verb_test_touch_motion_vs_v27.json`
  - `outputs/work_logs/task2_motion_exact_test_inventory_2026-04-06.json`
  - `outputs/task2_motion_exact_candidate/task2_change_summary.json`
  - `outputs/final_submission_v33_task2motionexact_on_v31_candidate/merge_summary.json`
- Evidence (files, commands, metrics):
  - In `task2_rule_probe_2026-04-06_rel_neighbor_vs_v27.json`, `verb_rel_or_have` only produced `4` triggered adjustments, `+0.000184` example delta, and `0` test changes relative to the current mainline-equivalent baseline.
  - The isolated `motion_only exact-surface` logic was verified to touch only `3` Task2 test rows in `task2_motion_exact_test_inventory_2026-04-06.json`:
    - `rumbles`: `0 -> 1`
    - `maneuvers`: `0 -> 1`
    - `skateboards`: `0 -> 1`
  - `outputs/final_submission_v33_task2motionexact_on_v31_candidate/merge_summary.json` confirms `v33` replaced only Task2 on top of `v31` and changed exactly `3` Task2 rows.
  - Live evaluation result on 2026-04-06: `v33 = 73.00`, which is `-0.05` vs the current working base `73.05`.

## 3. Open Issues and Risks
- Blocking issues:
  - The only newly isolated few-row Task2 candidate tested live in this phase failed to improve the official score.
  - There is no remaining Task2 verb-side micro-patch in the audited set with both a plausible online path and stronger evidence than the rejected `v33` candidate.
- Known risks:
  - Further Task2 submit iterations are likely to consume budget on marginal or test-only effects rather than produce a meaningful score gain.
  - The `verb_test_touch_*` family can look attractive when split narrowly, but the online result shows that even a clean `3`-row diff is not enough by itself.
  - Re-expanding toward `combo_v2`, helper broadening, or dependency-style allowlists would raise risk again without any new evidence that Task2 still has upside.
- Assumptions pending validation:
  - Assumption: the best remaining leverage is no longer in Task2 micro-patches.
  - Assumption: future Task2 work should require stronger evidence than “few-row clean diff” before earning another submit slot.

## 4. Candidate Next Directions
1. Direction A:
   - Expected benefit: Freeze the Task2 `motion_exact` line immediately and avoid spending more submit budget on a branch that already failed under live evaluation.
   - Estimated effort or cost: Low.
   - First action: Record `v33 = 73.00` as a negative result, keep the new switch default-off, and treat `v31` as the current stable package again.
2. Direction B:
   - Expected benefit: Continue mining Task2 for another ultra-narrow patch, such as `play_only` or a different exact-surface subset.
   - Estimated effort or cost: Medium.
   - First action: Require a new candidate to show both a tiny diff and a materially stronger offline or matched-row signal than `v33` before any new submit.
3. Direction C:
   - Expected benefit: Shift primary effort back to higher-upside lines such as narrow Task7 experiments, where there is already evidence of real judge gain instead of just local cleanliness.
   - Estimated effort or cost: Medium.
   - First action: Resume the Task7 candidate path from the current stable base and compare any new package against `v31` rather than reopening Task2.

## 5. Recommended Next Step
- Selected direction: Direction A.
- Why this direction now:
  - The key decision was whether a very small, low-risk Task2 patch could still buy score. We tested that directly, and the answer was no.
  - `v33` had the exact shape we wanted operationally: narrow scope, default-off implementation, and a verified `3`-row test diff. Even that was insufficient online.
  - Once that candidate fails, the burden of proof for more Task2 submit iteration rises sharply.
- Immediate next action (within 24 hours):
  - Freeze `OPENSEEK_TASK2_VERB_TEST_TOUCH_MOTION_EXACT` as research-only, do not promote it into default behavior, and move active submit attention away from Task2.
- Success signal:
  - Future package decisions stop treating Task2 exact-surface micro-patches as an active submit frontier unless a materially stronger evidence packet appears first.

## 6. Change Log
- [2026-04-06 17:02:46 CST] Initial draft created.
- [2026-04-06 17:12:00 CST] Added `v33 = 73.00` live result, recorded the `3`-row Task2 motion patch evidence, and froze the line as non-promotable.
