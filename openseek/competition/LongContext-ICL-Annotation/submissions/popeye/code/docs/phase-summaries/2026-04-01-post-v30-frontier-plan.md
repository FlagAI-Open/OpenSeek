---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-01 14:52:15 CST"
phase: "post-v30-frontier-plan"
status: "in-progress"
---

# Post-v30 Frontier Plan

## 1. Current Snapshot
- Scope: Close the `v28` to `v30` submit loop, update the official lineage through the latest neutral result, and reset the next frontier order relative to the current stable base.
- Time window: `2026-03-31` to `2026-04-01`.
- Target outcome for this phase: Turn the current state into one explicit decision packet: what is now closed for promotion, what remains viable as research, and which line should receive the next serious experiment cycle after `v27 = 73.05` held through `v30`.

## 2. Achieved Results
- Completed work:
  - Ran the broad dependency-aware task2 submit candidate `v28` and confirmed it is not promotable.
  - Ran the ultra-narrow dependency-aware recovery candidate `v29` and confirmed it is near-safe but still not better than `v27`.
  - Built and submitted the `v30` candidate that carries the 2-row audited-clean task8 subset from `v14` onto the `v27` base.
  - Updated the official score ledger through `v30` so the current formal version lineage is traceable from one place.
- Key outputs or artifacts:
  - `outputs/work_logs/task2_dependency_v28_failure_audit_2026-04-01.md`
  - `docs/phase-summaries/2026-04-01-task2-dependency-line-closeout.md`
  - `src/build_task8_v30_candidate.py`
  - `outputs/final_submission_v30_task8clean2_on_v27_candidate/merge_summary.json`
  - `outputs/work_logs/task8_v30_candidate_build_2026-04-01.md`
  - `outputs/work_logs/official_score_ledger_2026-04-01.json`
  - `outputs/work_logs/official_score_ledger_2026-04-01.md`
- Evidence (files, commands, metrics):
  - `v28 = 71.68`, which is `-1.37` vs `v27`; the failure audit showed `93` changed task2 test rows, `91` outside the intended target family, and outside-target rate `0.978495` in `outputs/work_logs/task2_dependency_v28_failure_audit_2026-04-01.md`.
  - `v29 = 73.03`, which is `-0.02` vs `v27`; it only changed `1` task2 test row (`rel_be_vbg:plain`) per `outputs/final_submission_v29_task2dependency_relplain_candidate/merge_summary.json`.
  - `v30 = 73.05`, which is `0.00` vs `v27`; it changed only `2` task8 test rows, both `0 blocker / 0 warning` under `task8_candidate_audit.py`, per `outputs/work_logs/task8_v30_candidate_build_2026-04-01.md`.
  - The official lineage now reads: `v25 = 73.03`, `v26 = 72.93`, `v27 = 73.05`, `v28 = 71.68`, `v29 = 73.03`, `v30 = 73.05`, in `outputs/work_logs/official_score_ledger_2026-04-01.md`.

## 3. Open Issues and Risks
- Blocking issues:
  - No line submitted after `v27` has improved on `73.05`.
  - The dependency-aware task2 branch is no longer promotion-ready because broad scope replacement failed catastrophically and the surviving ultra-narrow variant still did not beat base.
  - Task7 still lacks candidate-set visibility on the retained hard rows, so rerank-side tweaks remain structurally blocked.
- Known risks:
  - Any new task2 verb-count change that is not tiny-patch scale is high risk by default after the `v28` scope leak.
  - Task8 maintenance patches can be safe but are now confirmed to be low-amplitude; repeating them is unlikely to move the leaderboard.
  - Task5 and task6 still suffer from offline/online mismatch, so apparently positive local signals remain untrustworthy without a stronger gate.
- Assumptions pending validation:
  - Task7 may still be the best remaining higher-upside frontier if the next work attacks candidate generation / gold visibility rather than rerank thresholds.
  - A stronger calibration gate across task5/task6/task7 could reduce wasted submissions, but that alone may not produce score lift unless tied to a real method improvement.
  - Task2 may still produce another tiny patch in the future, but only if it satisfies the same strict diff-size and zero-regression profile as `v22 / v23 / v27`.

## 4. Candidate Next Directions
1. Direction A:
   - Expected benefit: Reopen `task7` only as a recall-first research line, which is the most plausible remaining path to a larger-than-maintenance gain because `v19` already proved that task7 can transfer online.
   - Estimated effort or cost: Medium.
   - First action: Build a retained-target candidate-visibility bundle around the current frozen hard rows and measure whether new retrieval / candidate-generation logic raises `gold_visible_in_candidates` before any rerank changes are reconsidered.
2. Direction B:
   - Expected benefit: Upgrade offline trust so future task5/task6/task7 experiments are filtered through better calibration rather than optimistic local signals.
   - Estimated effort or cost: Medium.
   - First action: Create a post-`v30` calibration table that explicitly tags lines as `trusted`, `conditional`, `research-only`, or `rejected`, then require new candidate packages to clear line-specific gates before submission.
3. Direction C:
   - Expected benefit: Keep squeezing task2 for one more tiny patch attempt under strict stop rules, preserving the only line with repeated positive transfer.
   - Estimated effort or cost: Low to medium.
   - First action: Reopen task2 residual search only if the search is hard-gated to `<= 3` changed test rows, `0 regression`, and no proxy/helper broadening relative to `v27`.

## 5. Recommended Next Step
- Selected direction: Direction A.
- Why this direction now: `task2` mainline is still the stable formal base, but the last three post-`v27` experiments clarified that current task2 and task8 extensions are either unsafe (`v28`) or too low-amplitude (`v29`, `v30`). Task7 is the only remaining line with both a real historical online win (`v19`) and a still-open mechanism question that is not just another threshold tweak: the candidate set is missing gold on the retained hard rows. That gives it more upside than another maintenance patch while still being better grounded than restarting task5 or task6 blindly.
- Immediate next action (within 24 hours): Start a `task7 recall-first` packet that freezes the current `v27` mainline, extracts the retained hard rows and their failed candidate sets, and adds one new experiment entry focused only on raising `gold_visible_in_candidates` for those rows.
- Success signal: A new task7 diagnosis artifact shows a measurable increase in `gold_visible_in_candidates` on retained targets, and the mechanism for that increase is clearly attributable to candidate generation rather than rerank/order noise.

## 6. Change Log
- [2026-04-01 14:52:15 CST] Initial draft created.
- [2026-04-01 15:25:00 CST] Added the `v28`, `v29`, and `v30` outcomes; recorded updated stop rules; and selected `task7 recall-first` as the recommended next frontier.
