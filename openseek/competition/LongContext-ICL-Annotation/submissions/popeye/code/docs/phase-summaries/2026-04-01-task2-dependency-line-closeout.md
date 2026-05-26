---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-01 14:24:53 CST"
phase: "task2-dependency-closeout"
status: "completed"
---

# Task2 Dependency Line Closeout

## 1. Current Snapshot
- Scope: Close the current task2 dependency-aware submit line after the second-round policy-table spike, the broad `v28` submit, and the ultra-narrow `v29` submit.
- Time window: 2026-03-31 to 2026-04-01.
- Target outcome for this phase: Decide whether dependency-aware task2 is ready for formal submission, and if not, define explicit stop rules before resources move elsewhere.

## 2. Achieved Results
- Completed work:
  - Implemented a second-round dependency policy-table spike with explicit switches for `compress_be_chain`, `count_trailing_vbg`, `count_to_inf`, and `count_after_discourse_marker`.
  - Built and submitted a broad dependency-aware candidate as `v28`.
  - Audited the `v28` failure and isolated the dominant failure mode as scope leakage outside the intended target family.
  - Built and submitted an ultra-narrow recovery candidate as `v29` that only touched `rel_be_vbg:plain`.
- Key outputs or artifacts:
  - `src/spike_task2_dependency_policy_table.py`
  - `src/task2_dependency_policy.py`
  - `src/audit_task2_dependency_v28_failure.py`
  - `outputs/work_logs/task2_dependency_policy_table_spike_2026-04-01.md`
  - `outputs/work_logs/task2_dependency_v28_failure_audit_2026-04-01.md`
  - `outputs/final_submission_v28_task2dependency_discourse_candidate/merge_summary.json`
  - `outputs/final_submission_v29_task2dependency_relplain_candidate/merge_summary.json`
- Evidence (files, commands, metrics):
  - Baseline remained `v27 = 73.05`, documented in `outputs/final_submission_v27_task2stuffed_on_v25_candidate/merge_summary.json`.
  - The policy-table spike looked positive offline: `compressed_plus_discourse_followup` improved primary focus exact from `0.876984` to `0.916667` and focus exact from `0.832787` to `0.865574`, with `0` defer changes and `0` protected changes in `outputs/work_logs/task2_dependency_policy_table_spike_2026-04-01.md`.
  - The broad submit candidate failed online: `v28 = 71.68`, which is `-1.37` vs `v27`, recorded in `outputs/work_logs/task2_dependency_v28_failure_audit_2026-04-01.md`.
  - `v28` changed `93` task2 test rows; `91` of those changes were outside the intended target family; the largest bucket was `outside_target:bare_finite_dropped = 45`.
  - The ultra-narrow `v29` candidate changed only `1` task2 test row, but still returned `73.03`, which is `-0.02` vs `v27`.

## 3. Open Issues and Risks
- Blocking issues:
  - No dependency-aware candidate has shown online gain over `v27` as of 2026-04-01.
  - The broad policy version cannot be trusted because its real submit-time behavior leaked far outside the intended allowlist.
- Known risks:
  - Offline focus-bundle gains are not sufficient promotion evidence for task2 if submit-time test diffs are not explicitly bounded.
  - Any future dependency-aware candidate that alters global task2 verb counting rather than a strict allowlist is high risk by default.
  - `to_inf_tail` remains unresolved and should be treated as a separate problem, not folded into the next submit candidate.
- Assumptions pending validation:
  - A future dependency-aware candidate may still work if it stays inside a tiny allowlist and preserves the existing global solver elsewhere.
  - That hypothesis is currently unproven and should not be treated as a near-term submit path.

## 4. Candidate Next Directions
1. Direction A:
   - Expected benefit: Freeze the dependency-aware line at research conclusion level and redirect main effort to higher-leverage lines without reopening this branch by accident.
   - Estimated effort or cost: Low.
   - First action: Record a hard stop rule for dependency-aware task2 and update future submit criteria to require `outside_target_family = 0` plus tiny test diff before promotion.
2. Direction B:
   - Expected benefit: Explore one more ultra-narrow dependency candidate under stricter gates to test whether a second tiny-patch-quality row exists beyond `v29`.
   - Estimated effort or cost: Medium.
   - First action: Enumerate only allowlists that keep task2 test diff under `<= 3` rows and reject any candidate with scope leak outside the explicit allowlist.
3. Direction C:
   - Expected benefit: Move primary resources away from task2 dependency-aware work toward a different task or method line with larger remaining upside.
   - Estimated effort or cost: Medium to high, but with better expected leverage than continuing to squeeze task2.
   - First action: Choose the next frontier line and build a fresh evidence packet relative to `v27 = 73.05`.

## 5. Recommended Next Step
- Selected direction: Direction A.
- Why this direction now: The current evidence is decisive enough to stop submit iteration on this branch. `v28` proved that broad dependency-aware replacement is unsafe, and `v29` proved that the surviving ultra-narrow single-row version is not enough to beat `v27`. Continuing to submit more variants now is more likely to consume budget than to produce a better official score.
- Immediate next action (within 24 hours): Treat `v27` as the only formal task2 base, keep dependency-aware work in research-only status, and shift primary experiment bandwidth to a different high-leverage line.
- Success signal: Future planning documents and submit candidates stop treating task2 dependency-aware as an active near-term promotion path unless a new candidate first satisfies the explicit stop-rule gate.

## 6. Change Log
- [2026-04-01 14:24:53 CST] Initial draft created.
- [2026-04-01 14:31:00 CST] Added final closeout after `v28 = 71.68` and `v29 = 73.03`, with submit stop rules and next-step recommendation.
