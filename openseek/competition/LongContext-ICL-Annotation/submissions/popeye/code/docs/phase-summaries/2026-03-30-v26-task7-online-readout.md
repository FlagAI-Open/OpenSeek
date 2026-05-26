---
project: "LongContext-ICL-Annotation"
created_at: "2026-03-30 16:47:18 CST"
phase: "v26-online-readout"
status: "diagnosis-first"
---

# v26 task7 online readout

## 1. Current Snapshot
- Scope: Read back the official `v26` online result, keep the formal stable base fixed at `v25 = 73.03`, and convert the next phase from promotion to diagnosis-first task7 analysis.
- Time window: 2026-03-30.
- Target outcome for this phase: Record the online verdict, preserve the last-tested `task7` narrow typed line as a reference, and generate the diagnosis outputs needed before choosing the next mainline.
- Current official state: Treat `v25 = 73.03` as the stable formal base and treat `v26 task7 typed narrow v3 = 72.93` as the returned official online result on `2026-03-30`.
- Main conclusion: `task7 typed narrow v3` is now `offline-promotable but online-unconfirmed`; it cannot be treated as a production-ready gain line.

## 2. Achieved Results
- Completed work: Preserved the `v26` candidate composition and offline rationale in [`outputs/final_submission_v26_task7typednarrowv3_candidate/merge_summary.json`](../../outputs/final_submission_v26_task7typednarrowv3_candidate/merge_summary.json).
- Completed work: Preserved the narrow typed gate evidence in [`outputs/holdout_eval/task7_typed_gate_summary_v3narrow.json`](../../outputs/holdout_eval/task7_typed_gate_summary_v3narrow.json).
- Completed work: Added diagnosis-oriented reporting support to [`src/validate_task7_candidate_rerank.py`](../../src/validate_task7_candidate_rerank.py) so triggered-row bundles can include candidate order, gold-presence buckets, normalized secondary-only candidates, and selection-risk markers.
- Completed work: Added new diagnosis scripts for triggered-row bundling, `v25` vs `v26` mismatch auditing, order-bias probing, and pairwise probing in [`src/build_task7_v26_diagnostics.py`](../../src/build_task7_v26_diagnostics.py), [`src/probe_task7_order_bias.py`](../../src/probe_task7_order_bias.py), and [`src/probe_task7_pairwise.py`](../../src/probe_task7_pairwise.py).
- Completed work: Generated the first diagnosis bundle outputs at [`outputs/work_logs/task7_v26_triggered_diagnostics_2026-03-30.json`](../../outputs/work_logs/task7_v26_triggered_diagnostics_2026-03-30.json), [`outputs/work_logs/task7_v26_online_mismatch_diagnosis_2026-03-30.json`](../../outputs/work_logs/task7_v26_online_mismatch_diagnosis_2026-03-30.json), [`outputs/work_logs/task7_v26_order_bias_probe_2026-03-30.json`](../../outputs/work_logs/task7_v26_order_bias_probe_2026-03-30.json), and [`outputs/work_logs/task7_v26_pairwise_probe_2026-03-30.json`](../../outputs/work_logs/task7_v26_pairwise_probe_2026-03-30.json).
- Completed work: Implemented and smoke-tested the exploratory `pairwise_triggered_v1` rerank path in [`src/validate_task7_candidate_rerank.py`](../../src/validate_task7_candidate_rerank.py), then ran 3-seed holdout reports at [`outputs/work_logs/task7_rerank_seed43_n40_typed_gated_append_unique_v3narrow_pairwise_triggered.json`](../../outputs/work_logs/task7_rerank_seed43_n40_typed_gated_append_unique_v3narrow_pairwise_triggered.json), [`outputs/work_logs/task7_rerank_seed123_n40_typed_gated_append_unique_v3narrow_pairwise_triggered.json`](../../outputs/work_logs/task7_rerank_seed123_n40_typed_gated_append_unique_v3narrow_pairwise_triggered.json), and [`outputs/work_logs/task7_rerank_seed2026_n40_typed_gated_append_unique_v3narrow_pairwise_triggered.json`](../../outputs/work_logs/task7_rerank_seed2026_n40_typed_gated_append_unique_v3narrow_pairwise_triggered.json).
- Completed work: Tightened the pairwise-triggered takeover rule to a stricter `pairwise_triggered_v2` path in [`src/validate_task7_candidate_rerank.py`](../../src/validate_task7_candidate_rerank.py), requiring `2/2` pairwise wins before a secondary candidate can replace the primary choice, then ran matched-config 3-seed holdout reports at [`outputs/work_logs/task7_rerank_seed43_n40_typed_gated_append_unique_v3narrow_pairwise_triggered_v2_matched.json`](../../outputs/work_logs/task7_rerank_seed43_n40_typed_gated_append_unique_v3narrow_pairwise_triggered_v2_matched.json), [`outputs/work_logs/task7_rerank_seed123_n40_typed_gated_append_unique_v3narrow_pairwise_triggered_v2_matched.json`](../../outputs/work_logs/task7_rerank_seed123_n40_typed_gated_append_unique_v3narrow_pairwise_triggered_v2_matched.json), and [`outputs/work_logs/task7_rerank_seed2026_n40_typed_gated_append_unique_v3narrow_pairwise_triggered_v2_matched.json`](../../outputs/work_logs/task7_rerank_seed2026_n40_typed_gated_append_unique_v3narrow_pairwise_triggered_v2_matched.json).
- Key outputs or artifacts: [`outputs/holdout_eval/task7_typed_gate_summary_v3narrow.json`](../../outputs/holdout_eval/task7_typed_gate_summary_v3narrow.json), [`outputs/final_submission_v26_task7typednarrowv3_candidate/merge_summary.json`](../../outputs/final_submission_v26_task7typednarrowv3_candidate/merge_summary.json), [`outputs/smoke_task7_typed_narrow_v3/openseek-7-v1.jsonl`](../../outputs/smoke_task7_typed_narrow_v3/openseek-7-v1.jsonl), [`outputs/work_logs/task7_v26_triggered_diagnostics_2026-03-30.json`](../../outputs/work_logs/task7_v26_triggered_diagnostics_2026-03-30.json), [`outputs/work_logs/task7_v26_online_mismatch_diagnosis_2026-03-30.json`](../../outputs/work_logs/task7_v26_online_mismatch_diagnosis_2026-03-30.json), [`outputs/work_logs/task7_v26_order_bias_probe_2026-03-30.json`](../../outputs/work_logs/task7_v26_order_bias_probe_2026-03-30.json), [`outputs/work_logs/task7_v26_pairwise_probe_2026-03-30.json`](../../outputs/work_logs/task7_v26_pairwise_probe_2026-03-30.json), [`outputs/work_logs/task7_pairwise_triggered_smoke_2026-03-30.json`](../../outputs/work_logs/task7_pairwise_triggered_smoke_2026-03-30.json), the 3-seed `pairwise_triggered_v1` holdout reports, and the matched 3-seed `pairwise_triggered_v2` holdout reports.
- Evidence (files, commands, metrics): `typed_gated_append_unique_v3narrow` passed its offline gate with `avg judge delta +0.0167`, `avg oracle delta +0.0250`, and `judge_nonnegative_seed_count = 2` across seeds `43 / 123 / 2026` in [`task7_typed_gate_summary_v3narrow.json`](../../outputs/holdout_eval/task7_typed_gate_summary_v3narrow.json).
- Evidence (files, commands, metrics): The same gate summary shows the gain remained narrow: average secondary trigger count `2.33`, max trigger count `4`, and `judge_selected_from_secondary_count = 0` on every seed.
- Evidence (files, commands, metrics): [`merge_summary.json`](../../outputs/final_submission_v26_task7typednarrowv3_candidate/merge_summary.json) records that `v26` replaced only `task7` on top of `v25`, used `append_unique` with `2` reserved secondary judge slots, restricted typed routing to `person_entity,title_or_place`, and changed `194` task7 test rows relative to `v25`.
- Evidence (files, commands, metrics): The first mismatch/probe bundle accounts for all `194` changed task7 rows, shows `7` triggered diagnosis rows, `6/7` rows marked `secondary_visible_not_selected`, order disagreement on `4/7` rows, and only limited pairwise secondary conversion (`1` strong win and `2` split outcomes across `16` comparisons).
- Evidence (files, commands, metrics): The exploratory `pairwise_triggered_v1` 3-seed holdout rerun produced only a narrow uplift versus the diagnosis baseline: average `judge_accuracy` improved from `0.2917` to `0.3083` (`+0.0167`) and average `oracle_hit_rate` improved from `0.3333` to `0.3417` (`+0.0083`), while average trigger count stayed fixed at `2.33`.
- Evidence (files, commands, metrics): `pairwise_triggered_v1` did increase average `judge_selected_from_secondary_count` from `0.00` to `1.00` across seeds, but the triggered-row conversion remained weak: seed `43` produced `0/3` secondary takeovers, seed `123` produced `2/2`, and seed `2026` produced `1/2`, with no positive average `judge_gain_on_triggered` or `oracle_gain_on_triggered` in the rerun reports.
- Evidence (files, commands, metrics): The matched-config `pairwise_triggered_v2` rerun removed those weak takeovers but also removed the only directional gain signal: average `judge_accuracy` fell to `0.2833` and average `oracle_hit_rate` fell to `0.3250`, both below the diagnosis baseline, while average `judge_selected_from_secondary_count` returned from `1.00` to `0.00`.
- Evidence (files, commands, metrics): By seed, `pairwise_triggered_v2` only matched `v1` on seed `43` (`0.275 / 0.325`), then regressed on seed `123` (`0.25 / 0.30`) and seed `2026` (`0.325 / 0.35`), with no positive average triggered-gain signal.
- Evidence (files, commands, metrics): The stricter `2/2` takeover rule therefore did not produce a more robust line; it simply suppressed the fragile secondary takeovers that `v1` was already using, leaving the whole pairwise-triggered path exploratory and non-promotable.

## 3. Open Issues and Risks
- Blocking issues: No new official submission should be generated before the diagnosis phase finishes and one next line clears a stricter gate than the previous offline-only standard.
- Known risks: The current offline gate was enough to promote `v26`, but it did not explain online conversion risk; the returned official score `72.93` is below the stable base `73.03`.
- Known risks: The narrow typed line appears amplitude-limited. The gate summary shows only a small number of triggered rows and zero observed judge selection from secondary candidates despite secondary visibility.
- Known risks: `pairwise_triggered_v1` improves secondary selection frequency in the rerun reports, but the improvement is still too small and too fragile to justify promotion because the average accuracy uplift remains narrow and triggered-row correctness conversion is still weak.
- Known risks: The stricter matched-config `pairwise_triggered_v2` rerun removes those fragile secondary takeovers without producing a stronger replacement signal, so the whole pairwise-triggered line currently fails both the gain test and the robustness test.
- Known risks: The fresh diagnosis-mode reruns are stochastic network evaluations, so their triggered-row identities are useful for bottleneck analysis but should not be treated as bit-for-bit reproductions of the archived `holdout_eval` artifacts.
- Known risks: If most triggered gains are secondary-visible but judge-unselected, the bottleneck is likely selection/conversion rather than candidate generation.
- Assumptions pending validation: The `72.93` readout from `2026-03-30` is the official online verdict for `v26`.
- Assumptions pending validation: `v25 = 73.03` remains the stable formal base until a later official package beats it.
- Assumptions pending validation: The next task7 decision should depend on measured diagnosis outputs rather than additional offline promotion passes.

## 4. Mismatch Diagnosis Checklist
- Phase A: Build a `v25` vs `v26` changed-row audit and account for all `194` changed `task7` test rows.
- Phase A: Build a triggered-row diagnosis bundle with candidate order, `secondary_only_candidates`, judge-visible set, gold-presence bucket, and selection-risk markers.
- Phase B: Run an order-bias probe on the triggered rows only, using `2-3` fixed permutations on seeds `43 / 123 / 2026`.
- Phase C: Run pairwise `A/B` and `B/A` probes for primary-top vs each secondary-only unique candidate on triggered rows only.
- Phase D: Use those results to choose exactly one next mainline direction: `task7 judge amplification`, `task2 low-risk patching`, or a dual-track setup with one exploratory line plus one low-risk line.

## 5. Candidate Next Directions
1. Direction A:
   - Expected benefit: Highest information gain for understanding why the narrow typed package failed online despite clearing the offline gate.
   - Estimated effort or cost: Medium analysis cost, low packaging risk, no official submission required.
   - First action: Finish the triggered-row bundle plus the `v25` vs `v26` mismatch report, then inspect whether the main failure pattern is `secondary visible but not selected`.
2. Direction B:
   - Expected benefit: If diagnosis shows judge order sensitivity or systematic non-selection of secondary-only uniques, a judge-amplification or pairwise rerank line may become justified.
   - Estimated effort or cost: Medium to high analysis cost, still exploratory until robustness evidence exists.
   - First action: Run the order-bias and pairwise probes on the current narrow typed line instead of introducing a new candidate branch.
   - Latest readout: `pairwise_triggered_v1` was directionally positive but still below promotion standard, and the stricter matched-config `pairwise_triggered_v2` rerun fell below both baseline and `v1`, so this branch should now stay as background research rather than the next mainline candidate.
3. Direction C:
   - Expected benefit: If diagnosis shows task7 amplitude is too small or too fragile, a task2 low-risk patch can reclaim the next official attempt with a better transfer profile.
   - Estimated effort or cost: Medium offline search cost, lower transfer uncertainty than another unguided task7 promotion.
   - First action: Hold task7 as background research and move the next official package to a higher-confidence low-risk line because the stricter pairwise-triggered follow-up failed to produce either gain or robustness.

## 6. Recommended Next Step
- Selected direction: Direction C for the next mainline, while keeping the completed task7 diagnosis bundle as background evidence.
- Why this direction now: The online readout was already a stop signal, and the matched `pairwise_triggered_v2` follow-up confirmed that tightening the takeover rule does not rescue the line. The task7 pairwise path now fails both the gain test and the robustness test, so it should not consume the next official attempt.
- Immediate next action (within 24 hours): Write back the final task7 pairwise readout, freeze `pairwise_triggered_v1` and `pairwise_triggered_v2` as exploratory artifacts only, and switch the next mainline search to a higher-confidence low-risk line.
- Success signal: The next official candidate should come from a lower-risk branch with a clearer transfer profile than the current task7 pairwise line; task7 should only be revisited if later diagnosis reveals a materially stronger robustness mechanism than simple triggered takeover thresholding.
- Current diagnosis signal: The first mismatch/probe bundle accounts for all `194` changed task7 rows, shows `7` triggered diagnosis rows, `6/7` rows marked `secondary_visible_not_selected`, order disagreement on `4/7` rows, and only limited pairwise secondary conversion (`1` strong win and `2` split outcomes across `16` comparisons).
- Current experiment readout: `pairwise_triggered_v1` was slightly better than the diagnosis baseline on average but still not promotable, and the matched `pairwise_triggered_v2` rerun was worse than both baseline and `v1`: average `judge_accuracy` fell to `0.2833`, average `oracle_hit_rate` fell to `0.3250`, and average `judge_selected_from_secondary_count` returned to `0.00`.
- Promotion policy for the next package: Do not promote another task7 package unless it shows positive average `judge_accuracy`, positive average `oracle_hit_rate`, and one new robustness signal from either low permutation disagreement or positive pairwise conversion on secondary-only unique candidates. The current pairwise-triggered branch does not satisfy that gate.

## 7. Facts vs Assumptions
- Facts from repo artifacts:
  - `v26` is a `task7`-only replacement on top of `v25` and changed `194` task7 test rows according to [`merge_summary.json`](../../outputs/final_submission_v26_task7typednarrowv3_candidate/merge_summary.json).
  - `typed_gated_append_unique_v3narrow` passed the offline gate and the smoke checks recorded in [`task7_typed_gate_summary_v3narrow.json`](../../outputs/holdout_eval/task7_typed_gate_summary_v3narrow.json).
  - Existing offline evidence already hints at a conversion bottleneck because triggered rows often make secondary candidates judge-visible without ever being selected.
- Assumptions carried into this readout:
  - The official online return on `2026-03-30` is `v26 = 72.93`.
  - The stable formal base remains `v25 = 73.03`.
- Recommended actions derived from those facts and assumptions:
  - Pause promotion.
  - Diagnose first.
  - Keep `pairwise_triggered_v1` exploratory only.
  - Keep `pairwise_triggered_v2` exploratory only.
  - Switch the next mainline attempt away from task7 pairwise-triggered and toward a higher-confidence low-risk line.

## 8. Change Log
- [2026-03-30 16:47:18 CST] Initial draft created.
- [2026-03-30] Expanded the readout into a diagnosis-first summary, cited the exact `v26` artifacts, and recorded the stricter post-readout promotion policy.
- [2026-03-30] Added the exploratory `pairwise_triggered_v1` rerun readout: slight average uplift, better secondary selection frequency, but still no robust triggered-gain signal and therefore not promotable.
- [2026-03-30] Added the matched-config `pairwise_triggered_v2` verdict: stricter `2/2` takeovers removed the weak secondary flips but also removed the only directional gain, so the whole task7 pairwise-triggered branch remains exploratory and the next mainline should move to a lower-risk line.
