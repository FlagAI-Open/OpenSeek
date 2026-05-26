---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-01 19:53:03 CST"
phase: "task7-direct-fact-projection"
status: "in-progress"
---

# Task7 Direct-Fact Projection Frontier

## 1. Current Snapshot
- Scope: Task7 post-freeze frontier only, focused on `direct-fact candidate expansion` and a new `author-answer catalog projection` layer for `quoted_work_author_relation`.
- Time window: 2026-04-01.
- Target outcome for this phase: Decide whether `direct-fact` has advanced from research-only visibility gains to a narrow, defensible experiment that can be wired behind a switch in the main Task7 path.

## 2. Achieved Results
- Completed work:
  - Confirmed that prompt-only `slot expansion` is exhausted and the active frontier is externalized candidate generation, not rerank threshold tuning.
  - Built a reusable direct-fact probe/audit/validation scaffold for three buckets: `legacy_show_host`, `explicit_company_publisher_book`, and narrowed `quoted_work_author_relation`.
  - Added a new author-only projection helper that maps direct-fact candidates onto known example answer surfaces using narrow rules:
    - initials-signature projection, e.g. `John Ronald Reuel Tolkien -> j.r.r. tolkien`
    - surname-only projection when the author catalog contains the surname form, e.g. `Pearl S. Buck -> buck`
  - Extended the judge validation script to compare multiple variants:
    - `append_unique_raw_r2`
    - `append_unique_projected_r2`
    - `append_unique_raw_r3`
    - `append_unique_projected_r3`
    - `projected_top_only`
- Key outputs or artifacts:
  - Baseline direct-fact research:
    - `outputs/work_logs/task7_direct_fact_probe_2026-04-01_r3.json`
    - `outputs/work_logs/task7_direct_fact_bucket_audit_2026-04-01.json`
    - `outputs/work_logs/task7_direct_fact_judge_validation_2026-04-01.json`
  - Projection follow-up:
    - `src/task7_direct_fact_projection.py`
    - `src/audit_task7_direct_fact_buckets.py`
    - `src/validate_task7_direct_fact_judge.py`
    - `outputs/work_logs/task7_direct_fact_bucket_audit_2026-04-01_projected.json`
    - `outputs/work_logs/task7_direct_fact_bucket_audit_2026-04-01_projected.md`
    - `outputs/work_logs/task7_direct_fact_judge_validation_2026-04-01_projected.json`
    - `outputs/work_logs/task7_direct_fact_judge_validation_2026-04-01_projected.md`
- Evidence (files, commands, metrics):
  - Author-bucket visibility improved from `4/10` baseline to `7/10` with projected direct-fact in `task7_direct_fact_bucket_audit_2026-04-01_projected.json`.
  - `append_unique_projected_r3` improved author-bucket judge correctness from `4/10` to `5/10`, with visibility from `4/10` to `6/10`, in `task7_direct_fact_judge_validation_2026-04-01_projected.json`.
  - `projected_top_only` also reached `5/10` judge correctness and `7/10` visibility on the same matched example set.
  - The first real judge gain is now confirmed on `openseek-7-9c6cc7af229c4ff1a5320a5702a2f52a`:
    - gold: `j.r.r. tolkien`
    - baseline prediction: `john r r tolkien`
    - projected variant prediction: `j.r.r. tolkien`
  - No matched-example regressions were observed for `append_unique_projected_r3` or `projected_top_only` relative to baseline in the current validation run.

### 2026-04-08 Update
- Fresh current-mainline Task7 full runs with fixed seed (`off` vs `on`) changed `36` / `500` rows, but only `2` of those changes were inside the intended `quoted_work_author_relation` bucket; `34` were non-target rows. See:
  - `outputs/work_logs/task7_author_projection_refresh_seed2026_diff_audit_2026-04-08.json`
  - `outputs/work_logs/task7_author_projection_refresh_seed2026_diff_audit_2026-04-08.md`
- Isolation replay of those `36` changed rows did **not** support using the full-run diff as the feature delta:
  - only `6` rows still changed in isolation,
  - all `6` were non-target rows,
  - the `2` target-bucket rows from the full-run diff did not reproduce as stable isolated deltas.
  - See `outputs/work_logs/task7_author_projection_refresh_seed2026_isolation_audit_2026-04-08.json`.
- The reliable signal now comes from per-row author-bucket traces rather than from the raw full-run diff:
  - `openseek-7-3deb283e13ec4b5393b1f1d61b426a4e`: stable judge correction `jerome k. jellicoe -> Joseph Conrad`
  - `openseek-7-88c5e426f6b84e959de64d77e0862297`: stable judge correction `john hersey -> John Dos Passos`
  - `openseek-7-02e748b0da7f4492864df5ab8b804556`: stable author projection trigger but no prediction change (`jack london` remains correct)
  - See `outputs/work_logs/task7_author_projection_refresh_seed2026_author_row_trace_2026-04-08.json`.
- Working interpretation after the 2026-04-08 refresh:
  - author projection still has real, narrow upside inside the author bucket,
  - but current mainline `off/on` full-run diffs are polluted by request-order / sampling effects,
  - so the frontier remains valid but is **not package-ready** based on the raw full-run diff.

## 3. Open Issues and Risks
- Blocking issues:
  - The gain is currently concentrated in one confirmed row (`j.r.r. tolkien`); there is not yet evidence of multiple independent judge gains.
  - Focus-bundle winners `art fleming` and `simon & schuster` still do not convert under the current judge-facing merge variants.
  - `buck` becomes visible after projection, but the judge still selects the wrong answer.
- Known risks:
  - `projected_top_only` is more aggressive than `append_unique_projected_r3`; it may suppress good baseline candidates on unseen author rows even though it showed no regression on the matched set.
  - Surname-only projection is intentionally narrow, but it still needs real task-level smoke validation before any submit consideration.
  - The current evidence is from matched example rows and targeted test preview rows, not a full offline-to-online calibrated evaluation.
- Assumptions pending validation:
  - Assumption: author-bucket example answers are a safe projection catalog for canonical Jeopardy surface forms.
  - Assumption: an author-only switch in the mainline Task7 rerank path will not damage non-author buckets because the trigger is bucket-scoped.
  - Assumption: `append_unique_projected_r3` is the safer first integration point than `projected_top_only`.

## 4. Candidate Next Directions
1. Direction A:
   - Expected benefit: Wire `append_unique_projected_r3` into `method.py` as an author-bucket-only experiment switch, preserving baseline behavior elsewhere while testing the first judge-positive variant in the real Task7 path.
   - Estimated effort or cost: Medium. Requires mainline integration plus task7-only smoke and artifact review.
   - First action: Add a disabled-by-default env flag for author-only direct-fact projection, then run a small Task7 smoke output and inspect affected rows.
2. Direction B:
   - Expected benefit: Push the more aggressive `projected_top_only` variant, which currently has the best author-bucket visibility (`7/10`) and equal correctness (`5/10`) on matched examples.
   - Estimated effort or cost: Medium-high. Higher upside, but also higher risk of unseen regressions because it replaces rather than supplements baseline judge candidates.
   - First action: Integrate `projected_top_only` behind a separate flag and compare it head-to-head with `append_unique_projected_r3` on a task7-only smoke run.
3. Direction C:
   - Expected benefit: Stop at research conclusion level and avoid integrating anything until more rows are converted offline, especially `buck`, `art fleming`, and `simon & schuster`.
   - Estimated effort or cost: Low immediate effort, but slows progress toward a package-worthy candidate.
   - First action: Record the current result as “first judge-positive signal” and pause mainline changes.

## 5. Recommended Next Step
- Selected direction: Direction A.
- Why this direction now:
  - It is the narrowest change that is now justified by evidence.
  - It keeps the gain-producing mechanism (`projection`) but avoids the higher replacement risk of `projected_top_only`.
  - It matches the gate we set before this phase: only wire it in once projected direct-fact demonstrates actual judge gains on matched rows.
- Immediate next action (within 24 hours):
  - Implement an author-bucket-only experiment switch in `method.py` for `append_unique_projected_r3`, keep the default path unchanged, and produce a dedicated Task7 smoke artifact with exact environment settings recorded.
- Success signal:
  - The smoke run shows author-bucket improvements or stable behavior on the known positive rows without harming non-author buckets, and the resulting package is credible enough to evaluate as a candidate rather than as research-only scaffolding.

## 6. Change Log
- [2026-04-01 19:53:03 CST] Initial draft created.
- [2026-04-01 19:57:00 CST] Added projection-phase summary, evidence files, risk assessment, and recommended next step.
