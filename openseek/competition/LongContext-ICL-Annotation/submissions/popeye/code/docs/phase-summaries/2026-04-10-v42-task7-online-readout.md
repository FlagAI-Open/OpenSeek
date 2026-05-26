---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-10 10:45:00 CST"
phase: "v42-online-readout"
status: "promoted"
---

# v42 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v42 = 73.98` as the returned official online score on `2026-04-10`.
- Base comparison: `v42` was built on `v40 = 73.85` and gained `0.13` online.
- Previous stable comparison: `v42` beat `v27 = 73.05` by `0.93`.
- Stable official baseline: `v42`.
- Main conclusion: v42 improved online to 73.98 and becomes the new best official package and stable baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v42_task7fact4_on_v41_candidate/final_submission_v42_task7fact4_on_v41_candidate.zip`
- Added fact-fix count vs `v41` package line: `4`
- Changed Task7 rows vs `v41`: `4`
- Changed Task7 rows vs `v30`: `191`

## 3. Operational Implications
- The ultra-narrow factual repair line is still transferring online even after several consecutive promotions.
- The default branch point now moves from v40 to v42.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Stay on v42 and test one last ultra-narrow factual patch package rather than reopening broad Task7 method experiments.
