---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-10 00:05:00 CST"
phase: "v36-online-readout"
status: "promoted"
---

# v36 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v36 = 73.6` as the returned official online score on `2026-04-09`.
- Base comparison: `v36` was built on `v35 = 73.45` and gained `0.15` online.
- Previous stable comparison: `v36` beat `v27 = 73.05` by `0.55`.
- Stable official baseline: `v36`.
- Main conclusion: v36 is now the best official package and the current stable official baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v36_task7manualfix6_on_v35_candidate/final_submission_v36_task7manualfix6_on_v35_candidate.zip`
- Manual factual fix count: `6`
- Changed Task7 rows vs `v35`: `6`
- Changed Task7 rows vs `v30`: `190`

## 3. Operational Implications
- The best official baseline is now the post-v35 manual-fix line, not only the revert69 cleanup line.
- Future Task7 candidates should branch from `v36` and preserve the known target rescues plus the six clue-level fixes unless a replacement is directly validated.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Branch from v36 and search for another ultra-narrow Task7 factual patch, starting from the unresolved shortlist rather than reopening broad judge changes.
