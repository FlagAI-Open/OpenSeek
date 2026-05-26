---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-10 00:25:00 CST"
phase: "v37-online-readout"
status: "promoted"
---

# v37 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v37 = 73.7` as the returned official online score on `2026-04-10`.
- Base comparison: `v37` was built on `v36 = 73.6` and gained `0.1` online.
- Previous stable comparison: `v37` beat `v27 = 73.05` by `0.65`.
- Stable official baseline: `v37`.
- Main conclusion: v37 is now the best official package and the current stable official baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v37_task7manualfix4_on_v36_candidate/final_submission_v37_task7manualfix4_on_v36_candidate.zip`
- Manual factual fix count: `4`
- Changed Task7 rows vs `v36`: `4`
- Changed Task7 rows vs `v30`: `189`

## 3. Operational Implications
- The best official baseline is now the post-v36 manual-fix4 line.
- Future Task7 candidates should branch from `v37` and preserve the known target rescues plus all verified manual factual fixes unless a replacement is directly validated.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Branch from v37 and continue the ultra-narrow factual patch line, starting with the newly verified Louisa, turban, and sodden corrections.
