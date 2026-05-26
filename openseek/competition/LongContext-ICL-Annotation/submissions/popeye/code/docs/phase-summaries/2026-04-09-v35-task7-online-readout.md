---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-09 23:40:00 CST"
phase: "v35-online-readout"
status: "promoted"
---

# v35 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v35 = 73.45` as the returned official online score on `2026-04-09`.
- Base comparison: `v35` was built on `v34 = 73.18` and gained `0.27` online.
- Previous stable comparison: `v35` beat `v27 = 73.05` by `0.4`.
- Stable official baseline: `v35`.
- Main conclusion: v35 is now the best official package and the current stable official baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v35_task7authorproj_revert69_on_v34_candidate/final_submission_v35_task7authorproj_revert69_on_v34_candidate.zip`
- Pairwise-audit revert count: `69`
- Changed Task7 rows vs `v34`: `69`
- Changed Task7 rows vs `v30`: `190`

## 3. Operational Implications
- The best official baseline is now the post-v34 cleanup line, not the raw stability-gate package.
- Future Task7 candidates should branch from `v35` and preserve the known author-projection rescues unless a replacement is directly validated.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Branch from v35 and hunt for an ultra-narrow Task7 patch that adds concrete factual fixes instead of broad semantic churn.
