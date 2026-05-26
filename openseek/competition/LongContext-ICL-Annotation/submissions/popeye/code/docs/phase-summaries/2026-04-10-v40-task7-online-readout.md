---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-10 03:10:00 CST"
phase: "v40-online-readout"
status: "promoted"
---

# v40 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v40 = 73.85` as the returned official online score on `2026-04-10`.
- Base comparison: `v40` was built on `v39 = 73.78` and gained `0.07` online.
- Previous stable comparison: `v40` beat `v27 = 73.05` by `0.8`.
- Stable official baseline: `v40`.
- Main conclusion: v40 improved online to 73.85 and becomes the new best official package and stable baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v40_task7combo6_on_v39_candidate/final_submission_v40_task7combo6_on_v39_candidate.zip`
- Added stack fix count vs `v39`: `3`
- Changed Task7 rows vs `v39`: `3`
- Changed Task7 rows vs `v30`: `187`

## 3. Operational Implications
- The independent Task7 gains do stack online, which is the most important method lesson from this round.
- The default branch point now moves from v39 to v40.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Stay on v40 and test the single-row SGP -> Singapore cleanup as the narrowest next Task7 challenge package.
