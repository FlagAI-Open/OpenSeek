---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-11 09:30:00 CST"
phase: "v44-online-readout"
status: "promoted"
---

# v44 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v44 = 74.1` as the returned official online score on `2026-04-11`.
- Base comparison: `v44` was built on `v43 = 74.05` and gained `0.05` online.
- Previous stable comparison: `v44` beat `v27 = 73.05` by `1.05`.
- Stable official baseline: `v44`.
- Main conclusion: v44 improved online to 74.10 and becomes the new best official package and stable baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v44_task7fact3_on_v43_candidate/final_submission_v44_task7fact3_on_v43_candidate.zip`
- Added fact-fix count vs `v43`: `3`
- Changed Task7 rows vs `v43`: `3`
- Changed Task7 rows vs `v30`: `192`

## 3. Operational Implications
- The ultra-narrow short-clue repair line kept transferring online and pushed the official score to 74.10.
- The default branch point now moves from v43 to v44.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Stay on v44 and keep testing only very hard, ultra-narrow Task7 factual repairs rather than reopening broad method changes.
