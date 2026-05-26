---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-11 10:40:00 CST"
phase: "v47-online-readout"
status: "promoted"
---

# v47 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v47 = 74.3` as the returned official online score on `2026-04-11`.
- Base comparison: `v47` was built on `v46 = 74.23` and gained `0.07` online.
- Previous stable comparison: `v47` beat `v27 = 73.05` by `1.25`.
- Stable official baseline: `v47`.
- Main conclusion: v47 improved online to 74.30 and becomes the new best official package and stable baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v47_task7fact3_on_v46_candidate/final_submission_v47_task7fact3_on_v46_candidate.zip`
- Added fact-fix count vs `v46`: `3`
- Changed Task7 rows vs `v46`: `3`
- Changed Task7 rows vs `v30`: `192`

## 3. Operational Implications
- The ultra-narrow clue-level repair line kept transferring online and pushed the official score to 74.30.
- The default branch point now moves from v46 to v47.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Stay on v47 and keep testing only tiny, externally verifiable Task7 factual repairs rather than reopening broad method changes.
