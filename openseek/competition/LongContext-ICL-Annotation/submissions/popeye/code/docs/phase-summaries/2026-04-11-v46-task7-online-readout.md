---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-11 10:20:00 CST"
phase: "v46-online-readout"
status: "promoted"
---

# v46 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v46 = 74.23` as the returned official online score on `2026-04-11`.
- Base comparison: `v46` was built on `v45 = 74.15` and gained `0.08` online.
- Previous stable comparison: `v46` beat `v27 = 73.05` by `1.18`.
- Stable official baseline: `v46`.
- Main conclusion: v46 improved online to 74.23 and becomes the new best official package and stable baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v46_task7fact3_on_v45_candidate/final_submission_v46_task7fact3_on_v45_candidate.zip`
- Added fact-fix count vs `v45`: `3`
- Changed Task7 rows vs `v45`: `3`
- Changed Task7 rows vs `v30`: `192`

## 3. Operational Implications
- The ultra-narrow clue-level repair line kept transferring online and pushed the official score to 74.23.
- The default branch point now moves from v45 to v46.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Stay on v46 and keep testing only tiny, externally verifiable Task7 factual repairs rather than reopening broad method changes.
