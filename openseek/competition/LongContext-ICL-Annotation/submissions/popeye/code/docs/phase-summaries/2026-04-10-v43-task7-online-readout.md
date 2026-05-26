---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-10 11:10:00 CST"
phase: "v43-online-readout"
status: "promoted"
---

# v43 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v43 = 74.05` as the returned official online score on `2026-04-10`.
- Base comparison: `v43` was built on `v42 = 73.98` and gained `0.07` online.
- Previous stable comparison: `v43` beat `v27 = 73.05` by `1.0`.
- Stable official baseline: `v43`.
- Main conclusion: v43 improved online to 74.05 and becomes the new best official package and stable baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v43_task7fact3_on_v42_candidate/final_submission_v43_task7fact3_on_v42_candidate.zip`
- Added fact-fix count vs `v42`: `3`
- Changed Task7 rows vs `v42`: `3`
- Changed Task7 rows vs `v30`: `192`

## 3. Operational Implications
- The ultra-narrow Task7 factual patch line delivered another online gain and crossed 74.0.
- The default branch point now moves from v42 to v43.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Keep v43 as the new stable Task7 base and only consider another package if there is a comparably hard, ultra-narrow factual patch set.
