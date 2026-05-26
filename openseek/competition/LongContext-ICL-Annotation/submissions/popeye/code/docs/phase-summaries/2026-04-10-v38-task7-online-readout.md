---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-10 02:10:00 CST"
phase: "v38-online-readout"
status: "promoted"
---

# v38 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v38 = 73.78` as the returned official online score on `2026-04-10`.
- Base comparison: `v38` was built on `v37 = 73.7` and gained `0.08` online.
- Previous stable comparison: `v38` beat `v27 = 73.05` by `0.73`.
- Stable official baseline after later tie handling: `v39`.
- Main conclusion: v38 improved online to 73.78 and is tied for the current best official score.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v38_task7manualfix3_on_v37_candidate/final_submission_v38_task7manualfix3_on_v37_candidate.zip`
- Manual factual fix count: `3`
- Changed Task7 rows vs `v37`: `3`
- Changed Task7 rows vs `v30`: `188`

## 3. Operational Implications
- The Louisa/turban/sodden micro-patch line has now converted online.
- This package remains an official best-score line even though the stable baseline later moved to the tied v39 variant.
- This readout records the official online score only. Leaderboard rank can still move as other teams continue to submit.

## 4. Recommended Next Step
- Compare and consolidate the two 73.78 Task7 lines rather than reopening broad method experiments.
