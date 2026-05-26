---
project: "LongContext-ICL-Annotation"
created_at: "2026-04-09 20:50:00 CST"
phase: "v34-online-readout"
status: "promoted"
---

# v34 task7 online readout

## 1. Current Snapshot
- Current official result: Treat `v34 = 73.18` as the returned official online score on `2026-04-09`.
- Base comparison: `v34` was built on `v30 = 73.05` and gained `0.13` online.
- Previous best comparison: `v34` beat `v27 = 73.05` by `0.13`.
- Stable official baseline: `v34`.
- Main conclusion: v34 is now the best official package and the current stable official baseline.

## 2. Promotion Evidence
- Package zip: `/mnt/nvme_data/qhuser/PJC/flagos/OpenSeek/openseek/competition/LongContext-ICL-Annotation/outputs/final_submission_v34_task7authorproj_stability_on_v30_candidate/final_submission_v34_task7authorproj_stability_on_v30_candidate.zip`
- Task7 changed rows vs `v30`: `259`
- Trace3 three-seed success: `True`
- 36-row calibration: stability gate `4`, baseline `6`, anchor1 `7`
- `88c5` seed `2027`: `john updike` -> `John Dos Passos` via `anchor1_stability_gate`
- `88c5` seed `2028`: `john updike` -> `John Dos Passos` via `anchor1_stability_gate`

## 3. Operational Implications
- The `author projection -> stability gate -> task7-only package` line is no longer research-only; it has now converted online.
- Future narrow candidates should branch from `v34`, not `v27` or `v30`.
- This readout records the official online score only. Leaderboard placement can still move as other teams submit.

## 4. Recommended Next Step
- Use v34 as the new base and search for a low-risk narrow patch worth at least +0.12 to challenge the current #6 score band.
