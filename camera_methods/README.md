# camera_methods — the camera-modality roster (official code, unchanged)

Every method runs its **authors' official code untouched**; the files here are
thin adapters (our data → their format) and runners (install → official CLI →
outputs → common `pred.csv` contract). Repos are pinned to verified commits
(2026-08-13). Target: free Colab T4 GPU.

| Row | Method | Family | Official repo (pin) | License | Runtime (T4) |
|---|---|---|---|---|---|
| B1 | GeM-ResNet50 retrieval | retrieval floor | torchvision weights | BSD | ~3 min |
| B2 | MS-Transformer (ICCV'21) | absolute pose regression | yolish/multi-scene-pose-transformer @56ca699 | **none published** | ~30–60 min train |
| B3 | NetVLAD (CVPR'16) via hloc | classic learned retrieval | cvg/Hierarchical-Localization @c13273b | Apache-2.0 | ~15 min |
| R1 | ACE (CVPR'23) | scene coordinate regression | nianticlabs/ace @e9e90f2 | Niantic **non-commercial** | ~20–35 min |
| R2 | Reloc3r-512 (CVPR'25) | retrieval + relative pose | ffrivera0/reloc3r @761fac6 | CC BY-NC-SA 4.0 | ~25–45 min |
| R2b | DINOv2-SALAD (CVPR'24) | recent retrieval | serizba/salad @6aede13 (torch.hub) | GPL-3.0 | ~5 min |
| R3 | DPVO / DPV-SLAM (NeurIPS'23) | visual odometry | princeton-vl/DPVO @859bbbf | MIT | ~25 min first run |

## Contract

- Input: `$RUNS/cam_index.csv` — the Section-2b `cam` substrate table (frame,
  GT pose, quality class, `db_keep`, provisional split), written by the notebook.
- Database/training frames: `split == train & db_keep` (resting episodes
  deduplicated). Queries: all `split == test` frames, no filtering (honest).
- Output: `$RUNS/<method>/pred.csv` with `frame, x, y, yaw` — base_footprint
  pose in the map frame (`yaw = nan` where not estimable). DPVO instead reports
  Sim(3)-aligned translation ATE (monocular scale is arbitrary) + aligned xy.
- `smoke_eval.py <method>` prints SMOKE metrics on the provisional split —
  plumbing validation only; the real protocol is frozen in the eval section.

## Geometry & calibration decisions (paper notes)

- Nominal, never-calibrated intrinsics (±1–2%) and unmodeled rolling shutter are
  dataset-level limitations shared by all geometric rows (ACE, Reloc3r, DPVO).
- ACE/Reloc3r inputs are undistorted with the nominal plumb_bob model;
  retrieval/APR rows use raw frames (no geometry consumed).
- ACE requires fx == fy: K files carry f_mean = 642.94 (0.2% averaging error).
- Reloc3r inputs are resized to 640×480 (its loader crops around a hardcoded
  principal point — raw 820×616 would be silently cropped asymmetrically).
- Poses handed to ACE/Reloc3r are 4×4 cam-to-world in OpenCV camera axes, built
  as `T_world_base(x, y, yaw) @ T_base_cam` (exact extrinsic from calibration).

## Session order (single Colab GPU session)

gem → salad → netvlad → mst → ace → reloc3r → dpvo (own venv, torch 2.3.1 pin).
If a later install breaks an earlier method's imports: Runtime → Restart session,
re-run the setup cell, then only the remaining methods.
