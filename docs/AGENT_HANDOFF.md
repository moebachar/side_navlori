# navlori — Agent Handoff

You are taking over the **side_navlori** project for a few days. This folder
(`docs/`) is your onboarding. Read all four files before touching anything:

1. **AGENT_HANDOFF.md** (this file) — project, state, pending work, guardrails.
2. **ENVIRONMENT.md** — the compute box (WSL2 + GPU), venvs, how to run things.
3. **DATA.md** — dataset layout, the exporters/SLAM pipeline, the new recordings.
4. **METHODS.md** — the WiFi/camera evaluation methods and how to re-run them.

> The previous agent could not access this repo's history in its own memory, and
> neither can you: **this doc set is the single source of truth.** If something
> here conflicts with the code, trust the code and update the doc.

---

## 1. What navlori is

**navlori** = robot **indoor localization with deep learning** — answering
"**where am I?**" from a robot's onboard sensors, where GPS doesn't work.
`side_navlori` is the **experimental sandbox**: a place to test methods on a
real **TurtleBot3 Waffle Pi** recording, away from the main project's
assumptions.

**End goal:** fair, reproducible validation numbers (localization error in
metres) for a **journal paper**, comparing classic baselines and recent SOTA
methods per sensor, then a personal contribution.

## 2. Deliverables & method (how this project works)

- The deliverables are **minimalist, "pitchable" notebooks** at the repo root:
  `side_navlori_<sensor>.ipynb` (currently `side_navlori_camera.ipynb`,
  `side_navlori_wifi.ipynb`).
- Each notebook is **one reveal per section**, entire pipeline visible in the
  cells, explanations kept in chat (not the notebook), executed with **outputs
  and figures embedded** (they must render on GitHub / in a pitch).
- Per sensor the method is: **(1)** understand the sensor → **(2)** clean /
  transform → **(3)** evaluate **2–3 classic baselines + 1–2 recent competitors
  using their official open-source code UNCHANGED, on the local GPU box** →
  **(4)** later, a personal innovation.
- SOTA "fair testing" rule: run each competitor's **original code**, each in its
  **own isolated sub-venv**, never a re-implementation. See METHODS.md.

## 3. Current state (as of this handoff)

**Done & committed:**
- `side_navlori_camera.ipynb` — camera notebook, executed with local results:
  6 methods + DPVO (commit `621a027`).
- `side_navlori_wifi.ipynb` — WiFi notebook complete through **§3**:
  §1 at-a-glance, §2 data understanding, **§3 methods** = 3 baselines
  (RADAR, Weighted-kNN, Horus) + **CNNLoc** + **Kim** (official code, isolated
  Py3.7/TF1 sub-venvs), protocols A/B, results table (§3c), per-method map view
  (§3d), real-time GIF (§3e). Commits `aa200cd`,`a3426dd`,`6cce782`,`59f864e`.
- Datasets `data/run1` (CESI 2026-07-24) and `data/run2` (2026-09-16), each
  fully exported with lidar-SLAM ground truth. `data/scripts/` holds the
  reproducible pipeline. See DATA.md.

**⚠️ Git:** the 4 WiFi §3 commits are **committed but NOT pushed**
(`main` is 4 ahead of `origin/main`). Push them (with the user's OK) so this
handoff state is on GitHub: `git push origin main`.

**In progress — THE MAIN TASK (see §4):** a **larger dataset was just recorded**
— six runs on the physical robot, sitting as raw rosbags in `data/navlori_*`
and half-processed in `data/staging/*` (lidar-SLAM ground truth built; camera
and WiFi **not yet exported**). These need finalizing, then the notebooks
re-run on the bigger data.

**Pending / not started:**
- **Camera intrinsic calibration** at 640×480 — the recordings use *nominal*
  intrinsics only (never calibrated on this robot). A checkerboard PDF is at
  `calib/checkerboard_9x6_25mm_A4.pdf`. See DATA.md §Calibration.
- Personal innovation (step 4) — later.

## 4. The main task: finalize the new dataset, then re-run everything

Six runs were recorded on 2026-09-17 (see DATA.md for the table + SLAM quality).
The immediate pipeline:

1. **Pick which run(s) to keep.** All six SLAM'd cleanly (7.7–12.3 mm median);
   the user believes one should be dropped but decides by looking at
   `data/staging/_compare_gt.png` (trajectory + per-pose error, all six).
   **This is the user's call — ask them; do not delete a run on your own.**
2. **Export camera + WiFi** for the kept runs (only lidar/telemetry are done in
   staging). Use `data/scripts/export_camera.py` and `export_wifi.py`. See
   DATA.md for the exact per-bag procedure (the exporters have hardcoded paths).
3. **Promote** kept runs to `data/run3`, `run4`, … (same layout as run1/run2),
   write a `data/runN/README.md` dataset card each, and **remove the raw
   `data/navlori_*` bags** once export is verified (they are large & gitignored).
4. **Camera calibration** at 640×480 → drop the intrinsics YAML into each run's
   `calib/`.
5. **Re-run the notebooks on the enlarged dataset** for the real numbers — this
   is the whole point. The WiFi §3 harness and both sub-venvs are already built
   to re-run untouched; just point `DATA` at the new run(s). See METHODS.md.

## 5. Guardrails (do not violate)

- **Notebooks are the deliverable** — keep them minimal, one reveal per section,
  outputs embedded. Prose/analysis goes to the user in chat, not into the .ipynb.
- **Official competitor code stays UNCHANGED** and runs in its own sub-venv.
- **Git:** never add attribution / "Co-authored-by" / tool signatures to commit
  messages. **Do not `git push` unless the user asks.** Commit at natural
  checkpoints when the user says so.
- **Large files stay out of git** — camera images, rosbags, zips, `figures/`,
  `_inspect/` are gitignored; keep it that way.
- **Don't trust stale artifacts** — `README.md` at the repo root predates the
  per-sensor-notebook structure; this `docs/` set supersedes it.
- The compute box, its data copy, and the sub-venvs are **machine-local**
  (see ENVIRONMENT.md). Re-runs happen there.

## 6. First moves

Read ENVIRONMENT.md → DATA.md → METHODS.md, then confirm the environment is
reachable (open the WiFi notebook's setup cell path, list `data/staging/`), and
report your understanding + the plan for step 4.1 back to the user before acting.
