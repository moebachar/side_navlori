# Environment — the compute box

Everything runs on **one Windows workstation with a WSL2 Linux box** (the "fablab
box"). Do heavy work inside WSL.

## Machine

- **OS:** Windows 10; WSL2 distro **`Ubuntu-WSL2`**.
- **GPU:** NVIDIA **GTX 1080**, 8 GB, compute capability **sm_61**, CUDA 12.1.
  Verified: `torch 2.5.1+cu121`, `cuda.is_available() == True`.
- **Repo:** `X:\side_navlori` (Windows) == **`/mnt/x/side_navlori`** (WSL).

## Running commands in WSL

If your terminal is Windows-side:
```
wsl -d Ubuntu-WSL2 -u root -- bash -lc '<command>'
```
If you can open a shell directly in the distro, just use it. **Gotchas when
bridging from a Windows shell:**
- Line endings: author any script as a **file** and strip CRLF
  (`sed 's/\r$//'`) before running it in WSL; complex inline bash with
  `$vars`/loops/redirects gets mangled by the Windows→WSL quoting layer. Prefer
  **script files invoked by path** over long inline one-liners.
- Long jobs (SLAM, deep-method runs, notebook execution) take **minutes**. Run
  them non-interactively, have the script `exec > logfile 2>&1`, and poll the
  log. (The previous agent launched durably via
  `Start-Process -WindowStyle Hidden wsl.exe ...`; use whatever backgrounding
  your host provides — just make jobs self-logging.)

## Python environments

| venv | path | purpose |
|---|---|---|
| **notebook kernel** | `/root/navlori/venv` | runs both notebooks + all export/SLAM/plotting scripts. Python 3.10; has torch(cu121), jupyter/nbconvert, numpy/pandas/scipy/matplotlib, `cv2` (5.0), `rosbags` (pure-python rosbag2 reader). |
| **CNNLoc sub-venv** | `/root/navlori/venvs_wifi/cnnloc` | Python **3.7** / TF **1.15.5** / Keras 2.2.5 (built via **micromamba**, uv dropped 3.7). Runs CNNLoc's official code unchanged. |
| **Kim sub-venv** | `/root/navlori/venvs_wifi/kim` | same legacy stack; runs Kim's SAE. |

Official competitor code lives in `/root/navlori/mrepos_wifi/{cnnloc,kimdnn}`
(cloned, not in git). The sub-venvs are **CPU-forced** (TF1.15 needs CUDA10; the
box has CUDA12 — irrelevant, the models are tiny and identical on CPU).

micromamba binary: `/root/navlori/mamba/bin/micromamba`. Sub-venv build scripts:
`/root/navlori/scripts_local/build_{cnnloc,kim}_venv.sh`.

## Two copies of the dataset (important)

- **Git/canonical copy:** `/mnt/x/side_navlori/data` (= `X:\side_navlori\data`).
  All the new recording work (raw bags `navlori_*`, `staging/`) is here.
- **WSL native copy the notebooks load:** `/root/navlori/data`, reached via the
  symlink **`/content` → `/root/navlori`**, so notebooks use
  `DATA = "/content/data/run2"` etc. It currently holds only `run1`, `run2`,
  `scripts`.

⇒ When you promote a new run to `data/runN`, the notebooks won't see it until
it's under `/root/navlori/data/runN` too. Either **copy the promoted run into
`/root/navlori/data/`** (matches how run1/run2 are loaded) or repoint the
notebook `DATA` at a `/mnt/x/...` path. Keep the git copy canonical.

## Running a notebook headless (to embed outputs)

```
/root/navlori/venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
  /mnt/x/side_navlori/side_navlori_wifi.ipynb
```
- Use the **default inline backend** so figures embed. **Do NOT** set
  `MPLBACKEND=Agg` for notebook execution — it suppresses the embedded figures.
- For standalone plotting/animation **scripts**, the opposite: set
  `MPLBACKEND=Agg` (headless, no display).
- WiFi §3 shells out to the sub-venvs via subprocess; that works from inside the
  notebook kernel, so a full headless run reproduces the deep-method numbers too
  (a few minutes).

## Scratch

Put temp scripts/outputs in a scratch dir (e.g. `/root/navlori/scripts_local/`
or a tmp dir), not in the repo. `figures/`, `_inspect/`, rosbags and zips are
gitignored — keep generated heavy artifacts out of git.
