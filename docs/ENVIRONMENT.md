# Environment: the compute box

Everything runs on **one Windows workstation with a WSL2 Linux box** (the "fablab
box"). Heavy work runs inside WSL. Layout as of the 2026-09-29 cleanup.

## Machine

- **OS:** Windows 10; WSL2 distro **`Ubuntu-WSL2`** (clock on UTC, Windows on local time).
- **GPU:** NVIDIA **GTX 1080**, 8 GB, compute capability **sm_61**, CUDA 12.1.
  It also drives the display: filling it freezes the whole PC.
- **Three project folders, nothing else on X:**

| Windows | WSL | what |
|---|---|---|
| `X:\side_navlori` | `/mnt/x/side_navlori` | this sandbox (notebooks, `tools/`, `docs/`) |
| `X:\navlori-fusion` | `/mnt/x/navlori-fusion` | main code; other agents work in `.worktrees\` |
| `X:\navlori-data` | `/mnt/x/navlori-data` | data vault (git + DVC) |

`side_navlori\data` → `navlori-data\robot` and `side_navlori\dataset_pipeline` →
`navlori-fusion\scripts\dataset` are **junctions**; WSL follows them through `/mnt/x`.
Both are **read-only for this agent** (owned by the dataset agent).

## The GPU lock (mandatory)

One GPU, four agents. Every GPU job goes through the shared lock:
```bash
LOCK="/root/navlori/venv/bin/python /mnt/x/navlori-fusion/scripts/gpu_lock.py"
$LOCK status
$LOCK run --wait --who side_navlori --what "camera notebook" -- <command>   # foreground
$LOCK acquire --wait --who side_navlori --what "..."                        # detached jobs:
trap '$LOCK release --who side_navlori' EXIT                                 # release on any exit
```
`tools/notebook_runs/run_camera.sh` and `rerun_figs.sh` already do this. The WiFi
notebook is CPU-only (see below) and needs no lock.

## Running commands in WSL

From a Windows shell:
```
wsl -d Ubuntu-WSL2 -u root -- bash /mnt/x/side_navlori/tools/notebook_runs/<script>.sh
```
- **Write scripts as files and run them by path.** Inline bash with `$vars`, loops or
  redirects gets mangled by the Windows→WSL quoting layer (and Git Bash needs
  `MSYS_NO_PATHCONV=1`). `.sh` files are LF via `.gitattributes`.
- Long jobs (notebook execution, deep methods) take minutes to hours. Make the script
  self-logging (`exec > /root/navlori/logs/<name>.log 2>&1`), launch it detached so it
  survives a disconnect, and poll the log:
  ```powershell
  Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-d','Ubuntu-WSL2','-u','root','--','bash','/mnt/x/side_navlori/tools/notebook_runs/run_camera.sh'
  ```

## Python environments (WSL, machine-local, not in git)

| venv | path | purpose |
|---|---|---|
| **notebook kernel** | `/root/navlori/venv` | both notebooks, plotting scripts. Python 3.10, torch 2.5.1+cu121, jupyter/nbconvert/papermill, numpy/pandas/scipy/matplotlib, `cv2` 5.0, `rosbags`. Rebuild: `tools/notebook_runs/01_make_venv.sh`. |
| **CNNLoc sub-venv** | `/root/navlori/venvs_wifi/cnnloc` | Python **3.7** / TF **1.15.5** / Keras 2.2.5 (micromamba). Runs CNNLoc's official code unchanged. Build: `tools/notebook_runs/build_cnnloc_venv2.sh`. |
| **Kim sub-venv** | `/root/navlori/venvs_wifi/kim` | same legacy stack; Kim's SAE. Build: `tools/notebook_runs/build_kim_venv.sh`. |
| **DPVO sub-venv** | `/root/navlori/venvs/dpvo` (= `/content/venvs/dpvo`) | built by the camera notebook's DPVO cell. |

- Competitor code is cloned (not in git) into `/root/navlori/mrepos_wifi/{cnnloc,kimdnn}`
  and `/root/navlori/mrepos/{hloc,mst,ace,reloc3r,DPVO}`.
- The WiFi sub-venvs are **CPU-forced** (TF 1.15 needs CUDA 10; the models are tiny).
- micromamba: `/root/navlori/mamba/bin/micromamba`.

## Where the notebooks read and write

- **Data:** `DATA = "/mnt/x/side_navlori/data/<run>"`, read straight through the junction
  (about 1 s to load a run; no native copy needed). If camera-image I/O ever becomes the
  bottleneck, copy the runs you need to a native WSL folder and delete the copy after.
- **Loader:** `sys.path.append("/mnt/x/side_navlori/dataset_pipeline/export")`, then
  `from load_dataset import Dataset`.
- **Scratch:** `/content` is a symlink to `/root/navlori` (native ext4). Method repos,
  checkpoints (`/content/ckpt`), per-method outputs (`/content/runs`) and logs
  (`/root/navlori/logs`) live there. Colab is no longer supported: the data is only in the vault.

## Running a notebook headless (to embed outputs)

```
/root/navlori/venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
  /mnt/x/side_navlori/side_navlori_wifi.ipynb
```
(`tools/notebook_runs/run_wifi_nb.sh`). For the camera notebook use
`tools/notebook_runs/run_camera.sh` (GPU lock, retry + resume, stall watchdog).
- Use the **default inline backend** so figures embed. **Do not** set `MPLBACKEND=Agg`
  for notebook execution; do set it for standalone plotting scripts.
- WiFi §3 shells out to the sub-venvs via subprocess, so a headless run reproduces the
  deep-method numbers too.

## Scratch and git

Temporary scripts go in the WSL home (e.g. `/root/navlori/tmp/`) or the session
scratchpad, never in a new top-level folder on X:. Versioned here: the notebooks,
`tools/`, `docs/`, `slides_progress/PROMPTS.md`. Everything heavy is git-ignored.
