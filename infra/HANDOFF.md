# side_navlori on the fablab GPU workstation

How this machine is set up and how a Claude Code session works on it.
Current as of 2026-09-07. Migration from Colab is complete; the notebook runs on the
GTX 1080 inside WSL2. For what the project *is*, the user gives that in chat.

## The two rules

1. **One repo.** The only copy that matters is inside WSL: `/root/navlori/side_navlori`.
   From Windows it is reachable as `X:\navlori_wsl\side_navlori` (a symlink to
   `\\wsl$\Ubuntu-WSL2\root\navlori`). Open *that* folder in VS Code; edit *there*.
   The folders `X:\navlori_staging\...` and `X:\navlori\side_navlori` are stale earlier
   copies — do not edit them; they can be deleted once this setup has proven itself.
2. **One driver.** Exactly one Claude session works on this repo at a time: the one in
   the user's VS Code window (Remote-SSH to this Windows box). Earlier today two agents
   (one on the user's laptop, one here) both touched the WSL repo and one reset the
   other's edit. A `claude` binary is also installed inside WSL — unused, ignore it.

## Where things live

| What | Path (inside WSL) |
|---|---|
| repo (canonical) | `/root/navlori/side_navlori` |
| dataset, unpacked (3465 images) | `/root/navlori/data` |
| checkpoints (notebook resumes from here) | `/root/navlori/ckpt` |
| headless-run output notebooks | `/root/navlori/runs/` |
| logs (jupyter.log, notebook.log) | `/root/navlori/logs/` |
| the one venv (py3.10, torch 2.3.1+cu121) | `/root/navlori/venv` |
| Jupyter token | `/root/navlori/jupyter_token` |
| runner scripts (copies of `infra/*.sh`, CRLF stripped) | `/root/navlori/*.sh` |
| `/content` | symlink → `/root/navlori`, so every Colab path in the notebook works unchanged |

Machine: Windows 10, user `fablab` (is an Administrator; the VS Code Claude session runs
elevated). WSL2 Ubuntu 22.04, distro `Ubuntu-WSL2`, default user root, vhdx on `X:`.
GPU GTX 1080 8 GB (sm_61), driver 566.36, visible in WSL. 16 GB RAM, `.wslconfig` gives
WSL 12 GB / 8 GB swap / 4 CPUs (`wslconfig.txt`). `C:` is nearly full — keep everything on `X:`.
The user reaches this box over Tailscale; **never stop the Tailscale service.**

## How Claude runs things here

The session runs on the Windows side. Anything that needs the GPU, the venv, or Linux
runs inside WSL. Pattern that works (quoting through `wsl -- bash -c` breaks; Git Bash
rewrites `/mnt/c/...` arguments):

1. Write an **LF-only** script to a scratch file on Windows (Git Bash heredoc/printf).
2. Run it: `wsl -d Ubuntu-WSL2 -u root -- bash /mnt/c/<path>/script.sh`
   from PowerShell, or from Git Bash with `MSYS_NO_PATHCONV=1` in front.
3. Anything that must outlive the command: `nohup setsid bash script.sh > /dev/null 2>&1 &`
   inside the script. WSL keeps running while such a process exists.

Files can be edited directly at `X:\navlori_wsl\side_navlori\...` (or `\\wsl$\...`);
Windows git works there (`safe.directory` already added). Creating *directories* over
the share from Git Bash fails — do that inside WSL.
Scripts in `infra/` are committed LF (`.gitattributes`), but anything else written from
Windows may carry CRLF: `tr -d '\r'` before executing it in WSL.
Permissions: `.claude/settings.local.json` in the repo allows `wsl *` for Bash and PowerShell.

## Jupyter and headless runs

- **Jupyter Lab** (already running, survives VS Code disconnects):
  `bash /root/navlori/start_jupyter.sh` — idempotent; 127.0.0.1:8888, root dir `/root/navlori`,
  token in `/root/navlori/jupyter_token`, log `logs/jupyter.log`. WSL's 8888 appears on the
  Windows host's localhost, so from the laptop: `ssh -L 8888:localhost:8888 fablab`
  → `http://localhost:8888/?token=<token>`. Notebook path in Lab: `side_navlori/side_navlori.ipynb`.
- **Headless full run** (the "launch, disconnect, come back" mode):
  `nohup setsid bash /root/navlori/run_notebook.sh &` — papermill executes the repo notebook
  and saves progress after every cell to `runs/side_navlori.run.ipynb` (open it in Lab to watch);
  cell output streams to `logs/notebook.log`. The repo notebook itself is not modified.
- Resumability comes from the notebook, not the runner: every split/trial checkpoints to
  `/content/ckpt`, so re-running skips finished work. Retraining from scratch is fine
  (the old Drive checkpoints are not needed).
- **Not yet exercised:** no full run has been launched on this machine. The first real
  test is: launch `run_notebook.sh`, disconnect, reconnect, confirm it kept going.

## What to expect from the notebook on this GPU (vs Colab T4)

- fp32 throughput ≈ T4: ACE ~25 min × 5 splits, MS-T ~45 min × 5, Reloc3r ~30 min × 5,
  retrieval trio cheap, DPVO 5 trials + 5 LC trials. Full arc from scratch ≈ 9–11 GPU h.
- 8 GB VRAM (T4 had 16). Likely tight: Reloc3r-512 inference, MS-T training. An OOM is a
  real finding — report it; do not silently change the official recipes.
- The DPVO cell provisions its own Python 3.11 sidecar venv with `uv` (installed) — expected.
- The whole-lap GIF and results-table cells run on CPU at the end.

## Git / GitHub

- Remote: `github.com/moebachar/side_navlori`, branch `main`. WSL repo is at `origin/main`
  (`b8d188c`, which already contains the init-cell fix below) plus this file and `.gitignore`.
- Fetch from WSL works without credentials. **Push is untested from WSL** (no credential
  helper there); either `gh auth login` inside WSL once, or push with Windows git from
  `X:\navlori_wsl\side_navlori`, where Git Credential Manager is configured.
- Identity for commits: Mohamed BACHAR <j.elfirqi@gmail.com> (set in both git configs).

## Done / left

Done (2026-09-07): WSL outbound network OK natively · `setup_wsl.sh` → SETUP_OK (CUDA 12.1,
torch 2.3.1+cu121, GPU smoke test) · `deploy_wsl.sh` → DEPLOY_OK (code, data, `/content`,
uv, init-cell dry run) · Jupyter Lab up on 8888 · init cell fixed and committed
(`find_spec("google.colab")` raised `ModuleNotFoundError` on a bare venv; the check now
guards on the parent package first).

Left:
- First headless run + disconnect test (see above).
- GitHub push auth from this box.
- Delete the stale copies (`X:\navlori_staging\side_navlori_repo`, `X:\navlori\side_navlori`,
  `X:\navlori\start_claude_fresh.cmd`) once the user says so.

## Warnings

- **X: Recycle Bin** holds `navlori-codebase`, `navlori-research`, `navlori-training`,
  `navlori-infra`, `navlori-writing` — restorable until emptied. Do not empty it; remind
  the user to restore.
- Do not touch `X:\navlori-data`, `X:\navlori-fusion`, `X:\navlori-archive` unless asked.
- If WSL loses outbound network after a reboot/update: run `infra/fix_wsl_nat.ps1` from an
  elevated PowerShell (stale HNS "WSL" network; removing it and restarting WSL fixes it).
  Symptom: WSL→host works, `curl https://1.1.1.1` from WSL times out. Do not debug
  anything else first, and do not build a proxy/bridge — that path is obsolete.

## History (compressed)

2026-09-03: migration planned from the laptop; on-machine session hit the WSL NAT blocker,
fixed it (HNS), ran setup. 2026-09-04–07: a second laptop-driven "fresh start" checkout at
`X:\navlori\side_navlori` was created in parallel. 2026-09-07: this session verified the
network, ran setup + deploy, found the init-cell bug, started Jupyter; the laptop agent
committed the same fix as `b8d188c` and fast-forwarded the WSL repo to it. Consolidated
into the single-repo / single-driver model described at the top.
