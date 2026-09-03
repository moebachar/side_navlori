# Handoff: migrate side_navlori onto the fablab GPU workstation

Context for the Claude Code session running ON the fablab machine (Windows 10, user
`fablab`, non-admin). Written 2026-09-03 by the session that drove the migration
remotely from the user's laptop. Goal: everything (code, data, one venv, Jupyter,
Claude) local on this machine, notebook runs on the GTX 1080 inside WSL2.

## Machine facts (verified)

- WSL2 Ubuntu 22.04 (`Ubuntu-WSL2`, default user **root**), vhdx on `X:\Ubuntu`, ~950 GB free.
- GPU: GTX 1080 8 GB (sm_61) — **visible inside WSL** (`nvidia-smi -L` works). Driver 566.36.
- Host: 16 GB RAM, i7-6700K. `C:\Users\fablab\.wslconfig` already set to 12 GB / 8 GB swap / 4 CPUs (see `wslconfig.txt`).
- `C:` has only ~18 GB free — put nothing heavy there. `X:` (3 TB) is the big disk.
- `X:\navlori_staging\` holds: `side_navlori_repo.tar` (repo incl. .git, commit c492ebd),
  `data.zip` (canonical dataset, 348 MB — same file Colab uses).

## THE BLOCKER — WSL has no outbound network

WSL2 NAT forwarding is broken: raw-IP internet (`curl https://1.1.1.1`) times out.
Tried and did NOT fix it: winnat restart, full reboot, disabling the VirtualBox NDIS6
binding on both `vEthernet (WSL)` and physical `Ethernet`. What DOES work:
- WSL → Windows host TCP on port 22 (`/dev/tcp/172.25.208.1/22` connects).
- WSL interop → Windows exes have full network (`/mnt/c/Windows/System32/curl.exe` → 200).

Fix options, in order of preference:
1. Keep debugging the host NAT (suspects left: Tailscale's WFP filters, other filter
   drivers — Oculus/Gameroom junk is installed; or HNS state: as admin,
   `Get-HnsNetwork | ? Name -eq WSL | Remove-HnsNetwork` then `wsl --shutdown`
   rebuilds the WSL network from scratch). Native NAT is the best end state.
2. **Interop bridge (no admin, no keys, proven premise)**: run `proxy_hostnet.py`
   (in this dir) inside WSL — a loopback HTTP proxy on `127.0.0.1:3128` that opens
   every upstream connection by spawning stock Windows PowerShell running
   `bridge_win.ps1` (also in this dir) via WSL interop; the Windows side has working
   network. Copy `bridge_win.ps1` to `X:\navlori_staging\` (the proxy hardcodes that
   path — adjust if moved), start `python3 proxy_hostnet.py`, then set
   `http_proxy/https_proxy=http://127.0.0.1:3128` + apt conf + flip sources.list to
   https (CONNECT-only is cleaner). The user has approved this direction in principle;
   confirm before making it permanent (autostart in .bashrc).
3. ssh -W variant of the same (WSL ssh → host sshd 127.0.0.1:22 with a new restricted
   key in `C:\Users\fablab\.ssh\authorized_keys`) — works too, but option 2 avoids
   touching authorized_keys.

Note: `/etc/wsl.conf` has `generateResolvConf=false` (set during debugging) and
`/etc/resolv.conf` is currently MISSING — once network works, either write
`nameserver 1.1.1.1` or revert wsl.conf. With the bridge, DNS resolves on the
Windows side (proxy CONNECT by hostname), so resolv.conf barely matters.

## Then: run these, in order (inside WSL, as root)

1. `setup_wsl.sh` — apt packages, CUDA toolkit 12.1 (for compiling ACE's dsacstar and
   DPVO's extensions; exports TORCH_CUDA_ARCH_LIST=6.1), **the one venv** at
   `/root/navlori/venv`: Python 3.10, `torch==2.3.1+cu121` (pin already validated for
   DPVO on Colab; wheels ship sm_61 kernels), numpy 1.26.4, jupyterlab. Ends with a
   GPU smoke test. Idempotent. If the bridge is in use, export the proxy env first.
2. `deploy_wsl.sh` — unpacks staging into `/root/navlori/side_navlori`, extracts
   data.zip, creates the **`/content` → `/root/navlori` symlink** (this is the trick:
   every Colab-hardcoded path in the notebook works verbatim), installs uv (the DPVO
   notebook cell provisions its own 3.11 sidecar venv with it — expected, leave it),
   dry-runs the notebook's init cell, installs Claude Code in WSL.
3. Jupyter: `source /root/navlori/venv/bin/activate && cd /root/navlori/side_navlori
   && jupyter lab --no-browser --port 8888 --allow-root`. The user reaches it via
   `ssh -L 8888:localhost:8888 fablab` from their laptop (WSL:8888 is visible on the
   Windows host's localhost automatically).
4. Open `side_navlori.ipynb` → Run all. The workstation branch (commit c492ebd) is
   already in the notebook: init cell detects non-Colab, checkpoint store becomes
   `/content/ckpt` (= `/root/navlori/ckpt`). Everything checkpoints/resumes; the
   4-fold Protocol A run re-trains what it needs (user said retraining from scratch
   is fine — the Drive checkpoints are NOT needed).

## Notebook expectations on this GPU (vs the Colab T4 baseline)

- fp32 throughput ≈ T4, so timings roughly match: ACE ~25 min × 5 splits, MS-T
  ~45 min × 5, Reloc3r ~30 min × 5, retrieval trio cheap, DPVO 5 trials + 5 LC trials.
  Full arc from scratch ≈ 9–11 GPU h, resumable at every split.
- 8 GB VRAM (T4 had 16). Expected tight spots: Reloc3r-512 inference and MS-T
  training. If OOM: that's a real finding — report it, don't silently change the
  official recipes.
- The whole-lap GIF cell and results-table cell run on CPU at the end.

## Repo / git

- Canonical: github.com/moebachar/side_navlori (private). HEAD = c492ebd. The tar in
  staging is that commit. For push/pull from this machine the user must set up auth
  (gh login or a deploy key) — not done yet.
- User identity for commits: Mohamed BACHAR <j.elfirqi@gmail.com>.

## Warnings

- **X: Recycle Bin** contains `navlori-codebase`, `navlori-research`,
  `navlori-training`, `navlori-infra`, `navlori-writing` — restorable until someone
  empties the bin. Do not empty it; remind the user to restore.
- Don't touch `X:\navlori-data`, `X:\navlori-fusion`, `X:\navlori-archive` (the
  user's disk-X restore) without being asked.
- Windows-authored shell scripts may carry CRLF — run `sed -i 's/\r$//'` (or
  `tr -d '\r'`) before executing anything from this dir inside WSL.
