#!/usr/bin/env bash
# Stream only the events worth waking for: run-start, attempt boundaries,
# failures/watchdog, and completion. Routine GPU-wait heartbeats are excluded.
tail -F -n 0 /root/navlori/logs/camera_run.log 2>/dev/null | stdbuf -oL grep -E \
  'GPU free|=== attempt|WATCHDOG|Exception encountered|Traceback|error:|ERROR|FAIL|Killed|OOM|No matching distribution|fatal:|no kernel image|CUDA error|RUN_DONE|FINISHED OK|STOPPED'
