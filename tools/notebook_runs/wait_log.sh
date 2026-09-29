#!/usr/bin/env bash
# Poll a log until a success or failure marker appears, then print the tail.
# usage: wait_log.sh <logfile> <success_regex>
L="$1"; OK="$2"
FAIL='ERROR|error:|No matching distribution|Traceback|Could not find|Killed|command not found|fatal:'
for i in $(seq 1 360); do
  if grep -qE "$OK|$FAIL" "$L" 2>/dev/null; then break; fi
  sleep 10
done
echo "=== waiter fired for $L ==="
tail -30 "$L" 2>/dev/null
