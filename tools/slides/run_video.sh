#!/bin/bash
exec > /root/navlori/logs/run_video.log 2>&1
/root/navlori/venv/bin/python /root/navlori/scripts_local/run_video.py golden_run_10
echo "EXIT=$?"
