#!/bin/bash
exec > /root/navlori/logs/run_video.log 2>&1
/root/navlori/venv/bin/python /mnt/x/side_navlori/tools/slides/run_video.py golden_run_10
echo "EXIT=$?"
