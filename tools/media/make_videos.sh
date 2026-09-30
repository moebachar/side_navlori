#!/usr/bin/env bash
exec > /root/navlori/logs/videos.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /mnt/x/side_navlori/tools/media/make_videos.py
echo VIDEOS_EXIT=$?
