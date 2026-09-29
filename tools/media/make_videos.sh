#!/usr/bin/env bash
exec > /root/navlori/logs/videos.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/make_videos.py
echo VIDEOS_EXIT=$?
