#!/usr/bin/env bash
exec > /root/navlori/logs/panel3.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /mnt/x/side_navlori/tools/media/make_3panel.py
echo PANEL3_EXIT=$?
