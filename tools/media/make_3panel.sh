#!/usr/bin/env bash
exec > /root/navlori/logs/panel3.log 2>&1
export MPLBACKEND=Agg
/root/navlori/venv/bin/python /root/navlori/scripts_local/make_3panel.py
echo PANEL3_EXIT=$?
