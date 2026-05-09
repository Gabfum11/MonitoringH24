#!/bin/bash
cd /Users/xiaomicamera/Documents/dev/MonitoringH24
source venv/bin/activate
PYTHONUNBUFFERED=1 nohup python Monitor.py > monitor.log 2>&1 &
echo "Monitor avviato (PID: $!)"