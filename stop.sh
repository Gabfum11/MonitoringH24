#!/bin/bash
kill -INT $(pgrep -f Monitor.py)
echo "Monitor fermato, diario in generazione..."