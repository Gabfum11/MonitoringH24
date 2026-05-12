#!/bin/bash
PID=$(pgrep -f Monitor.py)
if [ -z "$PID" ]; then
    echo "Monitor non in esecuzione"
    exit 0
fi
kill -INT $PID
echo "Monitor fermato, diario in generazione..."
# Attende che il processo finisca (max 3 minuti)
TIMEOUT=180
COUNT=0
while kill -0 $PID 2>/dev/null; do
    sleep 2
    COUNT=$((COUNT + 2))
    if [ $COUNT -ge $TIMEOUT ]; then
        echo "Timeout, forzatura arresto..."
        kill -9 $PID 2>/dev/null
        break
    fi
done
echo "Monitor terminato."
