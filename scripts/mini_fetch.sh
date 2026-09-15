#!/bin/bash
# Pull finished npz/json results (and the job log) back from the mini.
set -e
HOST=ee@mini-mini.local
REMOTE=~/fractal-basins-lab
rsync -a --include '*/' --include '*.npz' --include '*.json' --include 'job.log' \
      --exclude '*' "$HOST:$REMOTE/runs/" ./runs/
rsync -a "$HOST:$REMOTE/runs/job.log" ./runs/mini_job.log 2>/dev/null || true
echo "synced. exp01 dirs now:"
find runs/exp01 -name '*.npz' | sort
