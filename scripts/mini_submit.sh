#!/bin/bash
# Submit an experiment to the mini (Apple M4, MPS) as a detached job.
# usage: scripts/mini_submit.sh <exp_script> [FB_RES=200 FB_SEEDS=0,1,2 ...]
set -e
HOST=ee@mini-mini.local
REMOTE=~/fractal-basins-lab
SCRIPT=$1; shift

rsync -a --exclude .venv --exclude __pycache__ --exclude 'runs/figs' \
      ./ "$HOST:$REMOTE/"
ssh "$HOST" "mkdir -p $REMOTE/runs"
ssh "$HOST" "bash -lc 'cd $REMOTE && \
  nohup env $* .venv/bin/python $SCRIPT > runs/job.log 2>&1 & echo \"mini pid: \$!\"'"
echo "submitted $SCRIPT to mini; fetch results later with scripts/mini_fetch.sh"
