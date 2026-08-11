#!/bin/bash

YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
OUTDIR="sbatch-logs/$YEAR/$MONTH/$DAY"
mkdir -p "$OUTDIR"

JOBNAME="LLM_evolution_tournament"
sbatch --job-name="$JOBNAME" \
       --output="$OUTDIR/${JOBNAME}-%j.out" \
       --error="$OUTDIR/${JOBNAME}-%j.err" \
       run_job.sh