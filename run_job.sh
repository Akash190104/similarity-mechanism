#!/bin/bash
#SBATCH --job-name=LLM_evolution_tournament
#SBATCH --account=<your-slurm-account>
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --chdir=/path/to/your/checkout

source .venv312/bin/activate
export PYTHONPATH=.

python3 script/run_evolution.py --config legacy/prisoner_dilemma.yaml
