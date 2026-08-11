srun --account=<your-slurm-account> --time=1:00:00 --gres=gpu:l40s:2 --mem=48G -c 32 --chdir=/path/to/your/checkout --pty bash
srun --account=<your-slurm-account> --time=1:00:00 --gres=gpu:h100:2 --mem=48G -c 32 --chdir=/path/to/your/checkout --pty bash
sinfo -N --Format=NodeHost,Partition,GresUsed,Gres,CPUsState
squeue -u $USER
srun --account=<your-slurm-account> --time=1:00:00 --gres=gpu:l40s:2 --mem=80G -c 24 --chdir=/path/to/your/checkout --pty bash

srun --account=<your-slurm-account> --time=1:00:00 --gres=gpu:h100:1 --mem=60G -c 32 --chdir=/path/to/your/checkout --pty bash

# Increase -c to load model faster

srun --account=<your-slurm-account> --time=1:00:00 --mem=24G -c 32 --chdir=/path/to/your/checkout --pty bash
