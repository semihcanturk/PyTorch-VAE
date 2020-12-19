#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=v100:2
#SBATCH --mem=10G
#SBATCH -o /home/mila/s/semih.canturk/VAELogs/slurm-%j.out
module load python/3.7
source $HOME/venv/bin/activate
curl https://notify.run/tFWswgilC3IKNpRb -d "Hello from notify.run"
python run.py --config ./configs/cvae.yaml
curl https://notify.run/tFWswgilC3IKNpRb -d "Script complete!"

cp $SLURM\_TMPDIR/output /home/mila/s/semih.canturk/VAELogs/
