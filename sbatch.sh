#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --mem=0 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --time=8:00:00

module load python/3.11
source venv/bin/activate
python proj_2a.py
