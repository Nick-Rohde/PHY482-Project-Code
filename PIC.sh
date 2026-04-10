#!/bin/bash --login
#SBATCH --job-name=Particle-In-Cell
#SBATCH --mem=8G
#SBATCH --time=72:00:00

/mnt/home/rohdeni2/miniconda3/envs/pmm_env/bin/python /mnt/home/rohdeni2/PIC/Main.py