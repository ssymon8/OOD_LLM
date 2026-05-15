#!/bin/bash
#SBATCH --job-name=ministral_OOD
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=ministral_OOD_%j.log
#SBATCH --error=ministral_OOD_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# Load environment
source ~/OOD_LLM/.venv/bin/activate

# Navigate to project directory
cd ~/OOD_LLM

# Run inference
srun python run_ministral.py

echo "End of inference!"