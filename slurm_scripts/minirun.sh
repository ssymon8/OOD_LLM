#!/bin/bash
#SBATCH --job-name=ministral_OOD
#SBATCH --partition=ENSTA-l40s
#SBATCH --output=ministral_OOD_%j.log
#SBATCH --error=ministral_OOD_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodelist=ensta-l40s01.r2.enst.fr,
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Load required modules
echo "Loading CUDA modules..."
# Load environment
echo "Activating virtual environment..."
source ~/OOD_LLM/.venv/bin/activate

# Navigate to project directory
cd ~/OOD_LLM

# Verify model exists
if [ ! -d "ministral" ]; then
    echo "ERROR: Model directory not found at ~/OOD_LLM/ministral"
    exit 1
fi

echo "Model found. Starting inference..."
echo "=========================================="

# Run inference
srun python3 run_ministral.py

echo "End of inference!"