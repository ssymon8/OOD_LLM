#!/bin/bash
#SBATCH --job-name=ministral_OOD
#SBATCH --partition=ENSTA-h100
#SBATCH --output=ministral_OOD_%j.log
#SBATCH --error=ministral_OOD_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodelist=ensta-h10001.r2.enst.fr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Load required modules
echo "Loading CUDA modules..."
module load cuda/12.0 2>/dev/null || module load cuda 2>/dev/null || true
module load pytorch 2>/dev/null || true

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