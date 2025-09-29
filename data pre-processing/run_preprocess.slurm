#!/bin/bash
#SBATCH --account=csc1501s262
#SBATCH --partition=qCPU24          # A standard CPU partition is fine for this
#SBATCH --job-name=create-dataset
#SBATCH --nodes=1
#SBATCH --time=08:00:00             # 8 hours should be a safe time limit
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=preprocess-%j.out

echo "Setting up environment for dataset creation..."
module load python/3.11

# Change to the directory containing this script to ensure relative paths work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating and activating virtual environment..."
python -m venv venv-preprocess
source venv-preprocess/bin/activate

echo "Installing necessary libraries..."
pip install --upgrade pip
pip install datasets pyarrow # PyArrow is needed for saving to disk efficiently

echo "Running the preprocessing script..."
python preprocess.py

echo "Job finished."


