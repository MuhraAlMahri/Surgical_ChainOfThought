#!/bin/bash
#SBATCH --job-name=surgery_r1_split
#SBATCH --output=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/scripts/slurm/logs/make_surgery_r1_split_%j.out
#SBATCH --error=/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/scripts/slurm/logs/make_surgery_r1_split_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos

# Create EndoVis-18-VQLA train/val/test splits following the Surgery-R1 paper
# IMPORTANT: Runs on compute nodes (MBZUAI HPC policy)

set -e

echo "=========================================="
echo "Creating Surgery-R1 Split for EndoVis-18-VQLA"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "=========================================="

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
cd "${BASE_DIR}"

# Configuration
# IMPORTANT: Set QA_JSON to the path of your EndoVis-18-VQLA qa_pairs.json file
# Expected location: ${BASE_DIR}/datasets/EndoVis-18-VQLA/qa_pairs.json
# You can either:
#   1. Update the path below, OR
#   2. Submit with: QA_JSON=/path/to/qa_pairs.json sbatch make_surgery_r1_split.slurm
QA_JSON="${QA_JSON:-${BASE_DIR}/datasets/EndoVis-18-VQLA/qa_pairs.json}"
OUT_DIR="${BASE_DIR}/corrected_1-5_experiments/datasets/endovis18_surgery_r1_split"

# Create output directory and logs directory
mkdir -p "${OUT_DIR}"
mkdir -p "${BASE_DIR}/corrected_1-5_experiments/scripts/slurm/logs"

# Check if input file exists
if [ ! -f "${QA_JSON}" ]; then
    echo "ERROR: Input file not found: ${QA_JSON}"
    echo "Please update the QA_JSON path in this script."
    exit 1
fi

echo ""
echo "Input file: ${QA_JSON}"
echo "Output directory: ${OUT_DIR}"
echo ""

# Run the split script
python corrected_1-5_experiments/scripts/make_surgery_r1_split.py \
    --qa_json "${QA_JSON}" \
    --out_dir "${OUT_DIR}"

echo ""
echo "=========================================="
echo "Split Creation Completed"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Output files:"
echo "  - ${OUT_DIR}/train.json"
echo "  - ${OUT_DIR}/val.json"
echo "  - ${OUT_DIR}/test.json"
echo "=========================================="

