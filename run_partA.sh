#!/usr/bin/env bash
# =============================================================================
# run_partA.sh — Launch dfs_PartA_tuning.py on the processing node
#
# Usage:
#   chmod +x run_partA.sh          # first time only
#   ./run_partA.sh                 # standard run
#   ./run_partA.sh --resume        # resume an interrupted Optuna study
#   nohup ./run_partA.sh > logs/partA_run.log 2>&1 &   # run in background
# =============================================================================

set -euo pipefail   # exit on error, undefined variable, or pipe failure

# ── Configuration ─────────────────────────────────────────────────────────────
CONDA_ENV="dl"
SCRIPT="dfs_PartA_tuning.py"
LOG_DIR="logs"
OUTPUTS_DIR="outputs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/partA_${TIMESTAMP}.log"

# ── Setup output and log directories ──────────────────────────────────────────
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUTS_DIR"

# ── Locate conda ──────────────────────────────────────────────────────────────
# Checks common install locations; adjust CONDA_BASE if yours differs
if   [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/anaconda3"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$HOME/miniconda3"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/opt/conda"
else
    echo "❌ conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── Activate environment ───────────────────────────────────────────────────────
echo "⚙️  Activating conda environment: ${CONDA_ENV}"
conda activate "$CONDA_ENV" || {
    echo "❌ Environment '${CONDA_ENV}' not found."
    echo "   Run: conda env create -f environment-linux.yml"
    exit 1
}

# ── Confirm we're in the right directory ──────────────────────────────────────
if [ ! -f "$SCRIPT" ]; then
    echo "❌ ${SCRIPT} not found in $(pwd)"
    echo "   Make sure you run this script from the repo root."
    exit 1
fi

# ── Print run info ────────────────────────────────────────────────────────────
echo "============================================================"
echo "  DFS Part A — Transformer Tuning Run"
echo "  Start time : $(date)"
echo "  Script     : ${SCRIPT}"
echo "  Log file   : ${LOG_FILE}"
echo "  Device     : $(python -c "import torch; print('CUDA' if torch.cuda.is_available() else ('MPS' if torch.backends.mps.is_available() else 'CPU'))")"
echo "  PyTorch    : $(python -c "import torch; print(torch.__version__)")"
echo "  Python     : $(python --version)"
echo "============================================================"

# ── Run the script, tee output to both terminal and log file ──────────────────
python "$SCRIPT" 2>&1 | tee "$LOG_FILE"

# ── Move generated outputs to outputs directory ────────────────────────────────
echo ""
echo "📦 Moving outputs..."
[ -f "tuning_results_mbpp.csv" ]  && mv tuning_results_mbpp.csv  "${OUTPUTS_DIR}/" && echo "   ✅ tuning_results_mbpp.csv  → ${OUTPUTS_DIR}/"
[ -f "transformer_mbpp.pt" ]      && mv transformer_mbpp.pt      "${OUTPUTS_DIR}/" && echo "   ✅ transformer_mbpp.pt      → ${OUTPUTS_DIR}/"
[ -f "optuna_mbpp.db" ]           && mv optuna_mbpp.db           "${OUTPUTS_DIR}/" && echo "   ✅ optuna_mbpp.db           → ${OUTPUTS_DIR}/"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅ Run complete: $(date)"
echo "  Log saved to : ${LOG_FILE}"
echo "============================================================"
