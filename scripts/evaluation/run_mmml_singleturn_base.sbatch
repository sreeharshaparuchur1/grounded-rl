#!/bin/bash
# =============================================================================
# run_mmml_singleturn_base.sbatch
# Single-turn evaluation using base Qwen2.5-VL-3B-Instruct (no ViGoRL weights)
# on MMML baseline tasks (PSC Bridge-2)
#
# Uses sdpa attention — no flash-attn build required.
#
# Submit (from repo root):
#   sbatch scripts/evaluation/run_mmml_singleturn_base.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/mmml_base_eval_<JOBID>.out
# =============================================================================

#SBATCH --job-name=qwen_base_mmml
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260044p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mmml_base_eval_%j.out
#SBATCH --error=logs/mmml_base_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUREMAIL@andrew.cmu.edu   # <-- update or remove this line

# =============================================================================
# Configuration
# =============================================================================
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"   # base Qwen model, no ViGoRL weights

CONDA_ENV="grounded-rl"

REPO_ROOT="${SLURM_SUBMIT_DIR}"

TASKS_TXT="${REPO_ROOT}/scripts/evaluation/mmml_baseline_tasks.txt"
TASKS_JSON="${REPO_ROOT}/scripts/evaluation/mmml_baseline_tasks.json"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs"
RESULTS_JSON="${LOG_DIR}/mmml_base_results_${TIMESTAMP}.json"
export RESULTS_JSON

MAX_NEW_TOKENS=1024
TEMPERATURE=0.5
REPETITION_PENALTY=1.05

# =============================================================================
# Environment / conda activation (before set -e)
# =============================================================================
if [ -f "/jet/home/${USER}/.bashrc" ]; then
    # shellcheck disable=SC1090
    source "/jet/home/${USER}/.bashrc" 2>/dev/null || true
fi

if command -v conda &>/dev/null; then
    conda activate "${CONDA_ENV}" 2>/dev/null \
        || echo "[WARN] Could not activate conda env '${CONDA_ENV}' — using current Python"
else
    echo "[INFO] conda not on PATH — relying on module-loaded Python"
fi

# PSC modules
module load cuda/12.6.1

# HuggingFace cache — point to shared Ocean storage
export HF_HOME=/ocean/projects/cis260044p/sparuchu/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# =============================================================================
# Strict mode on — after conda
# =============================================================================
set -euo pipefail

mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Job name       : ${SLURM_JOB_NAME}"
echo "  Node           : ${SLURM_NODELIST}"
echo "  Submit dir     : ${SLURM_SUBMIT_DIR}"
echo "  Start time     : $(date)"
echo "  Repo root      : ${REPO_ROOT}"
echo "  Model          : ${MODEL}  (base — no ViGoRL weights)"
echo "  Attention      : sdpa  (no flash-attn required)"
echo "  Tasks txt      : ${TASKS_TXT}"
echo "  Tasks json     : ${TASKS_JSON}"
echo "  Results json   : ${RESULTS_JSON}"
echo "============================================================"

# --- Verify GPU ---
echo ""
echo "[INFO] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# --- Verify Python / torch ---
echo "[INFO] Python : $(python --version 2>&1)"
python - <<'PYEOF'
import torch
print(f"[INFO] PyTorch       : {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU           : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM total    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PYEOF
echo ""

# =============================================================================
# Step 1: Parse txt → JSON
# (re-runs every time so new tasks in the txt are picked up automatically)
# =============================================================================
echo "------------------------------------------------------------"
echo "[STEP 1] Parsing task file → JSON"
echo "------------------------------------------------------------"

cd "${REPO_ROOT}"

python scripts/evaluation/parse_mmml_tasks.py \
    --input  "${TASKS_TXT}" \
    --output "${TASKS_JSON}"

echo "[INFO] Task JSON ready: ${TASKS_JSON}"
echo ""

# =============================================================================
# Step 2: Run single-turn base model evaluation
# =============================================================================
echo "------------------------------------------------------------"
echo "[STEP 2] Running single-turn base Qwen evaluation"
echo "------------------------------------------------------------"

set +e
python scripts/evaluation/eval_mmml_singleturn_base.py \
    --model              "${MODEL}" \
    --tasks              "${TASKS_JSON}" \
    --output             "${RESULTS_JSON}" \
    --max_new_tokens     "${MAX_NEW_TOKENS}" \
    --temperature        "${TEMPERATURE}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --device             cuda
EXIT_CODE=$?
set -e

# =============================================================================
# Step 3: Print final summary
# =============================================================================
echo ""
echo "============================================================"
echo "  Job finished at : $(date)"
echo "  Exit code       : ${EXIT_CODE}"
echo "============================================================"

if [ -f "${RESULTS_JSON}" ]; then
    echo ""
    echo "[RESULTS SUMMARY]"
    python - <<'PYEOF'
import json, os, sys

path = os.environ.get("RESULTS_JSON", "")
if not path or not os.path.exists(path):
    print("  [WARN] Results file not found:", path)
    sys.exit(0)

data = json.loads(open(path).read())
s = data["summary"]

print(f"  Model               : {s['model']}")
print(f"  Total tasks         : {s['total_tasks']}")
print(f"  Successful (non-empty response) : {s['successful']}  ({s['success_rate']*100:.1f}%)")
print(f"  With <think> tags   : {s['with_thinking_tags']}")
print(f"  With <answer> tags  : {s['with_answer_tags']}")
print()
print("  Per-task breakdown:")
print(f"  {'#':>3}  {'OK?':<4}  {'time':>6}  response preview")
print(f"  {'-'*3}  {'-'*4}  {'-'*6}  {'-'*60}")
for r in data["results"]:
    status = "OK " if r["success"] else "---"
    preview = (r["raw_output"] or "(none)").replace("\n", " ")[:60]
    t_sec = r.get("generation_time_s", "?")
    print(f"  {r['task_id']:>3}  {status:<4}  {str(t_sec):>5}s  {preview}")
PYEOF
else
    echo "[WARN] Results file not found — evaluation may have failed."
fi

echo ""
echo "[INFO] Full results  : ${RESULTS_JSON}"
echo "[INFO] Stdout log    : ${LOG_DIR}/mmml_base_eval_${SLURM_JOB_ID}.out"
echo "[INFO] Stderr log    : ${LOG_DIR}/mmml_base_eval_${SLURM_JOB_ID}.err"
echo "============================================================"

exit "${EXIT_CODE}"
