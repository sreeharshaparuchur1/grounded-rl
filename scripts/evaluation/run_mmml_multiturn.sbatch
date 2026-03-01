#!/bin/bash
# =============================================================================
# run_mmml_multiturn.sbatch
# Multi-turn ViGoRL 3B evaluation on MMML baseline tasks (PSC Bridge-2)
#
# Submit (from repo root):
#   sbatch scripts/evaluation/run_mmml_multiturn.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/mmml_multiturn_eval_<JOBID>.out
# =============================================================================

#SBATCH --job-name=vigorl_mmml_mt
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260044p
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mmml_multiturn_eval_%j.out
#SBATCH --error=logs/mmml_multiturn_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUREMAIL@andrew.cmu.edu   # <-- update or remove this line

# =============================================================================
# Configuration — edit these as needed
# =============================================================================
MODEL="gsarch/ViGoRL-Multiturn-3b-Web-Grounding"
                                          # alternatives:
                                          #   gsarch/ViGoRL-Multiturn-3b-Visual-Search
                                          #   gsarch/ViGoRL-Multiturn-7b-Visual-Search

CONDA_ENV="grounded-rl"

REPO_ROOT="${SLURM_SUBMIT_DIR}"

TASKS_TXT="${REPO_ROOT}/scripts/evaluation/mmml_baseline_tasks.txt"
TASKS_JSON="${REPO_ROOT}/scripts/evaluation/mmml_baseline_tasks.json"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/logs"
RESULTS_JSON="${LOG_DIR}/mmml_multiturn_results_${TIMESTAMP}.json"
export RESULTS_JSON

CROP_DIR="${REPO_ROOT}/data/demo/crops_${TIMESTAMP}"

MAX_TURNS=5
MAX_NEW_TOKENS=512
TEMPERATURE=0.5
REPETITION_PENALTY=1.05
CROP_SIZE=512
CROP_OFFSET=100
DRAW_DOT="--draw_dot"   # set to "" to disable the red dot on crops

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
mkdir -p "${CROP_DIR}"

echo "============================================================"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Job name       : ${SLURM_JOB_NAME}"
echo "  Node           : ${SLURM_NODELIST}"
echo "  Submit dir     : ${SLURM_SUBMIT_DIR}"
echo "  Start time     : $(date)"
echo "  Repo root      : ${REPO_ROOT}"
echo "  Model          : ${MODEL}"
echo "  Tasks txt      : ${TASKS_TXT}"
echo "  Tasks json     : ${TASKS_JSON}"
echo "  Results json   : ${RESULTS_JSON}"
echo "  Crop dir       : ${CROP_DIR}"
echo "  Max turns      : ${MAX_TURNS}"
echo "  Crop size      : ${CROP_SIZE}  offset: ${CROP_OFFSET}"
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
# (always re-runs so new tasks added to the txt are automatically picked up)
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
# Step 2: Run multi-turn evaluation
# =============================================================================
echo "------------------------------------------------------------"
echo "[STEP 2] Running multi-turn ViGoRL evaluation"
echo "------------------------------------------------------------"

set +e
python scripts/evaluation/eval_mmml_multiturn.py \
    --model              "${MODEL}" \
    --tasks              "${TASKS_JSON}" \
    --output             "${RESULTS_JSON}" \
    --max_turns          "${MAX_TURNS}" \
    --max_new_tokens     "${MAX_NEW_TOKENS}" \
    --temperature        "${TEMPERATURE}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --crop_size          "${CROP_SIZE}" \
    --crop_offset        "${CROP_OFFSET}" \
    --crop_dir           "${CROP_DIR}" \
    ${DRAW_DOT} \
    --device cuda
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

print(f"  Model           : {s['model']}")
print(f"  Total tasks     : {s['total_tasks']}")
print(f"  Successful      : {s['successful']}  ({s['success_rate']*100:.1f}%)")
print(f"  Forced answers  : {s['forced_answers']}")
print(f"  With reasoning  : {s['with_thinking']}  ({s['thinking_rate']*100:.1f}%)")
print(f"  With answer     : {s['with_answer']}  ({s['answer_rate']*100:.1f}%)")
print(f"  Avg turns       : {s['avg_turns']}")
print(f"  Avg tool calls  : {s['avg_tool_calls']}")
print()
print("  Per-task breakdown:")
print(f"  {'#':>3}  {'OK?':<4}  {'turns':>5}  {'calls':>5}  {'forced':>6}  {'think':>5}  answer")
print(f"  {'-'*3}  {'-'*4}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*40}")
for r in data["results"]:
    status = "OK " if r["success"] else "---"
    ans    = (r["answer"] or "(none)")[:50]
    tlen   = len(r["thinking"]) if r["thinking"] else 0
    forced = "yes" if r["forced_answer"] else "no"
    has_t  = "yes" if r["has_thinking"] else "no"
    print(f"  {r['task_id']:>3}  {status:<4}  {r['n_turns']:>5}  {r['n_tool_calls']:>5}  "
          f"{forced:>6}  {has_t:>5}  {ans}  [{r['total_time_s']}s]")
PYEOF
else
    echo "[WARN] Results file not found — evaluation may have failed."
fi

echo ""
echo "[INFO] Full results  : ${RESULTS_JSON}"
echo "[INFO] Crops saved   : ${CROP_DIR}"
echo "[INFO] Stdout log    : ${LOG_DIR}/mmml_multiturn_eval_${SLURM_JOB_ID}.out"
echo "[INFO] Stderr log    : ${LOG_DIR}/mmml_multiturn_eval_${SLURM_JOB_ID}.err"
echo "============================================================"

exit "${EXIT_CODE}"
