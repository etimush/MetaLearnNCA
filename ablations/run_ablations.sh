#!/usr/bin/env bash
# Train all ablation runs, then evaluate them and build the tables.
#
# Usage (from the repository root):
#   bash ablations/run_ablations.sh                                  # main table, 1 GPU
#   GPUS="0 1 2" bash ablations/run_ablations.sh                     # spread over 3 GPUs
#   GPUS="0" JOBS_PER_GPU=3 bash ablations/run_ablations.sh          # 3 concurrent runs on one GPU
#   VARIANTS="no_feedback noncellular_updater short_unroll" bash ablations/run_ablations.sh   # appendix
#
# Environment variables:
#   VARIANTS      default: full global_program no_mask grad_on_s nca_protonet
#   SEEDS         default: 0 1 2
#   GPUS          default: 0           (space separated CUDA device ids)
#   JOBS_PER_GPU  default: 1           (a full-model run needs ~4 GB; grad_on_s needs much more)
#   OUT           default: ablation_runs
#   EXTRA         extra flags passed to every train_ablation.py call
#   RUN_EVAL      default: 1           (evaluate + build tables when training is done)
#   ORIG_SEEDS    original checkpoints to add as a reference row, e.g. "0 1 6" (default: none)
#
# Jobs are ordered seed-major (all variants for seed 0 first), so a partial
# sweep already gives one seed per variant. Every job uses --resume, so
# re-running this script continues interrupted runs and skips finished ones.
set -uo pipefail
cd "$(dirname "$0")/.."

VARIANTS=${VARIANTS:-"full global_program no_mask grad_on_s nca_protonet"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-0}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
OUT=${OUT:-ablation_runs}
EXTRA=${EXTRA:-}
RUN_EVAL=${RUN_EVAL:-1}
ORIG_SEEDS=${ORIG_SEEDS:-}
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

jobs=()
for s in $SEEDS; do
  for v in $VARIANTS; do
    jobs+=("$v $s")
  done
done

read -r -a gpu_arr <<< "$GPUS"
workers=$(( ${#gpu_arr[@]} * JOBS_PER_GPU ))
echo "Launching ${#jobs[@]} training job(s) on $workers worker(s) [GPUs: $GPUS]"

for w in $(seq 0 $((workers - 1))); do
  gpu=${gpu_arr[$(( w % ${#gpu_arr[@]} ))]}
  (
    for i in "${!jobs[@]}"; do
      if (( i % workers == w )); then
        read -r v s <<< "${jobs[$i]}"
        echo "[worker $w | GPU $gpu] $v seed $s"
        # shellcheck disable=SC2086
        if ! CUDA_VISIBLE_DEVICES=$gpu python ablations/train_ablation.py \
              --variant "$v" --seed "$s" --out_dir "$OUT" --resume $EXTRA \
              > "$LOGDIR/${v}_seed${s}.log" 2>&1; then
          echo "FAILED: $v seed $s (see $LOGDIR/${v}_seed${s}.log)"
        fi
      fi
    done
  ) &
done
wait
echo "Training finished. Status per run:"
for f in "$OUT"/*/seed*/status.json; do
  [ -e "$f" ] && echo "  $f: $(python -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$f")"
done

if [ "$RUN_EVAL" = "1" ]; then
  orig_args=()
  if [ -n "$ORIG_SEEDS" ]; then
    # shellcheck disable=SC2206
    orig_args=(--original_ckpts checkpoints --original_seeds $ORIG_SEEDS)
  fi
  CUDA_VISIBLE_DEVICES=${gpu_arr[0]} python ablations/eval_ablation.py \
      --runs_dir "$OUT" --out_dir ablation_results "${orig_args[@]}"
fi
