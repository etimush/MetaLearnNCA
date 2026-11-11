#!/usr/bin/env bash
# FC100 (CIFAR-100) few-shot experiment: train every method x seed, then evaluate and build the table.
#
# Usage (from the repository root):
#   bash cifar/run_cifar.sh                                   # all 6 methods x 5 seeds, then evaluation
#   METHODS="protonet matchingnet relationnet fomaml" bash cifar/run_cifar.sh     # Conv4 baselines only
#   GPUS="0 1" JOBS_PER_GPU=2 bash cifar/run_cifar.sh         # parallel workers
#
# Environment variables:
#   METHODS       default: metalearnnca nca_protonet protonet matchingnet relationnet fomaml
#   SEEDS         default: 0 1 2 3 4
#   GPUS          default: 0
#   JOBS_PER_GPU  default: 1   (the NCA runs need most of a 12 GB card; Conv4 runs are small)
#   OUT           default: cifar_runs
#   EXTRA         extra flags passed to every train_cifar.py call
#   RUN_EVAL      default: 1   (evaluate + build the table when training is done)
#   QUIET         1 = do not echo progress lines (they are always in $OUT/logs/)
#
# Jobs are ordered seed-major, every job uses --resume (re-running continues interrupted runs and
# skips finished ones), and Ctrl-C stops all workers and their training processes.
set -uo pipefail
cd "$(dirname "$0")/.."

METHODS=${METHODS:-"metalearnnca nca_protonet protonet matchingnet relationnet fomaml"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
GPUS=${GPUS:-0}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
OUT=${OUT:-cifar_runs}
EXTRA=${EXTRA:-}
RUN_EVAL=${RUN_EVAL:-1}
QUIET=${QUIET:-0}
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

jobs=()
for s in $SEEDS; do
  for m in $METHODS; do
    jobs+=("$m $s")
  done
done

read -r -a gpu_arr <<< "$GPUS"
workers=$(( ${#gpu_arr[@]} * JOBS_PER_GPU ))
echo "Launching ${#jobs[@]} training job(s) on $workers worker(s) [GPUs: $GPUS]  (Ctrl-C stops everything)"

worker_pids=()
kill_tree() {
  local kids
  kids=$(pgrep -P "$1" 2>/dev/null)
  kill -TERM "$1" 2>/dev/null
  for k in $kids; do kill_tree "$k"; done
}
stop_all() {
  trap - INT TERM
  echo ""
  echo "Stopping all CIFAR jobs... finished runs are kept; re-run the same command to resume"
  echo "(interrupted runs continue from their last checkpoint, saved every 250 updates)."
  for p in "${worker_pids[@]}"; do kill_tree "$p"; done
  wait 2>/dev/null
  exit 130
}
trap stop_all INT TERM

for w in $(seq 0 $((workers - 1))); do
  gpu=${gpu_arr[$(( w % ${#gpu_arr[@]} ))]}
  (
    for i in "${!jobs[@]}"; do
      if (( i % workers == w )); then
        read -r m s <<< "${jobs[$i]}"
        log="$LOGDIR/${m}_seed${s}.log"
        echo "[worker $w | GPU $gpu] $m seed $s  (full log: $log)"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 python cifar/train_cifar.py \
              --method "$m" --seed "$s" --out_dir "$OUT" --resume $EXTRA 2>&1 \
          | tee "$log" \
          | if [ "$QUIET" = "1" ]; then cat > /dev/null; else
              grep --line-buffered -E "\] up |\[oom\]|\[rewind\]|parameters|Resumed|\[skip\]|\[diverged\]|Finished|Error|Traceback|Killed"; fi
        if [ "${PIPESTATUS[0]}" != "0" ]; then
          echo "FAILED: $m seed $s (see $log)"
        fi
      fi
    done
  ) &
  worker_pids+=($!)
done
wait
echo "Training finished. Status per run:"
for f in "$OUT"/*/seed*/status.json; do
  [ -e "$f" ] && echo "  $f: $(python -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$f")"
done

if [ "$RUN_EVAL" = "1" ]; then
  CUDA_VISIBLE_DEVICES=${gpu_arr[0]} python cifar/eval_cifar.py --runs_dir "$OUT" --out_dir cifar_results
  echo "Table: cifar_results/table.md and cifar_results/table.tex"
fi
