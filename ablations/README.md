# Component ablations (ICLR 2027 revision)

Scripts for the ablation table requested by all three reviewers
(R1 Q2/W4, R2 Q1/W1, R3 Q3/W1–W2). Each variant changes exactly one component
of MetaLearnNCA; everything else is the training protocol of
`ML-NCA-Faster-trainign.py`, the script that produced `checkpoints/`.

| File | Purpose |
| --- | --- |
| `ablation_lib.py` | Models, variant definitions, data, the shared adapt/episode/predict code, statistics |
| `train_ablation.py` | Meta-train one variant for one seed |
| `eval_ablation.py` | Evaluate every run on the Table-1 protocol and write the tables (Markdown, LaTeX, CSV) |
| `eval_recurrence.py` | Test-time recurrence ablation and stability analysis on trained models (no retraining) |
| `run_ablations.sh` | Train all variants × seeds (multi-GPU), then evaluate |
| `make_table1.py` | Multi-seed Table 1: combines `baseline_results/results_raw.csv` (from `Train-Canonical-Models.py --seeds ...`) with the MetaLearnNCA rows of `ablation_results/results_raw.csv`; mean ± SD over training seeds, and a win/loss/n.s. call against the best baseline per row |

## Variants

| Variant (`--variant`) | Change relative to the full model | Question | Asked by | Tier |
| --- | --- | --- | --- | --- |
| `full` | none (retrained under this script with the same seeds) | reference | – | main |
| `global_program` | S is a 24-D vector: the Meta-NCA output is averaged over space (and the support batch) and tiled across the grid; the initial program is a random 24-D vector, also tiled | Does spatial structure matter? | R1, R3 | main |
| `no_mask` | the foreground mask is removed from the residual E_pixel, the query loss and the read-out (all 784 pixels used everywhere) | How much comes from masking? | R1, R3 | main |
| `grad_on_s` | Meta-NCA removed. S starts at 0 and takes K SGD steps on the masked support cross-entropy (CAVIA). The step size is meta-learned per channel (log-parameterised, initialised at 10, the largest stable value on a released checkpoint), and the meta-gradient is second-order. Uses test-time gradients by construction | Does the learned updater beat plain gradients? | R1, R2, R3 | main |
| `nca_protonet` | Meta-NCA and program adaptation removed. The same Active-NCA (2N steps) embeds each image as the masked mean of its 32 dynamic channels, and classification is by squared Euclidean distance to class prototypes. The 24 program channels receive one learned, task-independent spatial program | Is the gain from the NCA architecture or from adaptation? | R2, R3 | main |
| `no_feedback` | Meta-NCA reads Y_spatial ⊙ M (labelled strokes) instead of the residual Y − softmax(logits), so Active-NCA predictions never reach the updater (open loop) | Is adaptation actually error-driven? | R2, R3 | appendix |
| `noncellular_updater` | Meta-NCA perception reduced to the identity kernel (1×1, no neighbourhood communication); hidden width 686 instead of 256 so the parameter count is unchanged | Does the updater need local diffusion of errors? | R1, R3 | appendix |
| `short_unroll` | base N = 8 instead of 16 (train N′ ∈ [8, 11], test N = 11) | Is long recurrent unrolling necessary? | R3 | appendix |
| `img_conditioned` | Meta-NCA reads [E, support image] instead of E (mask kept); its hidden width is reduced to 251 so the parameter count is unchanged (118,271). This is the mask-free fix used on CIFAR | Does conditioning the updater on the image change the Omniglot results? | – | appendix |

Test-time recurrence (R3's "long recurrent unrolling") also needs no retraining:
`eval_recurrence.py` sweeps K up to 50 and N from 4 to 32 on existing checkpoints.

### Fixed protocol (identical for every variant)

* Omniglot background set, 10-way 1-shot 5-query episodes, 2 tasks × 8
  accumulation passes per step, 4000 epochs, AdamW (lr 2e-3, cosine to 5e-4,
  wd 1e-5), grad-norm clip 1.0, fp16 autocast, jitter N′ ∈ [N, N+3],
  K′ ∈ [K, K+2], W2 initialised to default/5, hidden width 256.
* Evaluation: Table-1 protocol of `ML-NCA-Testing.py` (10 classes, k-shot
  support, *all* remaining images of those classes as queries, K = 5, N = 19,
  evaluation seeds 43–52). Task splits come from `RandomState(eval_seed)`, so
  every model sees the same tasks, and they match the tasks used for Table 1.
* Each training seed's score is the mean over the 10 tasks. Tables report
  mean ± SD across training seeds, and Δ to the full model with a Welch 95% CI.

### Deliberate differences from the original training script

1. **bf16 instead of fp16** (on GPUs that support it, i.e. Ampere or newer). The NCA
   state can grow beyond fp16's range (65,504); the resulting NaN losses made
   the gradient scaler skip updates and, in the original script, ended runs at
   the first NaN. bf16 has the range of fp32 at the same speed. A resumed run
   keeps the precision it was started with. Report "bf16 mixed precision".
2. **Guards against non-finite values, all recorded in `status.json`:**
   * a pass with a non-finite loss is skipped (`skipped_passes`); the other
     passes of that update still count;
   * an update with non-finite gradients is skipped (`skipped_steps`);
   * if 10 updates in a row are skipped, or the weights become non-finite, the
     run rolls back to its last good state (kept in memory every 25 epochs) and
     continues with a different, deterministic random stream (`rewinds`);
   * after 10 roll-backs the run is marked `diverged`. The table shows
     `ok/total` runs per variant.
   The original script instead stopped at the first NaN loss and saved nothing.
   Report the number of skipped passes, skipped updates and roll-backs per
   variant (or state that there were none).
3. The logged loss is averaged over all accumulation passes.
4. Runs can be resumed (`--resume`, checkpoint every 250 epochs).
5. `grad_on_s` runs without `torch.compile`, which does not support double
   backward, and in fp32 on GPUs without bf16 (fp16 underflows in the inner
   gradient).

## How to run

One command for everything (8 variants × 5 seeds, then evaluation and tables); safe to re-run, it
resumes interrupted runs and skips finished ones:

```bash
VARIANTS=all SEEDS="0 1 2 3 4" GPUS="0" bash ablations/run_ablations.sh
```

A run that hits a CUDA out-of-memory error (most likely `grad_on_s`) automatically halves its
micro-batch and doubles the accumulation passes, so each optimiser step still uses 16 tasks; the
change is recorded in its `status.json`. Step by step:

```bash
# 0. CPU smoke test of every variant (synthetic data, ~1 min)
for v in full global_program no_mask grad_on_s nca_protonet no_feedback noncellular_updater short_unroll; do
  python ablations/train_ablation.py --variant $v --seed 0 --smoke --fake_data --out_dir /tmp/smoke_runs --overwrite
done
python ablations/eval_ablation.py --runs_dir /tmp/smoke_runs --datasets fake --shots 1 5 --eval_seeds 43 44 \
    --out_dir /tmp/smoke_results --max_test_per_class 10

# 1. Main table: 5 variants × 3 seeds = 15 runs (add GPUs / JOBS_PER_GPU to parallelise)
GPUS="0" SEEDS="0 1 2" bash ablations/run_ablations.sh

# 2. Optional appendix variants
VARIANTS="no_feedback noncellular_updater short_unroll" bash ablations/run_ablations.sh

# 3. Evaluate (resumable) + tables; add original checkpoints as an extra reference row
python ablations/eval_ablation.py --runs_dir ablation_runs --original_ckpts checkpoints --original_seeds 0 1 6

# 4. Multi-seed Table 1: baselines over 5 training seeds, then combine with the MetaLearnNCA rows of step 3
python Train-Canonical-Models.py --seeds 0 1 2 3 4 --epochs 5000 --meta_batch 32 --shots 1 5 10 --test_runs 10
python ablations/make_table1.py --baseline_csv baseline_results/results_raw.csv --nca_csv ablation_results/results_raw.csv

# 5. Test-time recurrence / stability (original checkpoints, no retraining)
python ablations/eval_recurrence.py --original_ckpts checkpoints --original_seeds 0 1 6 --shot 10
```

Outputs:
* `ablation_runs/<variant>/seed<s>/`: `config.json`, `train_log.csv`, `status.json`, `model.pt`
  (`full` also exports `active_nca.pth`/`meta_nca.pth`, loadable by the original scripts).
* `ablation_results/`: `results_raw.csv` (one row per task), `summary.csv`, `ablation_tables.md`, `ablation_tables.tex`.
* `recurrence_results/`: `recurrence_k.csv`, `recurrence_n.csv`, `recurrence_k.png`, `recurrence_n.png`.

Useful flags: `--max_test_per_class 100` for a fast first pass (use a separate
`--out_dir`), `--trajectory` to score k = 0..5, `--first_order` or
`--micro_batch 1 --accum 16` if `grad_on_s` runs out of memory (the second
keeps 16 tasks per step; say which one you used in the paper).

## What has been verified (CPU, before any GPU run)

* **Equivalence with the original code.** `full` loaded from `checkpoints/*seed55.pth`
  matches the classes in `ML-NCA-Testing.py` with the same seed up to float32 rounding (max abs.
  difference 2e-6 in the adapted program, 2e-5 in the logits; bit-identical before the speed-up
  below). The task splits are identical for evaluation seeds 43, 47 and 52.
* **Table-1 sanity check** (seed 55, 5-shot, 3 tasks, ≤40 test images/class):
  Omniglot 95.6% (Table 1: 95.20) and MNIST 83.5% (Table 1: 84.12).
* **Every variant** trains, checkpoints, resumes and evaluates end to end
  (`--smoke --fake_data`). Every parameter receives a finite, non-zero gradient,
  including the meta-learned step size and the learned programs. The
  second-order meta-gradient of `grad_on_s` differs from the first-order one (it
  is not silently dropped).
* **`grad_on_s` at test time** accumulates dL/dS over chunks of 25 support images
  to bound memory. This is exact (max difference 1e-13 in float64), because S is
  shared by all support images.
* **`grad_on_s` inner step size**: on a released checkpoint, lr = 10 lowers the
  support loss monotonically (2.89 → 0.99 in 3 steps) and lr = 100 diverges.
  Hence the initial value of 10; it is meta-learned from there.

Not verified here: actual accuracies of the retrained variants. There is no GPU
in this environment, and USPS, KMNIST and MedMNIST could not be downloaded.

## Speed

* Each NCA step computes W1 · P([static, state]); because the perception P is depthwise and W1 is
  linear, the static part (image + program, or the error signal) is computed once per pass instead
  of at every step (`BaseNCA.PRECOMPUTE_STATIC`). This is exact: losses identical and
  meta-gradients equal to 1e-8 or better in float64 for every variant, and accuracies on the
  released checkpoint unchanged. Measured on CPU: 2.1x faster for `full`, 1.8x for `grad_on_s`. Runs
  started before this change resume normally (checkpoints are unchanged).
* If `nvidia-smi` shows the GPU well below full utilisation, run several jobs per GPU
  (`JOBS_PER_GPU=2` or `3`; a full-model run needs about 4 GB). This does not change training.
* `EXTRA="--micro_batch 4 --accum 4"` (same 16 tasks per update, fewer, larger passes) is usually
  faster on a GPU, but the step-length jitter is then drawn 4 instead of 8 times per update. That is
  a small protocol change, so use it for all runs of the study or for none, and report it.

## Decide what each outcome means before you see the results

The revision plan asks you to commit to interpretations in advance. Suggested
criterion: a component "matters" if the full model beats the variant on the
OOD mean by more than the Welch 95% CI of the difference. With 3 seeds that
CI is wide, so also report the per-dataset numbers rather than only the verdict.

| Variant | If ≈ full (within CI) | If clearly below full | If above full |
| --- | --- | --- | --- |
| Global program | The spatial layout of S is not what helps. Drop "spatially distributed conditioning" as a contribution (the plan says to drop the spatial claim) | Supports the spatial-program claim (R3 strength 2) | Spatial program is harmful; rethink the framing |
| No mask | Masking is a minor implementation detail | Masking is a major contributor and should be presented as part of the method. It is also a domain prior (dark background, sparse strokes), relevant to the MedMNIST failure | Mask hurts; check MedMNIST especially |
| Gradient on S | The learned updater adds no accuracy over CAVIA-style gradients. Its only advantage is no test-time backprop; report cost honestly | The learned updater beats plain gradients: the strongest answer to R2's "incremental" | Main claim weakened: say so, and keep the result |
| NCA-ProtoNet | OOD gains come mainly from the NCA architecture (R3 W2 confound), not from adaptation | Gains need the Meta-NCA adaptation, not just the backbone | Adaptation hurts relative to prototypes on this backbone |
| No feedback | Adaptation is not error-driven. Drop "iterative, error-driven" wording | Supports the closed-loop, error-driven claim | – |
| Non-cellular updater | The updater does not need local diffusion; drop "diffusing errors across neighbourhoods" | Supports the cellular updater | – |
| Short unroll | Long unrolls are not needed (cheaper method) | Long recurrence matters | – |

Also compare NCA-ProtoNet with the Conv4 ProtoNet from Table 1: if NCA-ProtoNet ≫ Conv4
ProtoNet on OOD datasets, the NCA backbone itself transfers better. That is a
legitimate finding, but it is about architecture, not meta-learning.

## Compute estimate

A full-model run costs about 2.3 TFLOP per micro-batch (forward + backward),
about 18 TFLOP per epoch and about 75 PFLOP over 4000 epochs. On one modern GPU
that is roughly 1–3 h. `nca_protonet` is about 4× cheaper. `grad_on_s`
(second order) is about 2–3× more expensive and needs much more memory.
Evaluating one run on all 6 datasets × 3 shots × 10 tasks touches about 1.4 M
query images × 38 NCA steps.
