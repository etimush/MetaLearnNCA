# FC100 (CIFAR-100) few-shot meta-learning

Answers R2's "does it generalise to natural images (CIFAR-100, miniImageNet)?" with one
experiment in which MetaLearnNCA and the canonical baselines are trained and tested under
exactly the same conditions.

| File | Purpose |
| --- | --- |
| `cifar_lib.py` | FC100 data, episode samplers, all six models, parameter matching, the shared `REGIME` |
| `train_cifar.py` | Meta-train one method for one seed |
| `eval_cifar.py` | Evaluate every run on the same test episodes and build the table |
| `run_cifar.sh` | All methods × 5 seeds, then evaluation (Ctrl-C safe, resumable) |

## What is the same for every method

**Data and out-of-distribution test.** CIFAR-100 is split by *superclass* (FC100, Oreshkin et al.
2018; assignment taken from the authors' TADAM repository, `datasets/create_dataset_cifar100.py`):

| Split | Superclasses | Classes |
| --- | --- | --- |
| meta-train | fish, flowers, food containers, fruit & vegetables, household electrical devices, household furniture, large man-made outdoor things, large natural outdoor scenes, reptiles, trees, vehicles 1, vehicles 2 | 60 |
| validation | large carnivores, large omnivores & herbivores, non-insect invertebrates, small mammals | 20 |
| test | aquatic mammals, insects, medium mammals, people | 20 |

Test classes therefore come from superclasses never seen in meta-training: this is a genuine
few-shot test on novel, semantically distant categories, unlike a random class split (CIFAR-FS)
or the same-class diagonal of the paper's cross-dataset matrix. All 600 images per class are
used (CIFAR-100 train + test sets). Preprocessing: pixels scaled to [-1, 1]; no augmentation.

**Training regime** (`cifar_lib.REGIME`, identical for all methods):
5-way episodes, 1- or 5-shot (drawn per pass, so one model serves both test shots), 10 query
images per class, 16 tasks per update (2 per pass × 8 passes), 4,000 updates (64,000 episodes),
AdamW with lr 2e-3 cosine-annealed to 5e-4, weight decay 1e-5, gradient-norm clip 1.0, bf16
autocast where supported (fp32 otherwise). Same NaN/roll-back guards and out-of-memory fallback
as the Omniglot ablations.

**Test protocol** (`eval_cifar.py`): 5-way, 1-shot and 5-shot, 15 queries per class, 1,000
episodes per shot from the 20 test classes. Episode *e* depends only on `(episode_seed, shot, e)`,
so every model and seed is scored on exactly the same tasks. Per run: mean over episodes (with
its 95% CI); per method: mean ± SD over the 5 training seeds, and the difference to the best
canonical baseline with a Welch 95% CI. No hyper-parameter is tuned on test data. FOMAML's
test-time inner learning rate is chosen per shot on 200 episodes from the *validation* classes,
from {0.005, 0.01, 0.02, 0.05, 0.1}; the choice is recorded.

**Parameter budget.** Every model is sized to a Conv4-64 ProtoNet on 3×32×32 inputs (113,088):

| Method | Size knob | Parameters |
| --- | --- | --- |
| MetaLearnNCA | hidden width 251 (both NCAs) | 112,950 (−0.12%) |
| MetaLearnNCA, image-conditioned updater | hidden width 245 (both NCAs) | 113,190 (+0.09%) |
| NCA-ProtoNet | hidden width 329 (+ learned 24×32×32 program) | 113,077 (−0.01%) |
| ProtoNet | Conv4 width 64 | 113,088 |
| MatchingNet | Conv4 width 64 | 113,088 |
| RelationNet | Conv4 width 63, relation head 27 | 113,077 (−0.01%) |
| FOMAML | Conv4 width 64 + 5-way linear head | 113,413 (+0.29%) |

## Method details

* **MetaLearnNCA**: the paper's model with RGB input (3 image channels), 5 logit channels and
  **no foreground mask**: the residual `E = Y − softmax(logits)`, the query loss and the
  read-out (mean of the logit channels) use every pixel. Step jitter N′∈[16,19], K′∈[3,5];
  test N=19, K=5.
* **MetaLearnNCA, image-conditioned updater** (`metalearnnca_img`): identical, except that the
  Meta-NCA reads `[E, support image]` instead of `E`. Without a mask, plain MetaLearnNCA
  collapses on FC100 (loss stays at ln 5): once predictions are uniform, `E` is the same flat
  field for every image, the program carries no task information, and nothing restarts learning.
  Giving the updater the image breaks that loop, as the mask did on Omniglot. This is a change
  to the method: report it as a separate row, next to plain MetaLearnNCA. Its Omniglot
  counterpart is the ablation variant `img_conditioned` (mask kept), which shows whether the
  change affects the original results.
* **NCA-ProtoNet**: the same Active-NCA as an embedding (mean of its 32 state channels over all
  pixels), prototypes and squared Euclidean distance, one learned task-independent program.
* **ProtoNet / MatchingNet / RelationNet / FOMAML**: as in `Train-Canonical-Models.py` (Conv4,
  MatchingNet with cosine attention scaled by 100, RelationNet MLP head with MSE, first-order
  MAML with 5 inner steps at lr 0.01 during training), widened to RGB.

## How to run

```bash
# everything: 6 methods x 5 seeds (seed-major), then evaluation and table
bash cifar/run_cifar.sh

# e.g. without the (collapsing) plain MetaLearnNCA:
METHODS="metalearnnca_img nca_protonet protonet matchingnet relationnet fomaml" bash cifar/run_cifar.sh

# evaluation / table only (resumable; finished runs are cached in <run>/eval_fc100.json)
python cifar/eval_cifar.py --runs_dir cifar_runs
```

CIFAR-100 is downloaded by torchvision into `./data` on first use and cached as a uint8 tensor
in `./data/cifar_cache/`. Outputs: `cifar_runs/<method>/seed<s>/` and
`cifar_results/table.md`, `table.tex`, `summary.csv`.

CPU pipeline check (synthetic textures, no download):

```bash
python cifar/train_cifar.py --method metalearnnca --seed 0 --smoke --fake_data --out_dir /tmp/cifar_smoke
python cifar/eval_cifar.py --runs_dir /tmp/cifar_smoke --fake_data --episodes 8 --val_episodes 4 --out_dir /tmp/cifar_smoke_results
```

## Things to know before reading the results

* **Plain MetaLearnNCA without the mask does not learn on FC100.** Observed on seed 0: the loss
  stays at ln 5 ≈ 1.609 with a vanishing gradient norm, the same collapse as the `no_mask`
  ablation on Omniglot. Report it as a scope limitation (the error-driven updater needs a
  spatially informative residual), and show the image-conditioned variant as a separate row.
* **Compute.** By my estimate (not measured on a GPU), a MetaLearnNCA run costs ~1.3× an
  Omniglot run (32×32 RGB vs 28×28 grey), roughly 1.5–2 h on an RTX 4070 Ti; NCA-ProtoNet about
  a third of that; the Conv4 baselines minutes each (FOMAML longer). Evaluation of an NCA run
  takes a few minutes.
* FC100 is much harder than Omniglot or CIFAR-FS (disjoint superclasses, 32×32 natural images);
  expect low absolute accuracies for every method. The comparison between methods is what matters.
