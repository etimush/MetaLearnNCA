"""
Evaluate a single training run from its latest checkpoint, also while it is
still training (quick look; no CSV, no tables).

Uses the run's latest.pt (written every 250 epochs; overwritten each time) or
its final model.pt, rebuilds the variant from the run's config.json, and scores
it with the same protocol as eval_ablation.py (same tasks for a given
evaluation seed; K and N at the top of the run's training range).

Example (NCA-ProtoNet seed 0, latest checkpoint, OOD datasets, one test task):
  python ablations/eval_checkpoint.py --run_dir ablation_runs_bf16/nca_protonet/seed0 \
      --shots 5 --eval_seeds 43
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_lib as L  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="e.g. ablation_runs_bf16/nca_protonet/seed0")
    p.add_argument("--ckpt", choices=["latest", "final"], default="latest",
                   help="latest.pt (training checkpoint) or model.pt (finished run)")
    p.add_argument("--datasets", nargs="+", default=L.OOD_DATASETS,
                   help="default: the five OOD datasets; add omniglot for in-distribution")
    p.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--eval_seeds", nargs="+", type=int, default=[43])
    p.add_argument("--max_test_per_class", type=int, default=None, help="subsample the test set (speed)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    variant = cfg["variant"]
    learner = L.build_learner(variant, cfg["opts"])
    if args.ckpt == "latest":
        path = os.path.join(args.run_dir, "latest.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state, epoch = ck["model"], ck["epoch"]
    else:
        path = os.path.join(args.run_dir, "model.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state, epoch = ck["state_dict"], ck.get("epoch")
    learner.load_state_dict(state)
    learner.to(args.device).eval()
    K, N = cfg["k_repeats"] + 2, cfg["n_steps"] + 3     # top of the training range, as in eval_ablation
    print(f"{variant} seed {cfg['seed']}: {os.path.basename(path)} from epoch {epoch} of {cfg['epochs']} "
          f"| K={K}, N={N} | eval seeds {args.eval_seeds}"
          + (f" | <= {args.max_test_per_class} test images/class" if args.max_test_per_class else ""))

    rows = {}
    for d in args.datasets:
        images, targets = L.load_eval_dataset(d, args.data_root)
        images = images.to(args.device)
        for shot in args.shots:
            accs = [L.evaluate_task(learner, images, targets, shot, es, K, N, args.device,
                                    batch_size=args.batch_size,
                                    max_test_per_class=args.max_test_per_class)["acc"][K]
                    for es in args.eval_seeds]
            rows[(d, shot)] = float(np.mean(accs))
            print(f"  {L.DATASET_LABELS.get(d, d):<14} {shot:>2}-shot: {rows[(d, shot)]:6.2f}%", flush=True)

    print("\n| Dataset | " + " | ".join(f"{s}-shot" for s in args.shots) + " |")
    print("|---|" + "---|" * len(args.shots))
    for d in args.datasets:
        print(f"| {L.DATASET_LABELS.get(d, d)} | " + " | ".join(f"{rows[(d, s)]:.2f}" for s in args.shots) + " |")
    ood = [d for d in args.datasets if d in L.OOD_DATASETS]
    if len(ood) > 1:
        print("| OOD mean | " + " | ".join(f"{np.mean([rows[(d, s)] for d in ood]):.2f}" for s in args.shots) + " |")


if __name__ == "__main__":
    main()
