"""
Test-time recurrence ablation and stability analysis (no retraining).

Answers R3 ("is long recurrent unrolling responsible?") and R2 ("are the
updates stable / do they converge?") on already-trained models:

  A. K sweep : run the Active/Meta interaction for k = 1..K_max (default 50,
     training only ever used K' <= 5) and record, for every k,
       - rel_change_k = ||S_k - S_{k-1}||_F / ||S_{k-1}||_F   (does S settle?)
       - ||S_k||_F                                              (does S blow up?)
     and, at the k listed in --k_eval, query accuracy and support accuracy.
  B. N sweep : K fixed at 5, inner steps N in --n_sweep (training used
     N' in [16, 19]); N sets both the 2N Active-NCA steps and the N Meta-NCA steps.

Works with the original checkpoints or with any ProgramLearner run produced by
train_ablation.py (e.g. full, short_unroll).

Examples
--------
  python ablations/eval_recurrence.py --original_ckpts checkpoints --original_seeds 0 1 6
  python ablations/eval_recurrence.py --runs_dir ablation_runs --variant short_unroll --seeds 0 1 2
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_lib as L  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original_ckpts", default=None)
    p.add_argument("--original_seeds", nargs="+", default=None)
    p.add_argument("--runs_dir", default=None)
    p.add_argument("--variant", default="full")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--datasets", nargs="+", default=L.DATASETS)
    p.add_argument("--shot", type=int, default=10)
    p.add_argument("--eval_seeds", nargs="+", type=int, default=[43, 44, 45])
    p.add_argument("--max_test_per_class", type=int, default=200,
                   help="test images per class (None = all; the sweep is expensive)")
    p.add_argument("--k_max", type=int, default=50)
    p.add_argument("--k_eval", nargs="+", type=int,
                   default=[0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50])
    p.add_argument("--n_sweep", nargs="+", type=int, default=[4, 8, 12, 16, 19, 24, 32])
    p.add_argument("--n_sweep_k", type=int, default=5)
    p.add_argument("--skip_k_sweep", action="store_true")
    p.add_argument("--skip_n_sweep", action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="recurrence_results")
    return p.parse_args()


def collect_models(args):
    models = []
    if args.original_ckpts:
        for s, (ap, mp) in L.find_original_checkpoints(args.original_ckpts, args.original_seeds).items():
            models.append((f"original_s{s}", "full", lambda ap=ap, mp=mp: L.load_original_checkpoint(ap, mp, args.device)))
    if args.runs_dir:
        vdir = os.path.join(args.runs_dir, args.variant)
        for sd in sorted(os.listdir(vdir)):
            if args.seeds and int(sd[4:]) not in args.seeds:
                continue
            path = os.path.join(vdir, sd, "model.pt")
            if os.path.exists(path):
                models.append((f"{args.variant}_s{sd[4:]}", args.variant,
                               lambda path=path: L.load_run_checkpoint(path, args.device)[0]))
    return models


@torch.no_grad()
def accuracy(learner, x, y, state, N, bs, device):
    correct = 0
    for i in range(0, x.shape[0], bs):
        scores = learner.predict(x[i:i + bs].to(device), state, N)
        correct += (scores.argmax(1).cpu() == y[i:i + bs]).sum().item()
    return 100.0 * correct / max(1, x.shape[0])


@torch.no_grad()
def k_sweep(learner, images, targets, args, es, N):
    split = L.make_task_split(targets, L.N_WAY, args.shot, es, args.max_test_per_class)
    supp_x = images[split.supp_idx].to(args.device)
    supp_y = torch.as_tensor(split.supp_y, device=args.device)
    test_x, test_y = images[split.test_idx], torch.as_tensor(split.test_y)
    torch.manual_seed(es)
    progs = {}
    learner.adapt(supp_x, supp_y, 1, args.k_max, N, meta_train=False,
                  callback=lambda k, S: progs.__setitem__(k, S.float().cpu()))
    rows = []
    for k in range(args.k_max + 1):
        S = progs[k]
        rel = float("nan") if k == 0 else \
            float((S - progs[k - 1]).norm() / progs[k - 1].norm().clamp(min=1e-12))
        row = dict(k=k, rel_change=rel, program_norm=float(S.norm()), test_acc="", support_acc="")
        if k in args.k_eval:
            Sd = S.to(args.device)
            row["test_acc"] = accuracy(learner, test_x, test_y, Sd, N, args.batch_size, args.device)
            row["support_acc"] = accuracy(learner, supp_x.cpu(), supp_y.cpu(), Sd, N, args.batch_size, args.device)
        rows.append(row)
    return rows


def n_sweep(learner, images, targets, args, es):
    rows = []
    for N in args.n_sweep:
        res = L.evaluate_task(learner, images, targets, args.shot, es, args.n_sweep_k, N, args.device,
                              batch_size=args.batch_size, max_test_per_class=args.max_test_per_class)
        rows.append(dict(N=N, test_acc=res["acc"][args.n_sweep_k], adapt_time_s=res["adapt_time_s"]))
    return rows


def ci_curve(per_model):
    """per_model: {model: {x: [values over tasks]}} -> xs, mean, ci (across models)."""
    xs = sorted({x for d in per_model.values() for x in d})
    mean, ci = [], []
    for x in xs:
        vals = [np.mean(d[x]) for d in per_model.values() if x in d]
        m, _, c, _ = L.mean_sd_ci(vals)
        mean.append(m); ci.append(c)
    return np.array(xs), np.array(mean), np.array(ci)


def family(model_name):
    """'original_s55' -> 'original', 'short_unroll_s0' -> 'short_unroll'."""
    return model_name.rsplit("_s", 1)[0]


def family_spec(fam):
    return L.VARIANTS["full" if fam == "original" else fam]


def plot(out_dir, k_rows, n_rows, datasets, shot):
    """One curve per (model family, dataset); bands = 95% CI across checkpoints of that family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styles = ["-", "--", ":", "-."]
    fams = sorted({family(r["model"]) for r in k_rows + n_rows})
    many = len(fams) > 1

    def label(d, fam):
        return f"{L.DATASET_LABELS.get(d, d)} [{fam}]" if many else L.DATASET_LABELS.get(d, d)

    if k_rows:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for fi, fam in enumerate(fams):
            for di, d in enumerate(datasets):
                acc, rel = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
                for r in k_rows:
                    if r["dataset"] != d or family(r["model"]) != fam:
                        continue
                    if r["test_acc"] != "":
                        acc[r["model"]][r["k"]].append(r["test_acc"])
                    if r["k"] > 0:
                        rel[r["model"]][r["k"]].append(r["rel_change"])
                for ax, src in ((axes[0], acc), (axes[1], rel)):
                    if not src:
                        continue
                    xs, m, c = ci_curve(src)
                    ax.plot(xs, m, styles[fi % 4], color=colors[di % len(colors)],
                            marker="o" if ax is axes[0] else None, ms=3, label=label(d, fam))
                    ax.fill_between(xs, m - c, m + c, color=colors[di % len(colors)], alpha=0.12)
        for spec in {family_spec(f) for f in fams}:
            axes[0].axvspan(spec.k_repeats, spec.k_repeats + 2, color="grey", alpha=0.12)
        axes[0].set(xlabel="interactions k (shaded: training range K')", ylabel="test accuracy (%)",
                    ylim=(0, 100), title=f"Accuracy vs. k ({shot}-shot)")
        axes[1].set(xlabel="interactions k", ylabel=r"$\|S_k-S_{k-1}\| / \|S_{k-1}\|$",
                    yscale="log", title="Per-step program change")
        for ax in axes:
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "recurrence_k.png"), dpi=200)
        plt.close(fig)
    if n_rows:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for fi, fam in enumerate(fams):
            for di, d in enumerate(datasets):
                src = defaultdict(lambda: defaultdict(list))
                for r in n_rows:
                    if r["dataset"] == d and family(r["model"]) == fam:
                        src[r["model"]][r["N"]].append(r["test_acc"])
                if src:
                    xs, m, c = ci_curve(src)
                    ax.plot(xs, m, styles[fi % 4], color=colors[di % len(colors)], marker="o",
                            label=label(d, fam))
                    ax.fill_between(xs, m - c, m + c, color=colors[di % len(colors)], alpha=0.12)
        for spec in {family_spec(f) for f in fams}:
            ax.axvspan(spec.n_steps, spec.n_steps + 3, color="grey", alpha=0.12)
        ax.set(xlabel="test-time inner steps N (2N Active + N Meta per interaction; shaded: training N')",
               ylabel="test accuracy (%)", ylim=(0, 100), title=f"Accuracy vs. N (K=5, {shot}-shot)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "recurrence_n.png"), dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    models = collect_models(args)
    if not models:
        print("No models found (use --original_ckpts and/or --runs_dir).")
        return
    data = {d: L.load_eval_dataset(d, args.data_root) for d in args.datasets}
    k_rows, n_rows = [], []
    for name, variant, loader in models:
        learner = loader()
        learner.eval()
        if not isinstance(learner, L.ProgramLearner):
            print(f"[skip] {name}: no program to iterate")
            continue
        _, N_test = L.test_time_schedule(L.VARIANTS[variant])
        for d in args.datasets:
            images, targets = data[d]
            for es in args.eval_seeds:
                if not args.skip_k_sweep:
                    for r in k_sweep(learner, images, targets, args, es, N_test):
                        k_rows.append(dict(model=name, dataset=d, eval_seed=es, **r))
                if not args.skip_n_sweep:
                    for r in n_sweep(learner, images, targets, args, es):
                        n_rows.append(dict(model=name, dataset=d, eval_seed=es, **r))
            last = [r for r in k_rows if r["model"] == name and r["dataset"] == d and r["test_acc"] != ""]
            if last:
                by_k = defaultdict(list)
                for r in last:
                    by_k[r["k"]].append(r["test_acc"])
                print(f"{name} {d}: " + "  ".join(f"k={k}:{np.mean(v):.1f}" for k, v in sorted(by_k.items())),
                      flush=True)
    for fname, rows in (("recurrence_k.csv", k_rows), ("recurrence_n.csv", n_rows)):
        if rows:
            with open(os.path.join(args.out_dir, fname), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    plot(args.out_dir, k_rows, n_rows, args.datasets, args.shot)
    print(f"Wrote results and plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
