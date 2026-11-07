"""
Evaluate trained ablation runs (and optionally the original checkpoints) with
the Table-1 protocol, then build the ablation tables.

Protocol (identical to ML-NCA-Testing.py --run_eval_loop):
  for each evaluation seed (default 43..52) and dataset: draw 10 classes,
  k_shot support images per class, test on ALL remaining images of those
  classes; K_test = 5 interactions, N_test = 19 (top of the training jitter).
  Tasks depend only on (dataset, shot, eval seed), so every model is scored on
  exactly the same tasks (paired comparison).

Aggregation: each training run's score is its mean over evaluation seeds; the
table reports mean +- SD across training seeds (plus 95% t-CI in the CSV), and
the difference to the reference model with a Welch 95% CI.

Results are appended to <out_dir>/results_raw.csv as they are produced, so an
interrupted evaluation resumes where it stopped.

Examples
--------
  # everything trained under ablation_runs/, plus 3 original checkpoints as an extra row
  python ablations/eval_ablation.py --runs_dir ablation_runs \
      --original_ckpts checkpoints --original_seeds 0 1 6

  # quick look (100 test images per class, 3 tasks, 5-shot only)
  python ablations/eval_ablation.py --shots 5 --eval_seeds 43 44 45 --max_test_per_class 100 \
      --out_dir ablation_results_quick

  # rebuild tables from an existing CSV
  python ablations/eval_ablation.py --summarize_only --out_dir ablation_results
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_lib as L  # noqa: E402

ORIGINAL = "full_original"
RAW_FIELDS = ["model", "variant", "train_seed", "dataset", "shot", "eval_seed", "k", "acc",
              "adapt_time_s", "peak_mem_mb", "n_test", "K_test", "N_test", "n_params",
              "max_test_per_class"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default="ablation_runs")
    p.add_argument("--variants", nargs="+", default=None, help="default: all found in runs_dir")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="restrict training seeds")
    p.add_argument("--original_ckpts", default=None, help="e.g. checkpoints (adds row 'full_original')")
    p.add_argument("--original_seeds", nargs="+", default=None, help="seed ids, or 'all'")
    p.add_argument("--datasets", nargs="+", default=L.DATASETS)
    p.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--eval_seeds", nargs="+", type=int, default=list(range(43, 53)))
    p.add_argument("--k_test", type=int, default=None, help="override K_test for all models")
    p.add_argument("--n_test", type=int, default=None, help="override N_test for all models")
    p.add_argument("--trajectory", action="store_true", help="also score k = 0..K_test")
    p.add_argument("--max_test_per_class", type=int, default=None, help="subsample test set (speed)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--amp", default="none", choices=["none", "fp16", "bf16"],
                   help="evaluation precision (original evaluation: fp32)")
    p.add_argument("--include_diverged", action="store_true")
    p.add_argument("--reference", default="full", help="model the deltas are computed against")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="ablation_results")
    p.add_argument("--summarize_only", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------
def discover_runs(runs_dir, variants, seeds, include_diverged):
    runs, skipped = [], []
    if not os.path.isdir(runs_dir):
        return runs, skipped
    for variant in sorted(os.listdir(runs_dir)):
        if variant not in L.VARIANTS or (variants and variant not in variants):
            continue
        vdir = os.path.join(runs_dir, variant)
        for sd in sorted(os.listdir(vdir)):
            if not sd.startswith("seed"):
                continue
            seed = int(sd[4:])
            if seeds and seed not in seeds:
                continue
            rdir = os.path.join(vdir, sd)
            st_path, model_path = os.path.join(rdir, "status.json"), os.path.join(rdir, "model.pt")
            status = json.load(open(st_path)).get("status") if os.path.exists(st_path) else "missing"
            ok = os.path.exists(model_path) and (status == "ok" or (include_diverged and status == "diverged"))
            (runs if ok else skipped).append(dict(variant=variant, seed=seed, dir=rdir, status=status))
    return runs, skipped


def load_done(raw_path):
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for r in csv.DictReader(f):
                done.add((r["model"], r["train_seed"], r["dataset"], r["shot"], r["eval_seed"],
                          r["max_test_per_class"]))
    return done


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def evaluate_model(learner, model_name, variant, train_seed, args, data, done, writer, fh):
    spec = L.VARIANTS[variant]
    K, N = L.test_time_schedule(spec)
    K = args.k_test or K
    N = args.n_test or N
    amp = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    n_params = L.count_params(learner)
    learner.eval()
    mtpc = "" if args.max_test_per_class is None else str(args.max_test_per_class)
    for dname in args.datasets:
        images, targets = data[dname]
        for shot in args.shots:
            accs = []
            for es in args.eval_seeds:
                key = (model_name, str(train_seed), dname, str(shot), str(es), mtpc)
                if key in done:
                    continue
                res = L.evaluate_task(learner, images, targets, shot, es, K, N, args.device,
                                      batch_size=args.batch_size,
                                      max_test_per_class=args.max_test_per_class,
                                      k_eval=list(range(K + 1)) if args.trajectory else None,
                                      amp_dtype=amp)
                for k, acc in res["acc"].items():
                    writer.writerow(dict(model=model_name, variant=variant, train_seed=train_seed,
                                         dataset=dname, shot=shot, eval_seed=es, k=k, acc=f"{acc:.4f}",
                                         adapt_time_s=f"{res['adapt_time_s']:.5f}",
                                         peak_mem_mb=f"{res['peak_mem_mb']:.1f}", n_test=res["n_test"],
                                         K_test=K, N_test=N, n_params=n_params, max_test_per_class=mtpc))
                fh.flush()
                accs.append(res["acc"][K])
            if accs:
                print(f"  {model_name:<20} seed {train_seed:<4} {dname:<9} {shot:>2}-shot: "
                      f"{np.mean(accs):6.2f}% (over {len(accs)} tasks)", flush=True)


# ----------------------------------------------------------------------
# aggregation & tables
# ----------------------------------------------------------------------
def fmt_delta(r, with_ci, without_ci, missing):
    if not r or "delta_vs_ref" not in r:
        return missing
    if r["delta_ci95"] == r["delta_ci95"]:          # not NaN
        return with_ci.format(r["delta_vs_ref"], r["delta_ci95"])
    return without_ci.format(r["delta_vs_ref"])


def model_label(model):
    if model == ORIGINAL:
        return "Full (original checkpoints)"
    return L.VARIANTS[model].label if model in L.VARIANTS else model


def model_order(models):
    order = ["full", ORIGINAL] + L.MAIN_VARIANTS + L.APPENDIX_VARIANTS
    return sorted(models, key=lambda m: order.index(m) if m in order else len(order))


def summarize(out_dir, reference, runs_dir=None):
    raw_path = os.path.join(out_dir, "results_raw.csv")
    if not os.path.exists(raw_path):
        print("No results yet.")
        return
    rows = list(csv.DictReader(open(raw_path)))
    if len({r["max_test_per_class"] for r in rows}) > 1:
        print("WARNING: results_raw.csv mixes different --max_test_per_class settings; "
              "use separate --out_dir values for different protocols.")
    # per (model, seed, dataset, shot): mean over eval seeds at k == K_test
    per_run = defaultdict(list)
    meta = defaultdict(lambda: dict(params=set(), adapt=[], mem=[], ntask=0))
    for r in rows:
        if int(r["k"]) != int(r["K_test"]):
            continue
        per_run[(r["model"], r["train_seed"], r["dataset"], int(r["shot"]))].append(float(r["acc"]))
        m = meta[(r["model"], int(r["shot"]))]
        m["params"].add(int(r["n_params"]))
        m["adapt"].append(float(r["adapt_time_s"]))
        if r["peak_mem_mb"] not in ("", "nan"):
            m["mem"].append(float(r["peak_mem_mb"]))
    run_score = {k: float(np.mean(v)) for k, v in per_run.items()}
    n_tasks = {k: len(v) for k, v in per_run.items()}

    models = model_order({k[0] for k in run_score})
    datasets = [d for d in L.DATASETS + ["fake"] if any(k[2] == d for k in run_score)]
    shots = sorted({k[3] for k in run_score})

    def seed_scores(model, dataset, shot):
        return {k[1]: v for k, v in run_score.items() if k[0] == model and k[2] == dataset and k[3] == shot}

    def ood_scores(model, shot):
        ood = [d for d in L.OOD_DATASETS if d in datasets]
        per_seed = defaultdict(dict)
        for d in ood:
            for s, v in seed_scores(model, d, shot).items():
                per_seed[s][d] = v
        return {s: float(np.mean(list(v.values()))) for s, v in per_seed.items() if len(v) == len(ood)} \
            if ood else {}

    ref = reference if reference in models else (ORIGINAL if ORIGINAL in models else None)

    # divergence bookkeeping
    status_counts = {}
    if runs_dir and os.path.isdir(runs_dir):
        for v in os.listdir(runs_dir):
            vdir = os.path.join(runs_dir, v)
            if v not in L.VARIANTS or not os.path.isdir(vdir):
                continue
            c = defaultdict(int)
            for sd in os.listdir(vdir):
                p = os.path.join(vdir, sd, "status.json")
                if os.path.exists(p):
                    c[json.load(open(p)).get("status", "?")] += 1
            status_counts[v] = dict(c)

    # ---- summary.csv ----
    summary_rows = []
    for shot in shots:
        for m in models:
            cols = datasets + (["ood_mean"] if any(d in L.OOD_DATASETS for d in datasets) else [])
            for d in cols:
                sc = ood_scores(m, shot) if d == "ood_mean" else seed_scores(m, d, shot)
                if not sc:
                    continue
                mean, sd, ci, n = L.mean_sd_ci(list(sc.values()))
                row = dict(model=m, label=model_label(m), shot=shot, dataset=d, mean=round(mean, 3),
                           sd=round(sd, 3), ci95=round(ci, 3), n_seeds=n,
                           seeds=" ".join(sorted(sc, key=lambda x: int(x))))
                if d != "ood_mean":
                    row["tasks_per_seed"] = min(n_tasks[(m, s, d, shot)] for s in sc)
                if ref and m != ref:
                    rs = ood_scores(ref, shot) if d == "ood_mean" else seed_scores(ref, d, shot)
                    if rs:
                        diff, hw = L.welch_diff_ci(list(sc.values()), list(rs.values()))
                        row.update(delta_vs_ref=round(diff, 3), delta_ci95=round(hw, 3))
                summary_rows.append(row)
    fields = ["model", "label", "shot", "dataset", "mean", "sd", "ci95", "n_seeds", "seeds",
              "tasks_per_seed", "delta_vs_ref", "delta_ci95"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    lookup = {(r["model"], r["shot"], r["dataset"]): r for r in summary_rows}
    cols = datasets + (["ood_mean"] if any(d in L.OOD_DATASETS for d in datasets) else [])
    n_ood = sum(d in L.OOD_DATASETS for d in datasets)
    col_label = {**L.DATASET_LABELS,
                 "ood_mean": "OOD mean" if n_ood == len(L.OOD_DATASETS) else f"OOD mean ({n_ood} sets)"}

    # ---- markdown ----
    md = ["# MetaLearnNCA component ablations", "",
          "Accuracy (%) on 10-way tasks; mean ± SD across training seeds "
          "(each seed = mean over the evaluation tasks). "
          f"Δ = OOD-mean difference to `{ref}` with Welch 95% CI. "
          "Omniglot is in-distribution (evaluation alphabets); the rest are OOD.", ""]
    for shot in shots:
        md += [f"## {shot}-shot", "",
               "| Variant | " + " | ".join(col_label[c] for c in cols) +
               " | Δ OOD vs ref | #params | adapt. time (s) | runs ok |",
               "|---|" + "---|" * (len(cols) + 4)]
        for m in models:
            cells = []
            for c in cols:
                r = lookup.get((m, shot, c))
                cells.append(f"{r['mean']:.2f} ± {r['sd']:.2f} (n={r['n_seeds']})" if r else "–")
            r = lookup.get((m, shot, "ood_mean"))
            delta = fmt_delta(r, "{:+.2f} ± {:.2f}", "{:+.2f}", "–")
            mm = meta[(m, shot)]
            params = "/".join(str(p) for p in sorted(mm["params"]))
            at = f"{np.mean(mm['adapt']):.2f}" if mm["adapt"] else "–"
            sc = status_counts.get(m)
            ok = f"{sc.get('ok', 0)}/{sum(sc.values())}" if sc else "–"
            md.append(f"| {model_label(m)} | " + " | ".join(cells) + f" | {delta} | {params} | {at} | {ok} |")
        md.append("")
    if status_counts:
        md += ["Training status per variant (all runs found): " +
               "; ".join(f"{v}: {dict(c)}" for v, c in sorted(status_counts.items())), ""]
    with open(os.path.join(out_dir, "ablation_tables.md"), "w") as f:
        f.write("\n".join(md))

    # ---- LaTeX ----
    tex = []
    for shot in shots:
        tex += [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
                "Variant & " + " & ".join(col_label[c] for c in cols) + r" & $\Delta$ OOD \\",
                r"\midrule"]
        for m in models:
            cells = []
            for c in cols:
                r = lookup.get((m, shot, c))
                cells.append(f"${r['mean']:.1f}_{{\\pm {r['sd']:.1f}}}$" if r else "--")
            r = lookup.get((m, shot, "ood_mean"))
            delta = fmt_delta(r, "${:+.1f}_{{\\pm {:.1f}}}$", "${:+.1f}$", "--")
            tex.append(model_label(m).replace("_", r"\_") + " & " + " & ".join(cells) + f" & {delta} \\\\")
        tex += [r"\bottomrule", r"\end{tabular}",
                rf"\caption{{Component ablations, 10-way {shot}-shot. Mean$\pm$SD over training seeds; "
                r"$\Delta$: OOD-mean difference to the full model (Welch 95\% CI).}",
                rf"\label{{tab:ablation_{shot}shot}}", r"\end{table}", ""]
    with open(os.path.join(out_dir, "ablation_tables.tex"), "w") as f:
        f.write("\n".join(tex))
    print("\n".join(md))
    print(f"\nWrote {out_dir}/summary.csv, ablation_tables.md, ablation_tables.tex")


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.summarize_only:
        summarize(args.out_dir, args.reference, args.runs_dir)
        return

    runs, skipped = discover_runs(args.runs_dir, args.variants, args.seeds, args.include_diverged)
    for s in skipped:
        print(f"[not evaluated] {s['variant']} seed {s['seed']}: status={s['status']}")
    originals = {}
    if args.original_ckpts:
        originals = L.find_original_checkpoints(args.original_ckpts, args.original_seeds)
    if not runs and not originals:
        print("Nothing to evaluate.")
        return
    print(f"Evaluating {len(runs)} ablation run(s) + {len(originals)} original checkpoint(s) on "
          f"{args.datasets}, shots {args.shots}, {len(args.eval_seeds)} tasks each.")

    data = {}
    for d in args.datasets:
        images, targets = L.load_eval_dataset(d, args.data_root)
        data[d] = (images.to(args.device), targets)

    raw_path = os.path.join(args.out_dir, "results_raw.csv")
    done = load_done(raw_path)
    new_file = not os.path.exists(raw_path)
    with open(raw_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
        if new_file:
            writer.writeheader()
        for seed, (ap, mp) in originals.items():
            learner = L.load_original_checkpoint(ap, mp, args.device)
            evaluate_model(learner, ORIGINAL, "full", seed, args, data, done, writer, fh)
        for run in runs:
            learner, _ = L.load_run_checkpoint(os.path.join(run["dir"], "model.pt"), args.device)
            evaluate_model(learner, run["variant"], run["variant"], run["seed"], args, data, done, writer, fh)
    summarize(args.out_dir, args.reference, args.runs_dir)


if __name__ == "__main__":
    main()
