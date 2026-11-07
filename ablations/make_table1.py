"""
Build the multi-seed Table 1: baselines (Train-Canonical-Models.py) and
MetaLearnNCA (eval_ablation.py), mean +- SD across TRAINING seeds.

Aggregation (same for every method): for each training seed, average the
accuracy over its evaluation tasks (eval seeds 43..52 by default); then report
the mean and SD of those per-seed scores. This captures training variance,
which the submitted Table 1 (one checkpoint, CI over test tasks) did not.

For each row the script also compares the main MetaLearnNCA column with the
best baseline (Welch 95% CI of the difference) and labels the row win / loss /
n.s., so the abstract can state exactly where the method wins and loses.

Inputs
  --baseline_csv  baseline_results/results_raw.csv   (Train-Canonical-Models.py)
  --nca_csv       ablation_results/results_raw.csv   (ablations/eval_ablation.py)
                  models used: 'full' (retrained, ablation protocol) and
                  'full_original' (checkpoints/), whichever are present

Outputs (--out_dir, default table1/): table1.md, table1.tex, table1_summary.csv

Example
  python ablations/make_table1.py --baseline_csv baseline_results/results_raw.csv \
      --nca_csv ablation_results/results_raw.csv
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_lib as L  # noqa: E402

BASELINES = [("MatchingNet", "MatchingNet"), ("RelationNet", "RelationNet"),
             ("ProtoNet", "ProtoNet"), ("MAML", "FOMAML")]
NCA_LABELS = {"full": "MetaLearnNCA", "full_original": "MetaLearnNCA (orig. ckpts)"}
TEST_TIME_GRADS = {"MatchingNet": "No", "RelationNet": "No", "ProtoNet": "No",
                   "MAML": "Yes (5 steps)", "full": "No", "full_original": "No"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline_csv", nargs="+", default=["baseline_results/results_raw.csv"],
                   help="one or more results_raw.csv files (e.g. one per parallel baseline process)")
    p.add_argument("--nca_csv", default="ablation_results/results_raw.csv")
    p.add_argument("--nca_models", nargs="+", default=["full", "full_original"],
                   help="model names from the eval_ablation CSV to show as MetaLearnNCA columns; "
                        "the first one present is compared against the baselines")
    p.add_argument("--datasets", nargs="+", default=L.DATASETS)
    p.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--stat", choices=["sd", "ci"], default="sd",
                   help="what follows +- : SD across training seeds (default) or 95%% t-CI")
    p.add_argument("--allow_subsampled", action="store_true",
                   help="also use eval_ablation rows run with --max_test_per_class (quick looks only)")
    p.add_argument("--out_dir", default="table1")
    return p.parse_args()


def load_baselines(paths):
    """{(method, seed, dataset, shot): {eval_seed: acc}}, params {method: n}."""
    scores, params, configs = defaultdict(dict), {}, defaultdict(list)
    rows = []
    for path in paths:
        if os.path.exists(path):
            rows += list(csv.DictReader(open(path)))
        else:
            print(f"[warn] no baseline results at {path}")
    # if a model was run with several settings (e.g. epochs, matching_scale), keep the latest
    for r in rows:
        cfg = (r["epochs"], r["meta_batch"], r["matching_scale"])
        if cfg not in configs[r["model"]]:
            configs[r["model"]].append(cfg)
    latest = {m: c[-1] for m, c in configs.items()}
    for m, c in configs.items():
        if len(c) > 1:
            print(f"[warn] {m} was run with several settings (epochs, meta_batch, matching_scale) = {c}; "
                  f"using the latest {latest[m]}")
    for r in rows:
        if (r["epochs"], r["meta_batch"], r["matching_scale"]) != latest[r["model"]]:
            continue
        key = (r["model"], r["train_seed"], r["dataset"], int(r["shot"]))
        scores[key][r["eval_seed"]] = float(r["acc"])   # duplicates: last row wins
        params[r["model"]] = int(r["n_params"])
    return scores, params


def load_nca(path, models, allow_subsampled):
    scores, params = defaultdict(dict), {}
    if not os.path.exists(path):
        print(f"[warn] no MetaLearnNCA results at {path}")
        return scores, params
    skipped = 0
    for r in csv.DictReader(open(path)):
        if r["model"] not in models or int(r["k"]) != int(r["K_test"]):
            continue
        if r["max_test_per_class"] not in ("", None) and not allow_subsampled:
            skipped += 1
            continue
        key = (r["model"], r["train_seed"], r["dataset"], int(r["shot"]))
        scores[key][r["eval_seed"]] = float(r["acc"])
        params[r["model"]] = int(r["n_params"])
    if skipped:
        print(f"[info] ignored {skipped} subsampled MetaLearnNCA rows (use --allow_subsampled to include)")
    return scores, params


def main():
    args = parse_args()
    b_scores, b_params = load_baselines(args.baseline_csv)
    n_scores, n_params = load_nca(args.nca_csv, args.nca_models, args.allow_subsampled)
    scores = {**b_scores, **n_scores}
    params = {**b_params, **n_params}

    methods = [(m, lab) for m, lab in BASELINES if m in params] + \
              [(m, NCA_LABELS.get(m, m)) for m in args.nca_models if m in params]
    if not methods:
        print("No results found.")
        return
    baseline_ids = [m for m, _ in BASELINES if m in params]
    primary = next((m for m in args.nca_models if m in params), None)
    datasets = [d for d in args.datasets if any(k[2] == d for k in scores)]

    def per_seed(method, d, shot):
        return {k[1]: float(np.mean(list(v.values()))) for k, v in scores.items()
                if k[0] == method and k[2] == d and k[3] == shot}

    def task_sets(method, d, shot):
        return {frozenset(v) for k, v in scores.items() if k[0] == method and k[2] == d and k[3] == shot}

    summary, cells_md, cells_tex = [], {}, {}
    mismatched = set()
    for d in datasets:
        for shot in args.shots:
            stats = {}
            tasks = {}
            for m, _ in methods:
                ps = per_seed(m, d, shot)
                if ps:
                    mean, sd, ci, n = L.mean_sd_ci(list(ps.values()))
                    stats[m] = dict(mean=mean, sd=sd, ci=ci, n=n, seeds=ps)
                    tasks[m] = task_sets(m, d, shot)
            all_sets = set().union(*tasks.values()) if tasks else set()
            if len(all_sets) > 1:
                mismatched.add((d, shot))
            if not stats:
                continue
            best = max(stats, key=lambda m: stats[m]["mean"])
            best_base = max((m for m in baseline_ids if m in stats), key=lambda m: stats[m]["mean"], default=None)
            for m, lab in methods:
                if m not in stats:
                    continue
                st = stats[m]
                pm = st["sd"] if args.stat == "sd" else st["ci"]
                bold = m == best
                txt = f"{st['mean']:.2f} ± {pm:.2f}" if st["n"] > 1 else f"{st['mean']:.2f} (1 seed)"
                cells_md[(d, shot, m)] = f"**{txt}**" if bold else txt
                tex = f"{st['mean']:.2f}" + (f"_{{\\pm {pm:.2f}}}" if st["n"] > 1 else "")
                cells_tex[(d, shot, m)] = f"$\\mathbf{{{tex}}}$" if bold else f"${tex}$"
                summary.append(dict(dataset=d, shot=shot, method=lab, mean=round(st["mean"], 3),
                                    sd=round(st["sd"], 3), ci95=round(st["ci"], 3), n_seeds=st["n"],
                                    seeds=" ".join(sorted(st["seeds"], key=int))))
            if primary in stats and best_base:
                diff, hw = L.welch_diff_ci(list(stats[primary]["seeds"].values()),
                                           list(stats[best_base]["seeds"].values()))
                if hw == hw:   # not NaN
                    verdict = "win" if diff - hw > 0 else "loss" if diff + hw < 0 else "n.s."
                    cells_md[(d, shot, "delta")] = f"{diff:+.2f} ± {hw:.2f} vs {dict(BASELINES)[best_base]} ({verdict})"
                else:
                    verdict = "win" if diff > 0 else "loss"
                    cells_md[(d, shot, "delta")] = f"{diff:+.2f} vs {dict(BASELINES)[best_base]} ({verdict}, no CI: <2 seeds)"
                summary.append(dict(dataset=d, shot=shot, method=f"delta {NCA_LABELS.get(primary, primary)} - "
                                    f"{dict(BASELINES)[best_base]}", mean=round(diff, 3), ci95=round(hw, 3),
                                    verdict=verdict))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "table1_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "shot", "method", "mean", "sd", "ci95", "n_seeds",
                                          "seeds", "verdict"])
        w.writeheader()
        w.writerows(summary)

    pm_name = "SD" if args.stat == "sd" else "95% CI"
    n_seed_txt = ", ".join(f"{lab}: {max((s['n_seeds'] for s in summary if s['method'] == lab), default=0)}"
                           for _, lab in methods)
    labels = [lab for _, lab in methods]

    # ---------------- Markdown ----------------
    md = ["# Table 1 (multi-seed)", "",
          f"10-way accuracy (%), mean ± {pm_name} across training seeds; each seed is first averaged over "
          f"its evaluation tasks. Training seeds: {n_seed_txt}. Bold = best mean in the row. "
          "Last column: main MetaLearnNCA column minus the best baseline, Welch 95% CI; "
          "win/loss only when the CI excludes 0.", "",
          "| Dataset | Shot | " + " | ".join(labels) + (" | Δ vs best baseline |" if primary else " |"),
          "|---|---|" + "---|" * (len(labels) + (1 if primary else 0))]
    for d in datasets:
        for i, shot in enumerate(args.shots):
            name = L.DATASET_LABELS.get(d, d) + (" (in-dist.)" if d == "omniglot" else "")
            row = [name if i == 0 else "", str(shot)] + [cells_md.get((d, shot, m), "–") for m, _ in methods]
            if primary:
                row.append(cells_md.get((d, shot, "delta"), "–"))
            md.append("| " + " | ".join(row) + " |")
    md.append("| **Test-time gradients?** | – | " + " | ".join(TEST_TIME_GRADS.get(m, "?") for m, _ in methods)
              + (" | |" if primary else " |"))
    md.append("| **Parameters** | – | " + " | ".join(f"{params[m]:,}" for m, _ in methods)
              + (" | |" if primary else " |"))
    if mismatched:
        md += ["", "**Warning:** methods were scored on different evaluation tasks for: " +
               ", ".join(f"{d} {s}-shot" for d, s in sorted(mismatched)) +
               ". Use the same evaluation seeds for all methods."]
    with open(os.path.join(args.out_dir, "table1.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ---------------- LaTeX ----------------
    pm_tex = "SD" if args.stat == "sd" else r"95\% CI"
    n_tasks = {len(v) for v in scores.values()}
    task_txt = (f"the same {n_tasks.pop()} evaluation tasks for every method" if len(n_tasks) == 1 and not mismatched
                else "its evaluation tasks")
    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{ll" + "c" * len(methods) + "}", r"\toprule",
           "Dataset & Shot & " + " & ".join(labels) + r" \\", r"\midrule"]
    for di, d in enumerate(datasets):
        for i, shot in enumerate(args.shots):
            name = L.DATASET_LABELS.get(d, d) + (" (in-dist.)" if d == "omniglot" else "")
            tex.append((name if i == 0 else "") + f" & {shot} & " +
                       " & ".join(cells_tex.get((d, shot, m), "--") for m, _ in methods) + r" \\")
        tex.append(r"\midrule")
    tex.append(r"Test-time gradients? & & " + " & ".join(TEST_TIME_GRADS.get(m, "?") for m, _ in methods) + r" \\")
    tex.append(r"Parameters & & " + " & ".join(f"$\\approx {params[m] / 1000:.0f}$k" for m, _ in methods) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Few-shot accuracy (\%, 10-way), mean$\pm$" + pm_tex + " "
            r"across training seeds (each seed averaged over " + task_txt + "). "
            r"Bold: best mean per row.}", r"\label{tab:main_results}", r"\end{table}"]
    with open(os.path.join(args.out_dir, "table1.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\nWrote {args.out_dir}/table1.md, table1.tex, table1_summary.csv")


if __name__ == "__main__":
    main()
