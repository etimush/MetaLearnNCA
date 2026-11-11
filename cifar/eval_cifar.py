"""
Evaluate trained FC100 runs on the held-out superclasses and build the comparison table.

Protocol (identical for every method and seed):
  * 5-way episodes from the 20 TEST classes (superclasses aquatic mammals, insects, medium
    mammals, people: never seen during meta-training), 1-shot and 5-shot, 15 queries per class,
    1000 episodes per shot. Episode e is a fixed function of (episode_seed, shot, e), so every
    model is scored on exactly the same tasks.
  * No test-time tuning on test data. FOMAML is the only method with a test-time hyper-parameter
    (its inner learning rate); it is chosen per shot on episodes from the 20 VALIDATION classes
    (disjoint superclasses), and the choice is recorded.
  * Per run: mean accuracy over episodes (+ 95% CI over episodes). Per method: mean +- SD of those
    run means across training seeds.

Results per run are cached in <run_dir>/eval_fc100.json (re-running skips finished runs).

Examples
  python cifar/eval_cifar.py --runs_dir cifar_runs
  python cifar/eval_cifar.py --runs_dir cifar_runs --summarize_only
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar_lib as C  # noqa: E402
from train_cifar import load_model  # noqa: E402


def parse_args():
    R = C.REGIME
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default="cifar_runs")
    p.add_argument("--methods", nargs="+", default=C.METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--shots", nargs="+", type=int, default=list(R["test_shots"]))
    p.add_argument("--episodes", type=int, default=R["test_episodes"])
    p.add_argument("--query", type=int, default=R["test_query"])
    p.add_argument("--episode_seed", type=int, default=R["episode_seed"])
    p.add_argument("--batch_episodes", type=int, default=8, help="episodes processed together")
    p.add_argument("--fomaml_lrs", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05, 0.1],
                   help="candidate test-time inner learning rates for FOMAML (selected on validation classes)")
    p.add_argument("--val_episodes", type=int, default=200)
    p.add_argument("--include_diverged", action="store_true")
    p.add_argument("--force", action="store_true", help="re-evaluate runs that already have results")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fake_data", action="store_true")
    p.add_argument("--out_dir", default="cifar_results")
    p.add_argument("--summarize_only", action="store_true")
    return p.parse_args()


def discover(args):
    runs = []
    for m in args.methods:
        mdir = os.path.join(args.runs_dir, m)
        if not os.path.isdir(mdir):
            continue
        for sd in sorted(os.listdir(mdir)):
            if not sd.startswith("seed") or (args.seeds and int(sd[4:]) not in args.seeds):
                continue
            rdir = os.path.join(mdir, sd)
            sp = os.path.join(rdir, "status.json")
            status = json.load(open(sp)).get("status") if os.path.exists(sp) else "missing"
            ok = os.path.exists(os.path.join(rdir, "model.pt")) and \
                (status == "ok" or (args.include_diverged and status == "diverged"))
            runs.append(dict(method=m, seed=int(sd[4:]), dir=rdir, status=status, usable=ok))
    return runs


def episode_accuracies(model, bank, shot, n_episodes, args, seed, device):
    accs = []
    for start in range(0, n_episodes, args.batch_episodes):
        n = min(args.batch_episodes, n_episodes - start)
        sx, sy, qx, qy = C.fixed_episodes(bank, n, C.REGIME["n_way"], shot, args.query, seed, start=start)
        torch.manual_seed(seed * 7919 + shot * 104729 + start)       # NCA stochasticity, reproducible
        scores = model.predict(sx.to(device), sy.to(device), qx.to(device))
        accs += (scores.argmax(-1).cpu() == qy).float().mean(1).mul(100).tolist()
    return accs


def evaluate_run(run, banks, args, device):
    out_path = os.path.join(run["dir"], "eval_fc100.json")
    protocol = dict(episodes=args.episodes, query=args.query, episode_seed=args.episode_seed, shots=args.shots,
                    fake_data=args.fake_data, val_episodes=args.val_episodes, fomaml_lrs=args.fomaml_lrs)
    if os.path.exists(out_path) and not args.force:
        prev = json.load(open(out_path))
        if prev.get("protocol") == protocol:
            return prev
    model, ck = load_model(run["dir"], device)
    result = dict(method=run["method"], seed=run["seed"], n_params=ck["n_params"], protocol=protocol, shots={})
    t0 = time.time()
    for shot in args.shots:
        if run["method"] == "fomaml":
            val = {}
            for lr in args.fomaml_lrs:
                model.test_inner_lr = lr
                val[lr] = float(np.mean(episode_accuracies(model, banks["val"], shot, args.val_episodes, args,
                                                           args.episode_seed + 1, device)))
            model.test_inner_lr = max(val, key=val.get)
            result.setdefault("fomaml_val_selection", {})[str(shot)] = dict(chosen_lr=model.test_inner_lr,
                                                                           val_acc=val)
        accs = episode_accuracies(model, banks["test"], shot, args.episodes, args, args.episode_seed, device)
        mean, ci = C.mean_ci95(accs)
        result["shots"][str(shot)] = dict(mean=mean, ci95=ci, accs=accs)
        print(f"  {run['method']:<13} seed {run['seed']}  {shot}-shot: {mean:6.2f} +- {ci:.2f}%"
              + (f"  (FOMAML test lr {model.test_inner_lr}, chosen on val)" if run["method"] == "fomaml" else ""),
              flush=True)
    result["eval_time_s"] = time.time() - t0
    with open(out_path + ".tmp", "w") as f:
        json.dump(result, f)
    os.replace(out_path + ".tmp", out_path)
    return result


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float(a.mean() - b.mean()), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return float(a.mean() - b.mean()), 0.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, df))
    except ImportError:
        t = 1.96
    return float(a.mean() - b.mean()), t * se


def summarize(args, runs):
    results = {}
    for r in runs:
        p = os.path.join(r["dir"], "eval_fc100.json")
        if r["usable"] and os.path.exists(p):
            results.setdefault(r["method"], []).append(json.load(open(p)))
    if not results:
        print("No evaluated runs.")
        return
    methods = [m for m in C.METHODS if m in results]
    status = {m: (sum(r["status"] == "ok" for r in runs if r["method"] == m),
                  sum(1 for r in runs if r["method"] == m)) for m in methods}
    rows, cells = [], {}
    for shot in args.shots:
        per_seed = {m: [res["shots"][str(shot)]["mean"] for res in results[m] if str(shot) in res["shots"]]
                    for m in methods}
        for m in methods:
            v = np.asarray(per_seed[m])
            if not len(v):
                continue
            sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            ep_ci = float(np.mean([res["shots"][str(shot)]["ci95"] for res in results[m]]))
            rows.append(dict(method=C.METHOD_LABELS[m], shot=shot, mean=round(float(v.mean()), 2),
                             sd_over_seeds=round(sd, 2), n_seeds=len(v), mean_episode_ci95=round(ep_ci, 2),
                             n_params=results[m][0]["n_params"]))
            cells[(m, shot)] = (float(v.mean()), sd, len(v))
        baselines = [m for m in ("protonet", "matchingnet", "relationnet", "fomaml") if per_seed.get(m)]
        if baselines:
            best = max(baselines, key=lambda m: np.mean(per_seed[m]))
            for m in ("metalearnnca", "nca_protonet"):
                if per_seed.get(m):
                    d, hw = welch(per_seed[m], per_seed[best])
                    cells[(m, shot, "delta")] = (d, hw, best)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(m, shot):
        if (m, shot) not in cells:
            return "–"
        mean, sd, n = cells[(m, shot)]
        return f"{mean:.2f} ± {sd:.2f}" if n > 1 else f"{mean:.2f} (1 seed)"

    def fmt_delta(m, shot):
        if (m, shot, "delta") not in cells:
            return "–"
        d, hw, best = cells[(m, shot, "delta")]
        return f"{d:+.2f}" + (f" ± {hw:.2f}" if hw == hw else "") + f" vs {C.METHOD_LABELS[best]}"

    md = ["# FC100 (CIFAR-100, superclass-disjoint) few-shot results", "",
          f"5-way accuracy (%) on the 20 held-out test classes, {args.episodes} episodes per shot "
          f"({args.query} queries per class), identical episodes for every model. Mean ± SD across training "
          "seeds (each seed = mean over episodes). Δ: difference to the best canonical baseline, Welch 95% CI.", "",
          "| Method | " + " | ".join(f"{s}-shot" for s in args.shots) + " | "
          + " | ".join(f"Δ {s}-shot" for s in args.shots) + " | Parameters | Test-time gradients | Runs ok |",
          "|---|" + "---|" * (2 * len(args.shots) + 3)]
    for m in methods:
        md.append(f"| {C.METHOD_LABELS[m]} | " + " | ".join(fmt(m, s) for s in args.shots) + " | "
                  + " | ".join(fmt_delta(m, s) for s in args.shots) + f" | {results[m][0]['n_params']:,} | "
                  f"{C.TEST_TIME_GRADIENTS[m]} | {status[m][0]}/{status[m][1]} |")
    fom = [res.get("fomaml_val_selection") for res in results.get("fomaml", [])]
    if any(fom):
        md += ["", "FOMAML test-time inner learning rate (chosen on validation classes): " +
               "; ".join(f"seed {res['seed']}: " + ", ".join(f"{s}-shot {v['chosen_lr']}"
                                                            for s, v in res["fomaml_val_selection"].items())
                         for res in results["fomaml"])]
    with open(os.path.join(args.out_dir, "table.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{l" + "c" * len(args.shots) + "cc}", r"\toprule",
           "Method & " + " & ".join(f"{s}-shot" for s in args.shots) + r" & Params & Test-time grad. \\",
           r"\midrule"]
    for m in methods:
        vals = []
        for s in args.shots:
            if (m, s) in cells:
                mean, sd, n = cells[(m, s)]
                vals.append(f"${mean:.2f}" + (f"_{{\\pm {sd:.2f}}}" if n > 1 else "") + "$")
            else:
                vals.append("--")
        tex.append(f"{C.METHOD_LABELS[m]} & " + " & ".join(vals) +
                   f" & {results[m][0]['n_params'] / 1000:.1f}k & {C.TEST_TIME_GRADIENTS[m]} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{5-way few-shot accuracy (\%) on FC100 (CIFAR-100 split by superclass; test superclasses "
            r"unseen during meta-training). Mean$\pm$SD over training seeds; all models trained with the same "
            r"episodes, updates and optimiser, and matched to the same parameter budget.}",
            r"\label{tab:fc100}", r"\end{table}"]
    with open(os.path.join(args.out_dir, "table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"\nWrote {args.out_dir}/table.md, table.tex, summary.csv")


def main():
    args = parse_args()
    runs = discover(args)
    for r in runs:
        if not r["usable"]:
            print(f"[not evaluated] {r['method']} seed {r['seed']}: status {r['status']}")
    if not args.summarize_only:
        banks = C.make_fake_banks(seed=0) if args.fake_data else C.load_fc100(args.data_root)
        todo = [r for r in runs if r["usable"]]
        print(f"Evaluating {len(todo)} run(s): {args.episodes} test episodes x shots {args.shots}")
        for r in todo:
            evaluate_run(r, banks, args, args.device)
    summarize(args, runs)


if __name__ == "__main__":
    main()
