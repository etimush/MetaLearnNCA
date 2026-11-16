"""
Test-time compute cost of every model (R1 Q1, R3 W3/Q1): FLOPs, wall-clock latency and peak
memory, split into the two test-time stages every few-shot method has:

  adaptation   support set -> task state, once per task
               ProtoNet / RelationNet: embed the support set, class means
               MatchingNet:            embed + normalise the support set
               FOMAML:                 5 inner SGD steps (forward + backward) on the support set
               MetaLearnNCA:           K = 5 rounds of [Active-NCA on the support set (2N = 38 steps),
                                       Meta-NCA (N = 19 steps)] -> program S
               NCA-ProtoNet:           Active-NCA embedding of the support set, class means
               grad_on_s:              K gradient steps on S through the Active-NCA unroll
  query        task state + one query image -> class scores, reported per image

All models run the same code paths as their evaluation (the baselines' modules from
Train-Canonical-Models.py, the NCA learners from ablation_lib with the test-time K, N). The only
difference: the baselines' evaluation loop re-embeds the support set with every query batch; here
it is embedded once (the cheaper, deployable form). A start-up check verifies that the split
adapt/predict gives the same scores as each baseline's forward().

Measured, per model and shot (10-way tasks):
  * FLOPs: matrix multiplications and convolutions (incl. the depthwise perception filters and,
    for FOMAML / grad_on_s, the backward passes), counted by torch.utils.flop_counter, with one
    multiply-add = 2 FLOPs (so FLOPs = 2 x MACs; the same convention as calculate_conv4_flops in
    Train-Canonical-Models.py: 19.6 MFLOPs per Conv4 image); element-wise ops, normalisation,
    pooling and softmax are not counted (standard convention).
  * Wall-clock: median of --repeats timed runs after --warmup runs, synchronised on GPU.
  * Peak memory (GPU only): peak allocation above what was resident before the stage
    (weights + inputs), i.e. the activations / working memory of that stage.
Cost does not depend on the weights or the image content, so models are randomly initialised and
the images are random; parameter counts are those of the trained models.

Outputs (--out_dir, default cost_results/): cost.csv, cost.md, cost.tex

Examples
  python ablations/test_time_cost.py                         # default models, 1/5/10-shot
  python ablations/test_time_cost.py --shots 10 --repeats 5  # quicker
  python ablations/test_time_cost.py --query_batch 1         # single-image query latency
  python ablations/test_time_cost.py --models ProtoNet MAML full short_unroll no_feedback
"""
import argparse
import csv
import importlib.util
import math
import os
import platform
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ablation_lib as L  # noqa: E402

BASELINES = ["MatchingNet", "RelationNet", "ProtoNet", "MAML"]
LABELS = {"MAML": "FOMAML", "full": "MetaLearnNCA", "nca_protonet": "NCA-ProtoNet",
          "grad_on_s": "Gradient on $S$"}
DEFAULT_MODELS = BASELINES + ["full", "nca_protonet", "grad_on_s"]
MAML_INNER_STEPS = 5           # as in evaluate_global_dataset (Train-Canonical-Models.py)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help=f"baselines {BASELINES} and/or ablation variants {list(L.VARIANTS)}")
    p.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--n_way", type=int, default=L.N_WAY)
    p.add_argument("--query_batch", type=int, default=100,
                   help="query images per timed call (default 100 = 10 per class); per-image numbers "
                        "are divided by this. Use 1 for single-image latency.")
    p.add_argument("--repeats", type=int, default=10, help="timed runs per measurement (median reported)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--time_budget", type=float, default=30.0,
                   help="stop timing a measurement after this many seconds (at least 3 runs)")
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none",
                   help="precision for every model (evaluation default: fp32)")
    p.add_argument("--literal_nca", action="store_true",
                   help="NCA without the exact static-term precomputation (the form of the original "
                        "ML-NCA-Testing.py: same outputs, more FLOPs)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--latex_shot", type=int, default=None, help="shot shown in cost.tex (default: largest)")
    p.add_argument("--out_dir", default="cost_results")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------
# models behind one interface: adapt(supp_x, supp_y) -> state, predict(state, qry_x) -> scores
# ----------------------------------------------------------------------
def load_baseline_module():
    spec = importlib.util.spec_from_file_location("canonical_models",
                                                  os.path.join(ROOT, "Train-Canonical-Models.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BaselineRunner:
    def __init__(self, mod, name, n_way, device):
        self.name, self.n_way = name, n_way
        ctor = dict(ProtoNet=mod.ProtoNet, MatchingNet=mod.MatchingNet, RelationNet=mod.RelationNet)
        model = mod.MAMLNet(n_way=n_way) if name == "MAML" else ctor[name]()
        self.model = model.to(device).eval()
        if name != "MAML":                      # FOMAML differentiates w.r.t. its weights at test time
            self.model.requires_grad_(False)
        self.test_grads = f"Yes ({MAML_INNER_STEPS} steps)" if name == "MAML" else "No"
        self.schedule = ""

    def adapt(self, sx, sy):
        m = self.model
        if self.name == "MAML":
            with torch.enable_grad():
                return m.adapt(sx, sy, inner_steps=MAML_INNER_STEPS, inner_lr=0.03, grad_clip=1.0,
                               create_graph=False)
        f = m.backbone(sx)
        if self.name == "MatchingNet":
            return F.normalize(f, dim=-1), sy
        return torch.stack([f[sy == c].mean(0) for c in range(self.n_way)])

    def predict(self, state, qx):
        m = self.model
        if self.name == "MAML":
            return torch.func.functional_call(m, state, qx)
        fq = m.backbone(qx)
        if self.name == "ProtoNet":
            return -torch.cdist(fq, state)
        if self.name == "MatchingNet":
            fs, sy = state
            log_attn = F.log_softmax(m.scale * torch.mm(F.normalize(fq, dim=-1), fs.t()), dim=-1)
            in_class = F.one_hot(sy, num_classes=self.n_way).t().bool()
            masked = log_attn.unsqueeze(1).masked_fill(~in_class.unsqueeze(0), float("-inf"))
            return torch.logsumexp(masked, dim=-1)
        Q, D = fq.shape                                    # RelationNet
        pairs = torch.cat([fq.unsqueeze(1).expand(Q, self.n_way, D),
                           state.unsqueeze(0).expand(Q, self.n_way, D)], dim=-1)
        return m.relation_head(pairs.view(-1, 2 * D)).view(Q, self.n_way)

    @torch.no_grad()
    def check(self, sx, sy, qx):
        """max |split adapt/predict - forward()| (None for FOMAML: same calls as its evaluation)."""
        if self.name == "MAML":
            return None
        ref = self.model(sx, sy, qx, self.n_way)
        return (self.predict(self.adapt(sx, sy), qx) - ref).abs().max().item()


class NCARunner:
    def __init__(self, variant, device):
        self.name = variant
        # weights are frozen at test time (grad_on_s differentiates w.r.t. S only)
        self.model = L.build_learner(variant).to(device).eval().requires_grad_(False)
        spec = L.VARIANTS[variant]
        self.K, self.N = L.test_time_schedule(spec)
        if spec.updater == "gradient":
            self.test_grads = f"Yes ({self.K} steps)"
        else:
            self.test_grads = "No"
        self.schedule = f"K={self.K}, N={self.N}" if spec.updater != "prototype" else f"N={self.N}"

    def adapt(self, sx, sy):
        return self.model.adapt(sx, sy, 1, self.K, self.N, meta_train=False)

    def predict(self, state, qx):
        return self.model.predict(qx, state, self.N)

    def check(self, *a):
        return None


# ----------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------
def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def count_flops(fn):
    """FLOPs with one multiply-add = 2 FLOPs (torch.utils.flop_counter's convention)."""
    with FlopCounterMode(display=False) as fc:
        fn()
    return float(fc.get_total_flops())


def time_ms(fn, device, repeats, warmup, budget):
    for _ in range(warmup):
        fn()
    times, start = [], time.perf_counter()
    for i in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
        if i >= 2 and time.perf_counter() - start > budget:
            break
    return statistics.median(times), len(times)


def peak_mb(fn, device):
    if torch.device(device).type != "cuda":
        return float("nan")
    sync(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = fn()
    sync(device)
    peak = torch.cuda.max_memory_allocated(device) - base
    del out
    return peak / 2**20


def make_task(n_way, k_shot, n_query, device, gen):
    sx = torch.rand(n_way * k_shot, 1, L.GRID, L.GRID, generator=gen).mul(2).sub(1).to(device)
    sy = torch.arange(n_way).repeat_interleave(k_shot).to(device)
    qx = torch.rand(n_query, 1, L.GRID, L.GRID, generator=gen).mul(2).sub(1).to(device)
    return sx, sy, qx


def sig(x, n=3):
    """n significant digits, no scientific notation (e.g. 494, 12.3, 0.00982)."""
    if x is None or x != x:
        return "–"
    if x == 0:
        return "0"
    d = max(0, n - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:,.{d}f}"


def device_name(device):
    if torch.device(device).type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.literal_nca:
        L.BaseNCA.PRECOMPUTE_STATIC = False
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    dev = args.device
    unknown = [m for m in args.models if m not in BASELINES and m not in L.VARIANTS]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; choose from {BASELINES + list(L.VARIANTS)}")

    mod = load_baseline_module() if any(m in BASELINES for m in args.models) else None
    runners = [BaselineRunner(mod, m, args.n_way, dev) if m in BASELINES else NCARunner(m, dev)
               for m in args.models]
    dname = device_name(dev)
    precision = "fp32" if amp_dtype is None else args.amp
    print(f"Device: {dname} | precision: {precision} | {args.n_way}-way | query batch {args.query_batch}"
          f" | NCA static-term precomputation: {'off (literal)' if args.literal_nca else 'on (exact)'}")

    gen = torch.Generator().manual_seed(args.seed)
    sx, sy, qx = make_task(args.n_way, min(args.shots), args.query_batch, dev, gen)
    for r in runners:
        err = r.check(sx, sy, qx)
        if err is not None:
            status = "OK" if err < 1e-3 else "MISMATCH"
            print(f"[check] {r.name}: split adapt/predict vs forward(), max |diff| = {err:.2e}  {status}")
            if err >= 1e-3:
                sys.exit("split evaluation does not reproduce the model's forward(); not timing it")

    rows = []
    for k in args.shots:
        sx, sy, qx = make_task(args.n_way, k, args.query_batch, dev, gen)
        for r in runners:
            def adapt():
                with torch.no_grad(), L.amp_context(dev, amp_dtype):
                    return r.adapt(sx, sy)
            state = adapt()

            def query():
                with torch.no_grad(), L.amp_context(dev, amp_dtype):
                    return r.predict(state, qx)

            a_flops, q_flops = count_flops(adapt), count_flops(query)
            a_ms, a_n = time_ms(adapt, dev, args.repeats, args.warmup, args.time_budget)
            q_ms, q_n = time_ms(query, dev, args.repeats, args.warmup, args.time_budget)
            a_mb, q_mb = peak_mb(adapt, dev), peak_mb(query, dev)
            Q = args.query_batch
            row = dict(model=r.name, label=LABELS.get(r.name, r.name), shot=k, n_way=args.n_way,
                       n_support=sx.shape[0], query_batch=Q, params=L.count_params(r.model),
                       test_time_grads=r.test_grads, schedule=r.schedule,
                       adapt_gflops=a_flops / 1e9, adapt_ms=a_ms, adapt_peak_mb=a_mb,
                       query_mflops_per_img=q_flops / Q / 1e6, query_ms_per_img=q_ms / Q,
                       query_peak_mb=q_mb,
                       task_gflops=(a_flops + q_flops) / 1e9, task_ms=a_ms + q_ms,
                       timed_runs=f"{a_n}/{q_n}", device=dname, precision=precision,
                       nca_precompute=not args.literal_nca)
            rows.append(row)
            print(f"{k:>2}-shot {row['label']:<24} adapt {sig(row['adapt_gflops'])} GFLOPs "
                  f"{sig(a_ms)} ms {sig(a_mb)} MB | query/img {sig(row['query_mflops_per_img'])} MFLOPs "
                  f"{sig(row['query_ms_per_img'])} ms | {Q} queries: {sig(q_mb)} MB")
            del state
    write_outputs(rows, args, dname, precision)


# ----------------------------------------------------------------------
def write_outputs(rows, args, dname, precision):
    out = args.out_dir
    with open(os.path.join(out, "cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = {(r["model"], r["shot"]): r for r in rows}
    models = list(dict.fromkeys(r["model"] for r in rows))
    shots = sorted({r["shot"] for r in rows})
    Q = args.query_batch
    md = [f"# Test-time cost ({args.n_way}-way)", "",
          f"Device: {dname}; precision: {precision}; query numbers per image, measured on batches of {Q}; "
          f"wall-clock = median of up to {args.repeats} runs; peak memory = working memory above "
          "weights + inputs (GPU only). FLOPs count matrix multiplications and convolutions "
          "(incl. backward passes for test-time gradients), one multiply-add = 2 FLOPs. "
          "Task = adaptation + "
          f"{Q} query images.", ""]
    md += ["| Shot | Model | Test-time grads | Params | Adapt GFLOPs | Adapt ms | Adapt MB "
           "| Query MFLOPs/img | Query ms/img | Query MB (batch) | Task GFLOPs | Task ms |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in shots:
        for m in models:
            r = by.get((m, k))
            if r is None:
                continue
            md.append(f"| {k} | {r['label']} | {r['test_time_grads']} | {r['params']:,} | "
                      f"{sig(r['adapt_gflops'])} | {sig(r['adapt_ms'])} | {sig(r['adapt_peak_mb'])} | "
                      f"{sig(r['query_mflops_per_img'])} | {sig(r['query_ms_per_img'])} | "
                      f"{sig(r['query_peak_mb'])} | {sig(r['task_gflops'])} | {sig(r['task_ms'])} |")

    refs = [m for m in ("ProtoNet", "MAML") if m in models]
    others = [m for m in models if m not in BASELINES]
    if refs and others:
        md += ["", "## Relative cost (model / reference)", "",
               "| Shot | Model | vs | Adapt FLOPs | Adapt time | Query FLOPs | Query time | Task FLOPs | Task time |",
               "|---|---|---|---|---|---|---|---|---|"]
        for k in shots:
            for m in others:
                for ref in refs:
                    a, b = by.get((m, k)), by.get((ref, k))
                    if a is None or b is None:
                        continue
                    ratio = lambda c: sig(a[c] / b[c]) + "×" if b[c] else "–"
                    md.append(f"| {k} | {a['label']} | {b['label']} | {ratio('adapt_gflops')} | "
                              f"{ratio('adapt_ms')} | {ratio('query_mflops_per_img')} | "
                              f"{ratio('query_ms_per_img')} | {ratio('task_gflops')} | {ratio('task_ms')} |")
    open(os.path.join(out, "cost.md"), "w").write("\n".join(md) + "\n")

    k = args.latex_shot or max(shots)
    tex = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}lcrrrrrr@{}}", r"\toprule",
           r" & & & \multicolumn{3}{c}{Adaptation (per task)} & \multicolumn{2}{c}{Query (per image)} \\",
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
           r"Model & Test-time grads & Params & GFLOPs & ms & MB & MFLOPs & ms \\", r"\midrule"]
    for m in models:
        r = by.get((m, k))
        if r is None:
            continue
        tex.append(f"{r['label']} & {r['test_time_grads']} & {r['params'] / 1e3:.0f}k & "
                   f"{sig(r['adapt_gflops'])} & {sig(r['adapt_ms'])} & {sig(r['adapt_peak_mb'])} & "
                   f"{sig(r['query_mflops_per_img'])} & {sig(r['query_ms_per_img'])} \\\\")
    n_supp = by[(models[0], k)]["n_support"]
    tex += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{Test-time cost of one {args.n_way}-way {k}-shot task ({n_supp} support images) on "
            rf"{dname.replace('_', ' ')} ({precision}). Adaptation: support set to task state; query: "
            rf"per image, measured on batches of {Q}. FLOPs of convolutions and matrix multiplications "
            r"(one multiply-add = 2 FLOPs; including backward passes for test-time gradients); "
            r"wall-clock: median over repeated "
            r"runs; MB: peak working memory above weights and inputs.}",
            r"\label{tab:cost}", r"\end{table}"]
    open(os.path.join(out, "cost.tex"), "w").write("\n".join(tex).replace("–", "--") + "\n")
    print("\n" + "\n".join(md))
    print(f"\nWrote {out}/cost.csv, cost.md, cost.tex")


if __name__ == "__main__":
    main()
