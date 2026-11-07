"""
Meta-train one ablation variant for one seed on Omniglot (background set).

The training protocol is that of ML-NCA-Faster-trainign.py (the script that
produced checkpoints/): 10-way 1-shot 5-query episodes, 2 tasks x 8
accumulation passes per AdamW step, lr 2e-3 cosine-annealed to 5e-4,
weight decay 1e-5, global grad-norm clip 1.0, fp16 autocast, step jitter
N' ~ U{N..N+3}, K' ~ U{K..K+2}. Only the component named by --variant changes.

Differences from the original script (all deliberate, all logged):
  * A non-finite loss no longer silently discards the run. Optimiser steps with
    non-finite gradients are skipped; the run is marked "diverged" only after
    --nan_patience consecutive non-finite epochs, and that status is saved, so
    divergence rates can be reported per variant.
  * The logged loss is the mean over all accumulation passes (the original
    logged only the last pass).
  * Resumable: latest.pt is written every --ckpt_every epochs.

Outputs (in <out_dir>/<variant>/seed<seed>/):
  config.json, train_log.csv, status.json, model.pt, latest.pt
  + active_nca.pth / meta_nca.pth for variants whose networks are
    state-dict compatible with the original evaluation scripts.

Examples
--------
  python ablations/train_ablation.py --variant global_program --seed 0
  python ablations/train_ablation.py --variant grad_on_s --seed 1 --micro_batch 1 --accum 16
  python ablations/train_ablation.py --variant full --seed 0 --smoke --fake_data   # CPU check
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ablation_lib as L  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=list(L.VARIANTS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", default="ablation_runs")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    # protocol (defaults = ML-NCA-Faster-trainign.py)
    d = L.TRAIN_DEFAULTS
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--micro_batch", type=int, default=d["micro_batch"])
    p.add_argument("--accum", type=int, default=d["accum"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--lr_min", type=float, default=d["lr_min"])
    p.add_argument("--weight_decay", type=float, default=d["weight_decay"])
    p.add_argument("--grad_clip", type=float, default=d["grad_clip"])
    p.add_argument("--n_steps", type=int, default=None, help="override base N of the variant")
    p.add_argument("--k_repeats", type=int, default=None, help="override base K of the variant")
    p.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "none"],
                   help="auto: bf16 on GPUs that support it (Ampere or newer), otherwise fp16 (fp32 for "
                        "grad_on_s). The original script used fp16, whose range the NCA state can exceed "
                        "(NaN losses); bf16 has the range of fp32 at the same speed.")
    p.add_argument("--compile", default="auto", choices=["auto", "on", "off"],
                   help="torch.compile the NCA update (auto: on for CUDA, off for second-order grad_on_s)")
    # variant options
    p.add_argument("--hidden", type=int, default=L.DEFAULT_OPTS["hidden"])
    p.add_argument("--inner_lr_mode", default=L.DEFAULT_OPTS["inner_lr_mode"],
                   choices=["fixed", "per_channel", "per_element"])
    p.add_argument("--inner_lr", type=float, default=None)
    p.add_argument("--first_order", action="store_true", help="grad_on_s: FO-CAVIA (less memory)")
    p.add_argument("--s_init", default=L.DEFAULT_OPTS["s_init"], choices=["zeros", "uniform", "learned"])
    p.add_argument("--proto_embed", default=L.DEFAULT_OPTS["proto_embed"], choices=["pooled", "spatial"])
    p.add_argument("--proto_program", default=L.DEFAULT_OPTS["proto_program"], choices=["learned", "zeros"])
    # bookkeeping
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--nan_patience", type=int, default=10,
                   help="consecutive bad epochs (update skipped because of non-finite gradients) before "
                        "rolling back to the last good snapshot")
    p.add_argument("--snapshot_every", type=int, default=25,
                   help="keep an in-memory copy of the last good training state every N epochs")
    p.add_argument("--max_rewinds", type=int, default=10,
                   help="roll-backs allowed before the run is marked diverged")
    p.add_argument("--resume", action="store_true", help="resume from latest.pt if present")
    p.add_argument("--overwrite", action="store_true", help="retrain even if status.json says ok")
    p.add_argument("--fake_data", action="store_true", help="synthetic strokes (no download)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run for pipeline checks: 4 epochs, N=2, K=1, 1x2 tasks")
    return p.parse_args()


def resolve_amp(args, spec, device):
    if device.type != "cuda":
        return None if args.amp in ("auto", "none") else {"fp16": torch.float16, "bf16": torch.bfloat16}[args.amp]
    if args.amp == "none":
        return None
    if args.amp == "fp16":
        return torch.float16
    if args.amp == "bf16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16       # fp32 range: the NCA state cannot overflow as it does in fp16
    return None if spec.updater == "gradient" else torch.float16


def main():
    args = parse_args()
    spec = L.VARIANTS[args.variant]
    device = torch.device(args.device)

    n_base = args.n_steps if args.n_steps is not None else spec.n_steps
    k_base = args.k_repeats if args.k_repeats is not None else spec.k_repeats
    epochs, micro, accum = args.epochs, args.micro_batch, args.accum
    if args.smoke:
        epochs, micro, accum, n_base, k_base = 4, 1, 2, 2, 1
        args.log_every, args.ckpt_every = 1, 2

    run_dir = os.path.join(args.out_dir, args.variant, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    status_path = os.path.join(run_dir, "status.json")
    if os.path.exists(status_path) and not args.overwrite:
        st = json.load(open(status_path))
        if st.get("status") in ("ok", "diverged"):
            print(f"[skip] {run_dir} already finished with status '{st['status']}' (use --overwrite)")
            return

    opts = dict(L.DEFAULT_OPTS)
    opts.update(hidden=args.hidden, inner_lr_mode=args.inner_lr_mode, inner_lr=args.inner_lr,
                first_order=args.first_order, s_init=args.s_init,
                proto_embed=args.proto_embed, proto_program=args.proto_program)

    L.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    learner = L.build_learner(args.variant, opts).to(device)
    opts = learner.opts  # includes resolved defaults (e.g. inner_lr)
    amp_dtype = resolve_amp(args, spec, device)
    cfg_path = os.path.join(run_dir, "config.json")
    if args.resume and args.amp == "auto" and os.path.exists(cfg_path) \
            and os.path.exists(os.path.join(run_dir, "latest.pt")):
        # never switch precision in the middle of a run: keep the one it was started with
        prev = json.load(open(cfg_path)).get("amp")
        names = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16, "None": None}
        if prev in names and names[prev] != amp_dtype:
            print(f"[info] resuming with the run's original precision {prev} (current default: {amp_dtype})")
            amp_dtype = names[prev]

    second_order = spec.updater == "gradient" and not opts["first_order"]
    use_compile = (args.compile == "on") or (args.compile == "auto" and device.type == "cuda" and not second_order)
    if use_compile:
        try:
            for nca in (learner.active, getattr(learner, "meta", None)):
                if nca is not None:
                    nca.grid_update = torch.compile(nca.grid_update)
                    nca.dyn_update = torch.compile(nca.dyn_update)
        except Exception as e:  # pragma: no cover
            print(f"torch.compile skipped: {e}")
            use_compile = False

    config = dict(variant=args.variant, seed=args.seed, spec=asdict(spec), opts=opts,
                  n_steps=n_base, k_repeats=k_base, epochs=epochs, micro_batch=micro, accum=accum,
                  lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
                  grad_clip=args.grad_clip, amp=str(amp_dtype), compile=use_compile,
                  nan_patience=args.nan_patience, fake_data=args.fake_data, smoke=args.smoke,
                  n_params=L.count_params(learner), device=str(device),
                  torch_version=torch.__version__)
    L.write_json(os.path.join(run_dir, "config.json"), config)

    # ---------------- data ----------------
    if args.fake_data:
        bank = L.make_fake_train_bank(seed=args.seed)
    else:
        bank = L.load_omniglot_train_bank(args.data_root)
    bank = bank.to(device)
    print(f"Train bank: {tuple(bank.shape)} on {device}")
    print(f"Starting {args.variant} seed {args.seed}: {epochs} epochs, progress every {args.log_every} epochs"
          + (" (first epoch includes torch.compile, ~1-3 min)" if use_compile else ""), flush=True)

    # ---------------- optimisation ----------------
    params = [p for p in learner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr_min)
    use_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch, skipped_steps, skipped_passes = 1, 0, 0
    latest_path = os.path.join(run_dir, "latest.pt")
    log_path = os.path.join(run_dir, "train_log.csv")
    log_header = ["epoch", "loss", "acc", "lr", "grad_norm", "inner_lr_mean", "sec_per_epoch", "skipped_steps",
                  "skipped_passes", "rewinds"]
    if args.resume and os.path.exists(latest_path):
        # Load on the CPU: RNG states must stay CPU ByteTensors (map_location=cuda would move them to the
        # GPU and set_rng_state rejects that); model/optimizer state is copied to the GPU by load_state_dict.
        ck = torch.load(latest_path, map_location="cpu", weights_only=False)
        learner.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if use_scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        skipped_steps = ck.get("skipped_steps", 0)
        skipped_passes = ck.get("skipped_passes", 0)
        random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if device.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu(), device)
        print(f"Resumed from epoch {ck['epoch']}")
        with open(log_path) as f:                       # keep the column layout of an older log
            log_header = next(csv.reader(f), log_header)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_header)

    L.write_json(status_path, dict(status="running", epoch=start_epoch - 1))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def training_state():
        return copy.deepcopy(dict(model=learner.state_dict(), optimizer=optimizer.state_dict(),
                                  scheduler=scheduler.state_dict(), scaler=scaler.state_dict()))

    def params_finite():
        return bool(torch.stack([torch.isfinite(p).all() for p in params]).all())

    d = L.TRAIN_DEFAULTS
    t_start = time.time()
    status = "ok"
    oom_fallbacks, rewinds = [], []
    snapshot, snapshot_epoch = training_state(), start_epoch - 1   # always something to roll back to
    bad_streak = 0
    epoch = start_epoch
    while epoch <= epochs:
        t_ep = time.time()
        while True:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = acc_sum = 0.0
            good_passes, bad_passes, oom = 0, 0, False
            try:
                for _ in range(accum):
                    supp_x, supp_y, qry_x, qry_y = L.sample_meta_batch(bank, micro, d["n_way"], d["n_supp"], d["n_qry"])
                    N = random.randint(n_base, n_base + 3)
                    K = random.randint(k_base, k_base + 2)
                    with L.amp_context(device, amp_dtype):
                        loss, acc = learner.episode(supp_x, supp_y, qry_x, qry_y, micro, K, N)
                    lv, av = torch.stack([loss.detach().float(), acc.detach().float()]).tolist()
                    if not math.isfinite(lv):
                        # skip this pass only: backpropagating a non-finite loss would poison the
                        # gradient of the whole update
                        bad_passes += 1
                        continue
                    scaler.scale(loss / accum).backward()
                    loss_sum += lv
                    acc_sum += av
                    good_passes += 1
            except torch.cuda.OutOfMemoryError:
                oom = True
            if not oom:
                break
            # Out of GPU memory: halve the micro-batch and double the accumulation passes, so every
            # optimiser step still sees the same number of tasks, and redo this epoch.
            loss = acc = None
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if micro == 1:
                raise RuntimeError(f"{args.variant}: out of GPU memory even with --micro_batch 1. Free the GPU "
                                   "(fewer JOBS_PER_GPU) or, for grad_on_s, add --first_order and report it.")
            tasks = micro * accum
            micro = max(1, micro // 2)
            accum = math.ceil(tasks / micro)
            oom_fallbacks.append(dict(epoch=epoch, micro_batch=micro, accum=accum))
            print(f"[oom] {args.variant} seed {args.seed}: out of GPU memory at epoch {epoch}; retrying with "
                  f"--micro_batch {micro} --accum {accum} ({micro * accum} tasks per step)", flush=True)
        skipped_passes += bad_passes

        stepped = False
        gn = torch.tensor(float("nan"))
        if good_passes > 0:
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
            if use_scaler:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)          # skips internally on inf/nan grads
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
            elif torch.isfinite(gn):
                optimizer.step()
                stepped = True
        if not stepped:
            skipped_steps += 1
        scheduler.step()

        # ---- guard: roll back to the last good state if updates keep failing or weights broke ----
        weights_ok = params_finite()
        bad_streak = 0 if stepped else bad_streak + 1
        if bad_streak >= args.nan_patience or not weights_ok:
            if len(rewinds) >= args.max_rewinds:
                status = "diverged"
                print(f"[diverged] {args.variant} seed {args.seed}: still failing after {len(rewinds)} "
                      f"roll-backs (epoch {epoch})", flush=True)
                break
            learner.load_state_dict(snapshot["model"])
            optimizer.load_state_dict(snapshot["optimizer"])
            scheduler.load_state_dict(snapshot["scheduler"])
            scaler.load_state_dict(snapshot["scaler"])
            rewinds.append(dict(at_epoch=epoch, back_to_epoch=snapshot_epoch,
                                reason="non-finite weights" if not weights_ok else f"{bad_streak} skipped updates"))
            # a different (deterministic) random stream, so the same failing episodes are not replayed
            reseed = args.seed * 1000 + len(rewinds)
            random.seed(reseed); np.random.seed(reseed); torch.manual_seed(reseed)
            print(f"[rewind] {args.variant} seed {args.seed}: {rewinds[-1]['reason']} at epoch {epoch}; "
                  f"rolled back to epoch {snapshot_epoch} ({len(rewinds)}/{args.max_rewinds})", flush=True)
            bad_streak = 0
            epoch = snapshot_epoch + 1
            continue

        if stepped and epoch % args.snapshot_every == 0:
            snapshot, snapshot_epoch = training_state(), epoch

        if epoch % args.log_every == 0 or epoch == 1:
            sec = time.time() - t_ep
            ilr = float(learner.inner_lr().detach().mean()) if hasattr(learner, "log_inner_lr") else float("nan")
            n = max(1, good_passes)
            values = dict(epoch=epoch, loss=loss_sum / n if good_passes else float("nan"),
                          acc=100.0 * acc_sum / n if good_passes else float("nan"),
                          lr=scheduler.get_last_lr()[0], grad_norm=float(gn), inner_lr_mean=ilr,
                          sec_per_epoch=sec, skipped_steps=skipped_steps, skipped_passes=skipped_passes,
                          rewinds=len(rewinds))
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([values.get(c, "") for c in log_header])
            eta = (epochs - epoch) * (time.time() - t_start) / max(1, epoch - start_epoch + 1) / 60
            extra = f" | skipped passes {skipped_passes}" if skipped_passes else ""
            extra += f" | rewinds {len(rewinds)}" if rewinds else ""
            print(f"[{args.variant} s{args.seed}] ep {epoch:5d}/{epochs} | loss {values['loss']:.4f} | "
                  f"acc {values['acc']:5.1f}% | gn {values['grad_norm']:.3f} | {sec:.2f}s/ep | "
                  f"ETA {eta:.0f} min{extra}", flush=True)

        if epoch % args.ckpt_every == 0 and stepped:
            torch.save(dict(model=learner.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, skipped_steps=skipped_steps, skipped_passes=skipped_passes,
                            rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                     torch=torch.get_rng_state(),
                                     cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)),
                       latest_path)
            L.write_json(status_path, dict(status="running", epoch=epoch))
        epoch += 1
    epoch = min(epoch, epochs)

    wall = time.time() - t_start
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    L.save_run_checkpoint(os.path.join(run_dir, "model.pt"), learner, args.variant, opts,
                          extra=dict(status=status, epoch=epoch))
    if spec.updater in ("meta_nca", "label_only") and spec.updater_perception == "sobel" \
            and opts["hidden"] == 256:
        # drop-in files for ML-NCA-Testing.py & co.
        torch.save({k: v.cpu() for k, v in learner.active.state_dict().items()},
                   os.path.join(run_dir, "active_nca.pth"))
        torch.save({k: v.cpu() for k, v in learner.meta.state_dict().items()},
                   os.path.join(run_dir, "meta_nca.pth"))
    L.write_json(status_path, dict(status=status, epochs_completed=epoch, epochs_planned=epochs,
                                   wall_time_s=wall, skipped_steps=skipped_steps, skipped_passes=skipped_passes,
                                   rewinds=rewinds, amp=str(amp_dtype),
                                   peak_train_mem_mb=peak, n_params=L.count_params(learner),
                                   micro_batch_final=micro, accum_final=accum, oom_fallbacks=oom_fallbacks))
    print(f"Finished {args.variant} seed {args.seed}: {status} after {epoch} epochs "
          f"({wall / 60:.1f} min). -> {run_dir}")


if __name__ == "__main__":
    main()
