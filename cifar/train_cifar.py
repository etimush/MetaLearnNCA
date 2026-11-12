"""
Meta-train one method on FC100 (CIFAR-100, superclass split) with the shared regime.

Every method gets exactly the same training (cifar_lib.REGIME): 5-way episodes from the 60
meta-training classes, 1- or 5-shot (drawn per pass), 10 queries per class, 16 tasks per update
(2 per pass x 8 passes), 4000 updates, AdamW lr 2e-3 cosine-annealed to 5e-4, weight decay 1e-5,
gradient-norm clip 1.0, bf16 autocast where supported (fp32 otherwise), no data augmentation.
Model sizes are matched to one parameter budget (cifar_lib.matched_size). The NCAs use no
foreground mask.

Robustness (same as ablations/train_ablation.py): a pass with a non-finite loss is skipped; after
10 consecutive skipped updates, or non-finite weights, the run rolls back to its last good state
(in-memory snapshot every 25 updates) with a new deterministic random stream; after 10 roll-backs
it is marked diverged. On CUDA out-of-memory the pass size is halved (same tasks per update).
Resumable (latest.pt every 250 updates, written atomically).

Outputs in <out_dir>/<method>/seed<seed>/: config.json, train_log.csv, status.json, model.pt, latest.pt

Examples
  python cifar/train_cifar.py --method metalearnnca --seed 0
  python cifar/train_cifar.py --method protonet --seed 0 --smoke --fake_data     # CPU pipeline check
"""
import argparse
import collections
import copy
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar_lib as C  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=C.METHODS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", default="cifar_runs")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "none"],
                   help="auto: bf16 where supported, else fp32 (fp16 can overflow in the NCA state)")
    p.add_argument("--compile", default="auto", choices=["auto", "on", "off"],
                   help="torch.compile the NCA update (auto: on for CUDA)")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--nan_patience", type=int, default=10)
    p.add_argument("--snapshot_every", type=int, default=25)
    p.add_argument("--max_rewinds", type=int, default=10)
    p.add_argument("--val_every", type=int, default=C.REGIME["val_every"], help="0 disables validation")
    p.add_argument("--val_episodes", type=int, default=C.REGIME["val_episodes"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fake_data", action="store_true", help="synthetic textures instead of CIFAR-100")
    p.add_argument("--smoke", action="store_true", help="4 updates, 2 passes, short NCA unrolls")
    return p.parse_args()


def resolve_amp(arg, device):
    if arg == "none" or device.type != "cuda":
        return torch.bfloat16 if (arg == "bf16" and device.type == "cpu") else None
    if arg == "bf16" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def main():
    args = parse_args()
    device = torch.device(args.device)
    R = dict(C.REGIME)
    updates, micro = R["updates"], R["micro_batch"]
    accum = R["tasks_per_update"] // micro
    if args.smoke:
        updates, micro, accum = 4, 1, 2
        args.log_every, args.ckpt_every = 1, 2
        args.val_every, args.val_episodes = 2, 4

    run_dir = os.path.join(args.out_dir, args.method, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    status_path = os.path.join(run_dir, "status.json")
    if os.path.exists(status_path) and not args.overwrite:
        st = json.load(open(status_path))
        if st.get("status") in ("ok", "diverged"):
            print(f"[skip] {run_dir} already finished with status '{st['status']}' (use --overwrite)")
            return

    C.set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    size, n_params, target = C.matched_size(args.method, R["n_way"])
    learner = C.build_learner(args.method, R["n_way"], size).to(device)
    if args.smoke and hasattr(learner, "n_steps"):
        learner.n_steps = 2
        if hasattr(learner, "k_repeats"):
            learner.k_repeats = 1
    amp_dtype = resolve_amp(args.amp, device)

    cfg_path = os.path.join(run_dir, "config.json")
    if args.resume and os.path.exists(cfg_path) and os.path.exists(os.path.join(run_dir, "latest.pt")):
        prev = json.load(open(cfg_path)).get("amp")
        names = {"torch.bfloat16": torch.bfloat16, "None": None}
        if prev in names and names[prev] != amp_dtype:
            print(f"[info] resuming with the run's original precision {prev}")
            amp_dtype = names[prev]

    is_nca = args.method in ("metalearnnca", "metalearnnca_img", "nca_protonet")
    use_compile = is_nca and (args.compile == "on" or (args.compile == "auto" and device.type == "cuda"))
    if use_compile:
        for nca in (learner.active, getattr(learner, "meta", None)):
            if nca is not None:
                nca.dyn_update = torch.compile(nca.dyn_update)

    config = dict(method=args.method, seed=args.seed, size=list(size), n_params=n_params, param_target=target,
                  regime=R, nca_steps=C.NCA_STEPS if is_nca else None,
                  fomaml_inner=C.FOMAML_INNER if args.method == "fomaml" else None,
                  updates=updates, micro_batch=micro, accum=accum, amp=str(amp_dtype), compile=use_compile,
                  fake_data=args.fake_data, smoke=args.smoke, device=str(device), torch_version=torch.__version__,
                  split="FC100 (superclass-disjoint)", foreground_mask=False)
    with open(cfg_path + ".tmp", "w") as f:
        json.dump(config, f, indent=2)
    os.replace(cfg_path + ".tmp", cfg_path)

    banks = C.make_fake_banks(seed=0) if args.fake_data else C.load_fc100(args.data_root)
    bank = banks["train"].to(device)                     # uint8 on the device
    print(f"{args.method} seed {args.seed}: {n_params:,} parameters (target {target:,}) | train bank "
          f"{tuple(bank.shape)} | {updates} updates x {micro * accum} tasks | amp {amp_dtype}", flush=True)

    # fixed validation episodes (validation classes = disjoint superclasses; never the test classes)
    val_sets = {}
    if args.val_every > 0:
        for shot in R["test_shots"]:
            val_sets[shot] = C.fixed_episodes(banks["val"], args.val_episodes, R["n_way"], shot, R["test_query"],
                                              seed=R["episode_seed"] + 1)

    def validate():
        """Mean accuracy on the validation episodes; leaves the training random streams untouched."""
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        learner.eval()
        out = {}
        for shot, (sx, sy, qx, qy) in val_sets.items():
            hits = []
            for i in range(0, sx.shape[0], 8):
                torch.manual_seed(12345 + i)
                sc = learner.predict(sx[i:i + 8].to(device), sy[i:i + 8].to(device), qx[i:i + 8].to(device))
                hits.append((sc.argmax(-1).cpu() == qy[i:i + 8]).float().mean(1))
            out[shot] = 100.0 * torch.cat(hits).mean().item()
        learner.train()
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state(rng[3], device)
        return out

    params = [p for p in learner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=R["lr"], weight_decay=R["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=R["lr_min"])

    start, skipped_steps, skipped_passes = 1, 0, 0
    lr_scale, best_val, best_update, spike_skips = 1.0, -1.0, None, 0
    val_log = os.path.join(run_dir, "val_log.csv")
    if not (args.resume and os.path.exists(os.path.join(run_dir, "latest.pt"))) and args.val_every > 0:
        with open(val_log, "w", newline="") as f:
            csv.writer(f).writerow(["update"] + [f"val_{s}shot" for s in R["test_shots"]] + ["mean", "best"])
    latest_path = os.path.join(run_dir, "latest.pt")
    log_path = os.path.join(run_dir, "train_log.csv")
    header = ["update", "loss", "acc", "lr", "grad_norm", "sec_per_update", "skipped_steps", "skipped_passes",
              "rewinds"]
    if args.resume and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location="cpu", weights_only=False)
        learner.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start = ck["update"] + 1
        skipped_steps, skipped_passes = ck.get("skipped_steps", 0), ck.get("skipped_passes", 0)
        lr_scale, best_val = ck.get("lr_scale", 1.0), ck.get("best_val", -1.0)
        best_update, spike_skips = ck.get("best_update", None), ck.get("spike_skips", 0)
        random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"].cpu())
        if device.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu(), device)
        print(f"Resumed from update {ck['update']}", flush=True)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)
    with open(status_path, "w") as f:
        json.dump(dict(status="running", update=start - 1), f)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def state():
        return copy.deepcopy(dict(model=learner.state_dict(), optimizer=optimizer.state_dict(),
                                  scheduler=scheduler.state_dict(), lr_scale=lr_scale))

    recent = collections.deque(maxlen=R["spike_window"])     # losses of recent good passes

    snapshot, snapshot_update = state(), start - 1
    rewinds, oom_fallbacks, bad_streak, status = [], [], 0, "ok"
    t0 = time.time()
    u = start
    learner.train()
    while u <= updates:
        t_up = time.time()
        while True:
            optimizer.zero_grad(set_to_none=True)
            loss_sum = acc_sum = 0.0
            good, bad, oom = 0, 0, False
            try:
                for _ in range(accum):
                    shot = random.choice(R["train_shots"])
                    batch = C.sample_tasks(bank, micro, R["n_way"], shot, R["train_query"])
                    with C.amp_context(device, amp_dtype):
                        loss, acc = learner.train_loss(*batch)
                    lv, av = torch.stack([loss.detach().float(), acc.float()]).tolist()
                    spike = (len(recent) >= 50 and lv > R["spike_floor"]
                             and lv > R["spike_factor"] * float(np.median(recent)))
                    if not math.isfinite(lv) or spike:
                        # skip this pass: a non-finite or exploded loss would wreck the update
                        bad += 1
                        spike_skips += int(spike)
                        continue
                    (loss / accum).backward()
                    recent.append(lv)
                    loss_sum, acc_sum, good = loss_sum + lv, acc_sum + av, good + 1
            except torch.cuda.OutOfMemoryError:
                oom = True
            if not oom:
                break
            loss = acc = batch = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro == 1:
                raise RuntimeError("out of GPU memory even with 1 task per pass")
            tasks = micro * accum
            micro = max(1, micro // 2)
            accum = math.ceil(tasks / micro)
            oom_fallbacks.append(dict(update=u, micro_batch=micro, accum=accum))
            print(f"[oom] {args.method} seed {args.seed}: retrying with {micro} task(s) x {accum} passes", flush=True)
        skipped_passes += bad

        stepped, gn = False, torch.tensor(float("nan"))
        if good:
            gn = torch.nn.utils.clip_grad_norm_(params, R["grad_clip"])
            if torch.isfinite(gn):
                optimizer.step()
                stepped = True
        skipped_steps += 0 if stepped else 1
        scheduler.step()

        weights_ok = bool(torch.stack([torch.isfinite(p).all() for p in params]).all())
        bad_streak = 0 if stepped else bad_streak + 1
        if bad_streak >= args.nan_patience or not weights_ok:
            if len(rewinds) >= args.max_rewinds:
                status = "diverged"
                print(f"[diverged] {args.method} seed {args.seed} at update {u}", flush=True)
                break
            learner.load_state_dict(snapshot["model"])
            optimizer.load_state_dict(snapshot["optimizer"])
            scheduler.load_state_dict(snapshot["scheduler"])
            # halve the learning rate (relative to the snapshot's own scale) for the rest of the run
            new_scale = lr_scale * R["rewind_lr_factor"]
            ratio = new_scale / snapshot["lr_scale"]
            for g in optimizer.param_groups:
                g["lr"] *= ratio
            scheduler.base_lrs = [b * ratio for b in scheduler.base_lrs]
            scheduler.eta_min *= ratio
            lr_scale = new_scale
            recent.clear()
            rewinds.append(dict(at_update=u, back_to=snapshot_update, lr_scale=lr_scale,
                                reason="non-finite weights" if not weights_ok else f"{bad_streak} skipped updates"))
            reseed = args.seed * 1000 + len(rewinds)
            random.seed(reseed); np.random.seed(reseed); torch.manual_seed(reseed)
            print(f"[rewind] {args.method} seed {args.seed}: {rewinds[-1]['reason']} at update {u}; back to "
                  f"{snapshot_update}, learning rate x{lr_scale:g} ({len(rewinds)}/{args.max_rewinds})", flush=True)
            bad_streak, u = 0, snapshot_update + 1
            continue
        if stepped and u % args.snapshot_every == 0:
            snapshot, snapshot_update = state(), u

        if u % args.log_every == 0 or u == 1:
            sec = time.time() - t_up
            n = max(1, good)
            row = [u, loss_sum / n if good else float("nan"), 100 * acc_sum / n if good else float("nan"),
                   scheduler.get_last_lr()[0], float(gn), sec, skipped_steps, skipped_passes, len(rewinds)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            eta = (updates - u) * (time.time() - t0) / max(1, u - start + 1) / 60
            extra = (f" | skipped passes {skipped_passes}" if skipped_passes else "") + \
                    (f" | rewinds {len(rewinds)}" if rewinds else "")
            print(f"[{args.method} s{args.seed}] up {u:5d}/{updates} | loss {row[1]:.4f} | acc {row[2]:5.1f}% | "
                  f"gn {row[4]:.3f} | {sec:.2f}s/up | ETA {eta:.0f} min{extra}", flush=True)

        if args.val_every > 0 and stepped and (u % args.val_every == 0 or u == updates):
            va = validate()
            score = float(np.mean(list(va.values())))
            improved = score > best_val
            if improved:
                best_val, best_update = score, u
                best_path = os.path.join(run_dir, "best.pt")
                torch.save(dict(method=args.method, size=list(size), n_way=R["n_way"], n_params=n_params,
                                update=u, val=va, state_dict={k: v.detach().cpu() for k, v in
                                                              learner.state_dict().items()}), best_path + ".tmp")
                os.replace(best_path + ".tmp", best_path)
            with open(val_log, "a", newline="") as f:
                csv.writer(f).writerow([u] + [f"{va[s]:.2f}" for s in R["test_shots"]] + [f"{score:.2f}", int(improved)])
            print(f"[val] {args.method} s{args.seed} up {u}: " + ", ".join(f"{s}-shot {va[s]:.1f}%" for s in va)
                  + f" | best {best_val:.1f}% @ {best_update}" + (" (new best)" if improved else ""), flush=True)

        if stepped and u % args.ckpt_every == 0:
            torch.save(dict(model=learner.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), update=u, skipped_steps=skipped_steps,
                            skipped_passes=skipped_passes, lr_scale=lr_scale, best_val=best_val,
                            best_update=best_update, spike_skips=spike_skips,
                            rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                     torch=torch.get_rng_state(),
                                     cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)),
                       latest_path + ".tmp")
            os.replace(latest_path + ".tmp", latest_path)
            with open(status_path, "w") as f:
                json.dump(dict(status="running", update=u), f)
        u += 1
    u = min(u, updates)

    model_path = os.path.join(run_dir, "model.pt")
    torch.save(dict(method=args.method, size=list(size), n_way=R["n_way"], n_params=n_params, status=status,
                    update=u, state_dict={k: v.detach().cpu() for k, v in learner.state_dict().items()}),
               model_path + ".tmp")
    os.replace(model_path + ".tmp", model_path)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    with open(status_path + ".tmp", "w") as f:
        json.dump(dict(status=status, updates_completed=u, updates_planned=updates, wall_time_s=time.time() - t0,
                       skipped_steps=skipped_steps, skipped_passes=skipped_passes, rewinds=rewinds,
                       spike_skips=spike_skips, lr_scale_final=lr_scale, best_update=best_update,
                       best_val=best_val,
                       oom_fallbacks=oom_fallbacks, amp=str(amp_dtype), peak_train_mem_mb=peak,
                       n_params=n_params), f, indent=2)
    os.replace(status_path + ".tmp", status_path)
    print(f"Finished {args.method} seed {args.seed}: {status} after {u} updates "
          f"({(time.time() - t0) / 60:.1f} min) -> {run_dir}", flush=True)


def load_model(run_dir, device="cpu", which="best"):
    """Rebuild a trained learner from <run_dir>/best.pt (best validation accuracy) or model.pt (final)."""
    path = os.path.join(run_dir, "best.pt")
    if which != "best" or not os.path.exists(path):
        path = os.path.join(run_dir, "model.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["checkpoint_file"] = os.path.basename(path)
    learner = C.build_learner(ck["method"], ck["n_way"], tuple(ck["size"]))
    learner.load_state_dict(ck["state_dict"])
    return learner.to(device).eval(), ck


if __name__ == "__main__":
    main()
