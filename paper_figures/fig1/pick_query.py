"""Choose the query image shown in Figure 1 (b) and (c) for a loop recorded by run_loop.py, without
re-scoring the task's test set (the slow part of run_loop.py).

The query is a test image of the class of the front support image. By default it is the first such
image; with --correct it is the first one the adapted model classifies correctly (the accuracy row of
panel (c) is unchanged and still covers the whole test set). The query maps of every round, including
the final one used in panel (b), come from one run with a fixed seed. With --digits the class-score
bars are labelled with the dataset's class ids (the digits, for MNIST/USPS) instead of task indices.

Usage (from the repository root):
  python paper_figures/fig1/pick_query.py <in.npz> <out.npz> <ckpt seed> <eval seed> <dataset> [--correct] [--digits]
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "ablations")
import ablation_lib as L  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("inp"); p.add_argument("out"); p.add_argument("ckpt_seed")
p.add_argument("eval_seed", type=int); p.add_argument("dataset")
p.add_argument("--correct", action="store_true")
p.add_argument("--digits", action="store_true")
args = p.parse_args()

z = dict(np.load(args.inp))
learner = L.load_original_checkpoint(f"checkpoints/active_nca_seed{args.ckpt_seed}.pth",
                                     f"checkpoints/meta_nca_seed{args.ckpt_seed}.pth").eval()
d = torch.load(f"data/ablation_cache/eval_{args.dataset}.pt", weights_only=False)
imgs, tg = d["images"], torch.as_tensor(d["targets"])
split = L.make_task_split(tg, 10, 1, args.eval_seed)
assert np.array_equal(split.supp_y, z["sy"]), "npz was recorded on a different task"
tx, ty = imgs[split.test_idx], torch.as_tensor(split.test_y)
N = 19
cls = int(z["sy"][2])                      # class of the front support image in panel (a)
cand = np.nonzero((ty == cls).numpy())[0]
S_final = torch.as_tensor(z["S"][-1])[None]

chosen, tried = int(cand[0]), 1
if args.correct:
    with torch.no_grad():
        for tried, i in enumerate(cand, 1):
            torch.manual_seed(args.eval_seed)
            if int(learner.predict(tx[i:i + 1], S_final, N).argmax()) == cls:
                chosen = int(i)
                break

qx = tx[chosen:chosen + 1]
with torch.no_grad():
    torch.manual_seed(args.eval_seed)
    qlog_k = np.stack([learner.active(qx, torch.as_tensor(Sk)[None], steps=2 * N)[:, :10].numpy()[0]
                       for Sk in z["S"]])
    qmask = learner.alive_mask(qx)
    qscore = L.masked_readout(torch.as_tensor(qlog_k[-1:]), qmask)
z.update(qx=qx.numpy(), qy=cls, qlog=qlog_k[-1:], qlog_k=qlog_k, qmask=qmask.numpy(), qscore=qscore.numpy())
if args.digits:
    z["tick_labels"] = np.array([str(c) for c in split.classes])
np.savez(args.out, **z)
print(f"query: test image {chosen} (candidate {tried} of {len(cand)} of class {cls}); "
      f"predicted {int(qscore.argmax())}" + (f"; classes = {split.classes}" if args.digits else ""))
