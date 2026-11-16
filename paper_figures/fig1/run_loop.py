"""Run one released checkpoint on one 10-way 1-shot Omniglot task and save every intermediate
tensor of the interaction loop (same math as ablation_lib.ProgramLearner.adapt / predict).

Usage (from the repository root): python paper_figures/fig1/run_loop.py <ckpt seed> <eval seed> <out.npz>
"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "ablations")
import ablation_lib as L

seed_ckpt, eval_seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
learner = L.load_original_checkpoint(f"checkpoints/active_nca_seed{seed_ckpt}.pth",
                                     f"checkpoints/meta_nca_seed{seed_ckpt}.pth").eval()
d = torch.load("data/ablation_cache/eval_omniglot.pt", weights_only=False)
imgs, tg = d["images"], torch.as_tensor(d["targets"])
split = L.make_task_split(tg, 10, 1, eval_seed)
sx, sy = imgs[split.supp_idx], torch.as_tensor(split.supp_y)
tx, ty = imgs[split.test_idx], torch.as_tensor(split.test_y)
K, N = 5, 19
torch.manual_seed(eval_seed)
with torch.no_grad():
    mask = learner.alive_mask(sx)
    y_sp = L.spatial_onehot(sy, 28, 28)
    S = learner.init_program(1, 28, 28, "cpu")
    rec = dict(S=[S[0].numpy()], P=[], E=[], acc=[], qprob=[])
    D = None

    def score(S):
        sc = torch.cat([learner.predict(tx[i:i + 256], S, N) for i in range(0, len(tx), 256)])
        return (sc.argmax(1) == ty).float().mean().item() * 100

    rec["acc"].append(score(S))
    for k in range(1, K + 1):
        dyn = learner.active(sx, L.broadcast_program(S, sx.shape[0]), steps=2 * N)
        probs = F.softmax(dyn[:, :10], dim=1)
        E = (y_sp - probs) * mask
        D = learner.meta(E, D, steps=N)
        S = learner.program_from_meta(D, 1)
        rec["P"].append(probs.numpy()); rec["E"].append(E.numpy()); rec["S"].append(S[0].numpy())
        rec["acc"].append(score(S))
    # one query image: logits map and class scores with the final program
    qi = int(np.nonzero((ty == sy[2]).numpy())[0][0])
    qx = tx[qi:qi + 1]
    qlog_k = np.stack([learner.active(qx, torch.as_tensor(Sk)[None], steps=2 * N)[:, :10].numpy()[0]
                       for Sk in rec["S"]])
    qdyn = learner.active(qx, S, steps=2 * N)
    qlog = qdyn[:, :10]
    qscore = L.masked_readout(qlog.float(), learner.alive_mask(qx))
np.savez(out, sx=sx.numpy(), sy=sy.numpy(), mask=mask.numpy(), S=np.stack(rec["S"]),
         P=np.stack(rec["P"]), E=np.stack(rec["E"]), acc=np.array(rec["acc"]),
         qx=qx.numpy(), qy=int(ty[qi]), qlog=qlog.numpy(), qprob=F.softmax(qlog, 1).numpy(),
         qscore=qscore.numpy(), qlog_k=qlog_k, qmask=learner.alive_mask(qx).numpy())
print("acc per k:", np.round(rec["acc"], 1), "| query true", int(ty[qi]), "pred", int(qscore.argmax()))
