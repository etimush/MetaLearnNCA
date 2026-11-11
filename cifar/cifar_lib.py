"""
CIFAR-100 few-shot meta-learning (FC100 split) for MetaLearnNCA and the canonical baselines.

Goal: one experiment where every method gets
  * the same data and the same out-of-distribution test: FC100 splits CIFAR-100 by
    SUPERCLASS (Oreshkin et al., 2018; assignment from the authors' TADAM repository,
    datasets/create_dataset_cifar100.py), so the 20 test classes come from 4 superclasses
    never seen in meta-training (aquatic mammals, insects, medium mammals, people);
  * the same episodes, number of updates, optimiser, schedule and precision (REGIME);
  * the same parameter budget: each model's width / hidden size is chosen so that its
    parameter count matches a Conv4-64 ProtoNet on 3x32x32 images (113,088);
  * the same test episodes (deterministic, identical for every method and seed).

The NCAs use NO foreground mask: natural images have no background to mask out, so the
residual, the loss and the read-out use every pixel.

Methods: metalearnnca, nca_protonet, protonet, matchingnet, relationnet, fomaml.
"""
from __future__ import annotations

import functools
import math
import os
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH, IMG = 3, 32
PROGRAM_CH, DYN_CH = 24, 32

# ---------------------------------------------------------------------------------------------
# Shared regime (identical for every method; change it here, not per method)
# ---------------------------------------------------------------------------------------------
REGIME = dict(
    n_way=5,
    train_shots=(1, 5),        # each pass draws its shot from this set (one model serves 1- and 5-shot)
    train_query=10,            # query images per class during meta-training
    updates=4000,
    tasks_per_update=16,
    micro_batch=2,             # tasks per forward pass (2 x 8 passes = 16 tasks per update)
    lr=2e-3, lr_min=5e-4,      # AdamW, cosine annealing
    weight_decay=1e-5,
    grad_clip=1.0,
    test_shots=(1, 5),
    test_query=15,             # standard: 15 queries per class
    test_episodes=1000,
    episode_seed=2027,         # test / validation episodes are a fixed function of this seed
)
NCA_STEPS = dict(n_steps=16, k_repeats=3)       # jitter N' in [N, N+3], K' in [K, K+2]; test N+3, K+2
FOMAML_INNER = dict(steps=5, lr=0.01)           # training value; the test value is chosen on val classes

METHODS = ["metalearnnca", "metalearnnca_img", "nca_protonet", "protonet", "matchingnet", "relationnet", "fomaml"]
METHOD_LABELS = {"metalearnnca": "MetaLearnNCA", "metalearnnca_img": "MetaLearnNCA (image-conditioned updater)",
                 "nca_protonet": "NCA-ProtoNet", "protonet": "ProtoNet",
                 "matchingnet": "MatchingNet", "relationnet": "RelationNet", "fomaml": "FOMAML"}
TEST_TIME_GRADIENTS = {"metalearnnca": "No", "metalearnnca_img": "No", "nca_protonet": "No", "protonet": "No", "matchingnet": "No",
                       "relationnet": "No", "fomaml": "Yes"}

# ---------------------------------------------------------------------------------------------
# Data: FC100
# ---------------------------------------------------------------------------------------------
COARSE_NAMES = ["aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
                "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
                "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
                "large_omnivores_and_herbivores", "medium_mammals", "non-insect_invertebrates", "people",
                "reptiles", "small_mammals", "trees", "vehicles_1", "vehicles_2"]
FC100_SPLIT = {"train": [1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19],
               "val": [8, 11, 13, 16],
               "test": [0, 7, 12, 14]}


def load_fc100(data_root="./data", cache=True) -> Dict[str, torch.Tensor]:
    """{split: uint8 tensor (n_classes, 600, 3, 32, 32)}. All 600 images of each class
    (CIFAR-100 train + test sets), grouped by fine class, split by superclass."""
    cache_path = os.path.join(data_root, "cifar_cache", "fc100_uint8.pt")
    if cache and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    import torchvision
    for train in (True, False):
        torchvision.datasets.CIFAR100(data_root, train=train, download=True)
    xs, fine, coarse = [], [], []
    for part in ("train", "test"):
        # the same pickle torchvision reads; it also contains the superclass ("coarse") labels
        with open(os.path.join(data_root, "cifar-100-python", part), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        xs.append(np.asarray(d["data"], dtype=np.uint8))
        fine += d["fine_labels"]
        coarse += d["coarse_labels"]
    x = np.concatenate(xs).reshape(-1, 3, IMG, IMG)
    fine, coarse = np.asarray(fine), np.asarray(coarse)
    banks = {}
    for split, supers in FC100_SPLIT.items():
        classes = sorted(set(fine[np.isin(coarse, supers)].tolist()))
        per_class = [torch.from_numpy(x[fine == c]) for c in classes]
        assert all(len(p) == 600 for p in per_class), "every CIFAR-100 class should have 600 images"
        banks[split] = torch.stack(per_class)
    assert [banks[s].shape[0] for s in ("train", "val", "test")] == [60, 20, 20]
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(banks, cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)
    return banks


def make_fake_banks(seed=0, per_class=40) -> Dict[str, torch.Tensor]:
    """Synthetic coloured textures for smoke tests (no download)."""
    g = torch.Generator().manual_seed(seed)
    banks = {}
    for split, n_cls in (("train", 12), ("val", 6), ("test", 6)):
        tmpl = torch.rand(n_cls, 1, 3, 8, 8, generator=g)
        tmpl = F.interpolate(tmpl.flatten(0, 1), size=(IMG, IMG), mode="bilinear", align_corners=False)
        noise = 0.15 * torch.randn(n_cls, per_class, 3, IMG, IMG, generator=g)
        imgs = (tmpl.view(n_cls, 1, 3, IMG, IMG) + noise).clamp(0, 1)
        banks[split] = (imgs * 255).round().to(torch.uint8)
    return banks


def to_float(x_uint8):
    """uint8 [0, 255] -> float [-1, 1] (same preprocessing for every method; no augmentation)."""
    return x_uint8.float() / 127.5 - 1.0


@torch.no_grad()
def sample_tasks(bank, n_tasks, n_way, shot, query):
    """Training episodes, sampled on the bank's device. Returns
    supp_x (T, n_way*shot, 3, H, W), supp_y (T, n_way*shot), qry_x (T, n_way*query, ...), qry_y."""
    n_cls, n_ex = bank.shape[:2]
    dev = bank.device
    cls = torch.rand(n_tasks, n_cls, device=dev).argsort(dim=1)[:, :n_way]
    ex = torch.rand(n_tasks, n_way, n_ex, device=dev).argsort(dim=2)[:, :, :shot + query]
    imgs = to_float(bank[cls.unsqueeze(-1), ex])                       # (T, n_way, shot+query, 3, H, W)
    supp_x = imgs[:, :, :shot].reshape(n_tasks, n_way * shot, IN_CH, IMG, IMG)
    qry_x = imgs[:, :, shot:].reshape(n_tasks, n_way * query, IN_CH, IMG, IMG)
    labels = torch.arange(n_way, device=dev)
    return (supp_x, labels.repeat_interleave(shot).repeat(n_tasks, 1),
            qry_x, labels.repeat_interleave(query).repeat(n_tasks, 1))


def fixed_episodes(bank, n_episodes, n_way, shot, query, seed, start=0):
    """Deterministic evaluation episodes: episode e depends only on (seed, shot, e), so every method,
    seed and machine is scored on exactly the same tasks. Returns CPU tensors stacked over episodes."""
    n_cls, n_ex = bank.shape[:2]
    sx, sy, qx, qy = [], [], [], []
    labels = torch.arange(n_way)
    for e in range(start, start + n_episodes):
        g = torch.Generator().manual_seed(seed * 1_000_003 + shot * 10_007 + e)
        cls = torch.randperm(n_cls, generator=g)[:n_way]
        ex = torch.stack([torch.randperm(n_ex, generator=g)[:shot + query] for _ in range(n_way)])
        imgs = to_float(bank[cls.unsqueeze(-1), ex])
        sx.append(imgs[:, :shot].reshape(n_way * shot, IN_CH, IMG, IMG))
        qx.append(imgs[:, shot:].reshape(n_way * query, IN_CH, IMG, IMG))
        sy.append(labels.repeat_interleave(shot))
        qy.append(labels.repeat_interleave(query))
    return torch.stack(sx), torch.stack(sy), torch.stack(qx), torch.stack(qy)


# ---------------------------------------------------------------------------------------------
# Canonical baselines (as in Train-Canonical-Models.py, with RGB input and a width parameter)
# ---------------------------------------------------------------------------------------------
class Conv4(nn.Module):
    def __init__(self, width=64, in_ch=IN_CH, track_running_stats=True):
        super().__init__()
        layers, c = [], in_ch
        for i in range(4):
            layers += [nn.Conv2d(c, width, 3, padding=1),
                       nn.BatchNorm2d(width, track_running_stats=track_running_stats),
                       nn.ReLU(inplace=True)]
            layers.append(nn.MaxPool2d(2) if i < 3 else nn.AdaptiveAvgPool2d((1, 1)))
            c = width
        self.encoder = nn.Sequential(*layers)
        self.out_dim = width

    def forward(self, x):
        return self.encoder(x).flatten(1)


def _prototypes(feat, y, n_way):
    return torch.stack([feat[y == c].mean(0) for c in range(n_way)])


class _PerTaskLearner(nn.Module):
    """Baselines process one task per forward call (their BatchNorm statistics are per call,
    as in Train-Canonical-Models.py)."""

    def train_loss(self, supp_x, supp_y, qry_x, qry_y):
        losses, accs = [], []
        for t in range(supp_x.shape[0]):
            scores, loss = self.task_scores_and_loss(supp_x[t], supp_y[t], qry_x[t], qry_y[t])
            losses.append(loss)
            accs.append((scores.argmax(1) == qry_y[t]).float().mean())
        return torch.stack(losses).mean(), torch.stack(accs).mean().detach()

    @torch.no_grad()
    def predict(self, supp_x, supp_y, qry_x):
        return torch.stack([self.task_scores(supp_x[t], supp_y[t], qry_x[t]) for t in range(supp_x.shape[0])])

    def task_scores_and_loss(self, sx, sy, qx, qy):
        scores = self.task_scores(sx, sy, qx)
        return scores, F.cross_entropy(scores.float(), qy)


class ProtoNet(_PerTaskLearner):
    def __init__(self, n_way, width=64):
        super().__init__()
        self.n_way, self.backbone = n_way, Conv4(width)

    def task_scores(self, sx, sy, qx):
        protos = _prototypes(self.backbone(sx).float(), sy, self.n_way)
        return -torch.cdist(self.backbone(qx).float(), protos)


class MatchingNet(_PerTaskLearner):
    """Cosine attention scaled by 100 before the softmax (Chen et al., 2019), stable log-probs."""

    def __init__(self, n_way, width=64, scale=100.0):
        super().__init__()
        self.n_way, self.backbone, self.scale = n_way, Conv4(width), scale

    def task_scores(self, sx, sy, qx):
        fs = F.normalize(self.backbone(sx).float(), dim=-1)
        fq = F.normalize(self.backbone(qx).float(), dim=-1)
        log_attn = F.log_softmax(self.scale * fq @ fs.t(), dim=-1)
        in_class = F.one_hot(sy, self.n_way).t().bool()
        return torch.logsumexp(log_attn.unsqueeze(1).masked_fill(~in_class.unsqueeze(0), float("-inf")), dim=-1)


class RelationNet(_PerTaskLearner):
    """Prototype + MLP relation module, sigmoid scores trained with MSE (as in the repo)."""

    def __init__(self, n_way, width=64, head_hidden=64):
        super().__init__()
        self.n_way, self.backbone = n_way, Conv4(width)
        self.head = nn.Sequential(nn.Linear(2 * width, head_hidden), nn.ReLU(inplace=True),
                                  nn.Linear(head_hidden, 1), nn.Sigmoid())

    def task_scores(self, sx, sy, qx):
        protos = _prototypes(self.backbone(sx).float(), sy, self.n_way)
        fq = self.backbone(qx).float()
        Q, D = fq.shape
        pairs = torch.cat([fq.unsqueeze(1).expand(Q, self.n_way, D), protos.unsqueeze(0).expand(Q, self.n_way, D)], -1)
        return self.head(pairs).view(Q, self.n_way)

    def task_scores_and_loss(self, sx, sy, qx, qy):
        scores = self.task_scores(sx, sy, qx)
        return scores, F.mse_loss(scores.float(), F.one_hot(qy, self.n_way).float())


class FOMAML(_PerTaskLearner):
    """First-order MAML on Conv4 (+ linear head). BatchNorm uses batch statistics (no running stats)."""

    def __init__(self, n_way, width=64, inner_steps=FOMAML_INNER["steps"], inner_lr=FOMAML_INNER["lr"]):
        super().__init__()
        self.n_way = n_way
        self.backbone = Conv4(width, track_running_stats=False)
        self.classifier = nn.Linear(width, n_way)
        self.inner_steps, self.inner_lr = inner_steps, inner_lr
        self.test_inner_lr = inner_lr          # may be replaced by a value selected on validation classes

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def adapt(self, sx, sy, lr):
        fast = dict(self.named_parameters())
        with torch.enable_grad():
            for _ in range(self.inner_steps):
                loss = F.cross_entropy(torch.func.functional_call(self, fast, (sx,)).float(), sy)
                grads = torch.autograd.grad(loss, list(fast.values()))
                fast = {k: p - lr * g.detach() for (k, p), g in zip(fast.items(), grads)}  # first order
        return fast

    def task_scores_and_loss(self, sx, sy, qx, qy):
        scores = torch.func.functional_call(self, self.adapt(sx, sy, self.inner_lr), (qx,))
        return scores, F.cross_entropy(scores.float(), qy)

    def task_scores(self, sx, sy, qx):
        fast = self.adapt(sx, sy, self.test_inner_lr)
        with torch.no_grad():
            return torch.func.functional_call(self, fast, (qx,)).float()

    @torch.no_grad()
    def predict(self, supp_x, supp_y, qry_x):
        return torch.stack([self.task_scores(supp_x[t], supp_y[t], qry_x[t]) for t in range(supp_x.shape[0])])


# ---------------------------------------------------------------------------------------------
# NCAs (same update rule as the paper; RGB input; n_way logit channels; no foreground mask)
# ---------------------------------------------------------------------------------------------
def _perception_filters():
    ident = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    lap = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    return torch.stack([ident, sobel_x, sobel_x.T, lap])


class BaseNCA(nn.Module):
    """perception (identity, Sobel x/y, Laplacian; circular padding) -> 1x1 conv -> ReLU -> 1x1 conv;
    stochastic (p=0.5) cell updates. The static input channels' contribution to the first layer is
    computed once per call (exact: perception is depthwise and the layer is linear)."""

    def __init__(self, c_in, c_out, hidden, w2_init_scale=0.2):
        super().__init__()
        self.c_in = c_in
        self.register_buffer("filters", _perception_filters().repeat(c_in, 1, 1).unsqueeze(1))
        self.w1 = nn.Conv2d(c_in * 4, hidden, 1)
        self.w2 = nn.Conv2d(hidden, c_out, 1, bias=False)
        self.w2.weight.data.mul_(w2_init_scale)

    def perceive(self, x, first_channel=0):
        c = x.shape[1]
        f = self.filters[first_channel * 4:(first_channel + c) * 4]
        return F.conv2d(F.pad(x, [1, 1, 1, 1], mode="circular"), f, groups=c)

    def static_term(self, static):
        n = static.shape[1] * 4
        return F.conv2d(self.perceive(static, 0), self.w1.weight[:, :n], self.w1.bias)

    def dyn_update(self, static_term, dyn, n_static: int):
        h = static_term + F.conv2d(self.perceive(dyn, n_static), self.w1.weight[:, n_static * 4:])
        return self.w2(F.relu(h, inplace=True))

    def run(self, static, dyn, steps):
        B, _, H, W = dyn.shape
        st = self.static_term(static.to(dyn.dtype))
        for _ in range(steps):
            ds = self.dyn_update(st, dyn, static.shape[1])
            dyn = dyn + ds * (torch.rand(B, 1, H, W, device=dyn.device) > 0.5).float()
        return dyn


class ActiveNCA(BaseNCA):
    def __init__(self, hidden):
        super().__init__(IN_CH + PROGRAM_CH + DYN_CH, DYN_CH, hidden)

    def forward(self, img, program, steps):
        dyn = torch.rand(img.shape[0], DYN_CH, IMG, IMG, device=img.device)
        return self.run(torch.cat([img, program], dim=1), dyn, steps)


class MetaNCA(BaseNCA):
    """Reads [residual E (n_way channels), optionally the support image (extra_in channels), own state]."""

    def __init__(self, n_way, hidden, extra_in=0):
        super().__init__(n_way + extra_in + DYN_CH, DYN_CH, hidden)

    def forward(self, signal, dyn, steps):
        if dyn is None:
            dyn = torch.rand(signal.shape[0], DYN_CH, IMG, IMG, device=signal.device)
        return self.run(signal, dyn, steps)


def _broadcast(S, n):
    T, C, H, W = S.shape
    return S.unsqueeze(1).expand(T, n, C, H, W).reshape(T * n, C, H, W)


class MetaLearnNCA(nn.Module):
    """Active-NCA + Meta-NCA. Adaptation: K interactions of [Active-NCA on support (2N steps) ->
    residual E = Y - softmax(logits) at EVERY pixel -> Meta-NCA (N steps) -> program = support mean
    of the Meta-NCA's first 24 channels]. Prediction: mean of the logit channels over all pixels.
    image_conditioned=True (method metalearnnca_img): the Meta-NCA reads [E, support image]."""

    def __init__(self, n_way, hidden, n_steps=NCA_STEPS["n_steps"], k_repeats=NCA_STEPS["k_repeats"],
                 image_conditioned=False):
        super().__init__()
        self.n_way, self.n_steps, self.k_repeats = n_way, n_steps, k_repeats
        self.image_conditioned = image_conditioned
        self.active = ActiveNCA(hidden)
        self.meta = MetaNCA(n_way, hidden, extra_in=IN_CH if image_conditioned else 0)

    def adapt(self, supp_x, supp_y, K, N):
        T, S = supp_x.shape[:2]
        x, y = supp_x.flatten(0, 1), supp_y.flatten()
        prog = torch.rand(T, PROGRAM_CH, IMG, IMG, device=x.device)
        y_sp = F.one_hot(y, self.n_way).float()[:, :, None, None].expand(-1, -1, IMG, IMG)
        D = None
        for _ in range(K):
            dyn = self.active(x, _broadcast(prog, S), 2 * N)
            E = y_sp - F.softmax(dyn[:, :self.n_way].float(), dim=1)     # no mask: every pixel
            # image-conditioned variant: the updater also sees each support image, so its input
            # carries image structure even when the predictions (and hence E) are spatially uniform
            signal = torch.cat([E, x.float()], dim=1) if self.image_conditioned else E
            D = self.meta(signal, D, N)
            prog = D.view(T, S, DYN_CH, IMG, IMG)[:, :, :PROGRAM_CH].mean(dim=1)
        return prog

    def query_logits(self, qry_x, prog, N):
        T, Q = qry_x.shape[:2]
        return self.active(qry_x.flatten(0, 1), _broadcast(prog, Q), 2 * N)[:, :self.n_way].float()

    def train_loss(self, supp_x, supp_y, qry_x, qry_y):
        K = random.randint(self.k_repeats, self.k_repeats + 2)
        N = random.randint(self.n_steps, self.n_steps + 3)
        logits = self.query_logits(qry_x, self.adapt(supp_x, supp_y, K, N), N)   # (T*Q, n_way, H, W)
        y = qry_y.flatten()
        loss = F.cross_entropy(logits, y[:, None, None].expand(-1, IMG, IMG))     # every pixel
        acc = (logits.mean(dim=(2, 3)).argmax(1) == y).float().mean()
        return loss, acc.detach()

    @torch.no_grad()
    def predict(self, supp_x, supp_y, qry_x):
        K, N = self.k_repeats + 2, self.n_steps + 3
        logits = self.query_logits(qry_x, self.adapt(supp_x, supp_y, K, N), N)
        return logits.mean(dim=(2, 3)).view(qry_x.shape[0], qry_x.shape[1], self.n_way)


class NCAProtoNet(nn.Module):
    """The same Active-NCA as an embedding (mean of its 32 state channels over all pixels), prototype
    classification by squared Euclidean distance; one learned, task-independent program."""

    def __init__(self, n_way, hidden, n_steps=NCA_STEPS["n_steps"]):
        super().__init__()
        self.n_way, self.n_steps = n_way, n_steps
        self.active = ActiveNCA(hidden)
        self.S0 = nn.Parameter(torch.rand(1, PROGRAM_CH, IMG, IMG))

    def embed(self, x, N):
        return self.active(x, self.S0.expand(x.shape[0], -1, -1, -1), 2 * N).float().mean(dim=(2, 3))

    def scores(self, supp_x, supp_y, qry_x, N):
        T, S = supp_x.shape[:2]
        Q = qry_x.shape[1]
        es = self.embed(supp_x.flatten(0, 1), N).view(T, S, -1)
        eq = self.embed(qry_x.flatten(0, 1), N).view(T, Q, -1)
        onehot = F.one_hot(supp_y, self.n_way).float()                    # (T, S, n_way)
        protos = onehot.transpose(1, 2) @ es / onehot.sum(1).clamp(min=1).unsqueeze(-1)
        d2 = eq.pow(2).sum(-1, keepdim=True) + protos.pow(2).sum(-1).unsqueeze(1) - 2 * eq @ protos.transpose(1, 2)
        return -d2.clamp(min=0.0)                                          # (T, Q, n_way), squared Euclidean

    def train_loss(self, supp_x, supp_y, qry_x, qry_y):
        N = random.randint(self.n_steps, self.n_steps + 3)
        sc = self.scores(supp_x, supp_y, qry_x, N)
        loss = F.cross_entropy(sc.flatten(0, 1), qry_y.flatten())
        return loss, (sc.argmax(-1) == qry_y).float().mean().detach()

    @torch.no_grad()
    def predict(self, supp_x, supp_y, qry_x):
        return self.scores(supp_x, supp_y, qry_x, self.n_steps + 3)


# ---------------------------------------------------------------------------------------------
# Parameter-matched construction
# ---------------------------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _make(method, n_way, size):
    if method == "protonet":
        return ProtoNet(n_way, width=size[0])
    if method == "matchingnet":
        return MatchingNet(n_way, width=size[0])
    if method == "relationnet":
        return RelationNet(n_way, width=size[0], head_hidden=size[1])
    if method == "fomaml":
        return FOMAML(n_way, width=size[0])
    if method == "metalearnnca":
        return MetaLearnNCA(n_way, hidden=size[0])
    if method == "metalearnnca_img":
        return MetaLearnNCA(n_way, hidden=size[0], image_conditioned=True)
    if method == "nca_protonet":
        return NCAProtoNet(n_way, hidden=size[0])
    raise ValueError(method)


def _count_on_meta(method, n_way, size):
    with torch.device("meta"):
        return count_params(_make(method, n_way, size))


def param_target(n_way):
    """The budget: a Conv4-64 ProtoNet on 3x32x32 inputs."""
    return _count_on_meta("protonet", n_way, (64,))


@functools.lru_cache(maxsize=None)
def matched_size(method, n_way, target=None) -> Tuple[tuple, int, int]:
    """Width / hidden size whose parameter count is closest to the target. Returns (size, count, target)."""
    target = target or param_target(n_way)
    if method in ("protonet", "matchingnet", "fomaml"):
        grid = [(w,) for w in range(48, 81)]
    elif method == "relationnet":
        grid = [(w, h) for w in range(56, 65) for h in range(16, 129)]
    else:
        grid = [(h,) for h in range(128, 513)]
    counts = {s: _count_on_meta(method, n_way, s) for s in grid}
    # closest count; for RelationNet, among matches within 0.5% keep the Conv4 width closest to the
    # canonical 64 (the relation head absorbs the difference)
    best = min(grid, key=lambda s: (abs(counts[s] - target) > 0.005 * target,
                                    abs(s[0] - 64) if method == "relationnet" else 0,
                                    abs(counts[s] - target)))
    return best, counts[best], target


def build_learner(method, n_way=REGIME["n_way"], size=None):
    if size is None:
        size, _, _ = matched_size(method, n_way)
    return _make(method, n_way, tuple(size))


def mean_ci95(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()) if len(a) else float("nan"), float("nan")
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, len(a) - 1))
    except ImportError:
        t = 1.96
    return float(a.mean()), t * float(a.std(ddof=1)) / math.sqrt(len(a))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(device, dtype):
    import contextlib
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(torch.device(device).type, dtype=dtype)
