"""
Shared code for the MetaLearnNCA component-ablation study (ICLR 2027 revision).

Design rules
------------
* One code path. Meta-training (`episode`) and evaluation (`adapt` + `predict`)
  call the same functions, so a variant cannot behave differently at test time.
* Single-factor changes. Every variant differs from `full` in exactly one
  component; everything else (data, episode sampling, step jitter, optimiser,
  schedule, AMP, clipping) is identical and comes from
  `ML-NCA-Faster-trainign.py`, the script that produced `checkpoints/`.
* Paired evaluation. Test tasks are drawn from a dedicated RNG seeded with the
  evaluation seed, so every variant and every checkpoint sees exactly the same
  support/query splits. With the default seeds (43..52) and `RandomState`, the
  class choice and support/test split reproduce the ones used by
  `ML-NCA-Testing.py --run_eval_loop` (MODE A) for Table 1.

Variants (see VARIANTS below and ablations/README.md)
------------------------------------------------------
main table : full, global_program, no_mask, grad_on_s, nca_protonet
appendix   : no_feedback, noncellular_updater, short_unroll
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# 0. CONSTANTS (identical to the original scripts)
# ======================================================================
ACTIVE_DYN_CHN = 32   # Active-NCA dynamic channels (0..9 = class logits)
META_DYN_CHN = 32     # Meta-NCA dynamic channels (0..23 -> program)
STATIC_CHN = 24       # spatial program S
LOSS_CHN = 10         # number of classes / logit channels
N_WAY = 10
GRID = 28
MASK_THRESHOLD = -0.8  # foreground ("alive") pixels: x > -0.8 on [-1, 1] images

# Hyper-parameters of ML-NCA-Faster-trainign.py (the script that produced the
# released checkpoints). NOTE: Section 5 of the submitted paper states E=5000,
# B=32, lr=1e-3 and W2 initialised to zero; the fast script uses the values
# below. Please reconcile the paper text with whichever script produced Table 1.
TRAIN_DEFAULTS = dict(
    epochs=4000,
    micro_batch=2,        # tasks processed in parallel
    accum=8,              # accumulation passes -> 16 tasks per optimiser step
    lr=2e-3,
    lr_min=5e-4,          # cosine annealing floor
    weight_decay=1e-5,
    grad_clip=1.0,
    n_way=10, n_supp=1, n_qry=5,
)


# ======================================================================
# 1. VARIANT SPECIFICATIONS
# ======================================================================
@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str              # display name for tables
    change: str             # what is changed relative to `full`
    question: str           # reviewer question it answers
    asked_by: str
    tier: str = "main"      # "main" table or "appendix"
    program: str = "spatial"            # spatial | global
    updater: str = "meta_nca"           # meta_nca | gradient | label_only | prototype
    updater_perception: str = "sobel"   # sobel (3x3 identity/sobel/laplacian) | identity (1x1)
    use_mask: bool = True
    n_steps: int = 16       # base inner steps N (train jitter N' in [N, N+3]; test N+3)
    k_repeats: int = 3      # base interactions K (train jitter K' in [K, K+2]; test K+2)


VARIANTS: Dict[str, VariantSpec] = {
    "full": VariantSpec(
        "full", "Full MetaLearnNCA", "none (reference)",
        "reference", "-", tier="main"),
    "global_program": VariantSpec(
        "global_program", "Global program (24-D, tiled)",
        "S is a 24-D vector (spatial mean of the Meta-NCA output) tiled over the grid",
        "Does the spatial structure of the program matter?", "R1, R3",
        program="global"),
    "no_mask": VariantSpec(
        "no_mask", "No foreground mask",
        "mask removed from residual E_pixel, loss and read-out (all pixels used)",
        "How much comes from masking?", "R1, R3",
        use_mask=False),
    "grad_on_s": VariantSpec(
        "grad_on_s", "Gradient on S (CAVIA-style)",
        "Meta-NCA replaced by SGD steps on S using the masked support loss "
        "(meta-learned per-channel step size, second-order meta-gradient)",
        "Does the learned updater beat plain gradients?", "R1, R2, R3",
        updater="gradient"),
    "nca_protonet": VariantSpec(
        "nca_protonet", "NCA-ProtoNet",
        "same Active-NCA used as an embedding (masked mean of its 32 dynamic "
        "channels); prototype classification; no program adaptation",
        "Is the gain from the NCA architecture or from the adaptation?", "R2, R3",
        updater="prototype"),
    # ---------------- appendix variants (run if time allows) ----------------
    "no_feedback": VariantSpec(
        "no_feedback", "No error feedback (labels only)",
        "Meta-NCA reads Y_spatial * M (labelled strokes) instead of the residual "
        "Y - softmax(logits); Active-NCA predictions never reach the updater",
        "Is adaptation actually error-driven (closed loop)?", "R2, R3",
        tier="appendix", updater="label_only"),
    "noncellular_updater": VariantSpec(
        "noncellular_updater", "Non-cellular updater (1x1)",
        "Meta-NCA perception reduced to the identity kernel (no neighbourhood "
        "communication); hidden width enlarged to match the parameter count",
        "Does the updater need local diffusion of errors?", "R1, R3",
        tier="appendix", updater_perception="identity"),
    "short_unroll": VariantSpec(
        "short_unroll", "Short unroll (N=8)",
        "base inner steps N=8 instead of 16 (train N' in [8,11], test N=11)",
        "Is long recurrent unrolling necessary?", "R3",
        tier="appendix", n_steps=8),
}

MAIN_VARIANTS = [v for v, s in VARIANTS.items() if s.tier == "main"]
APPENDIX_VARIANTS = [v for v, s in VARIANTS.items() if s.tier == "appendix"]

# Options that are not part of a variant's identity but are needed to rebuild it.
DEFAULT_OPTS = dict(
    hidden=256,              # MLP hidden width of both NCAs
    w2_init_scale=0.2,       # fast script: default init / 5 (paper says zero)
    match_params=True,       # noncellular_updater: enlarge hidden to match #params
    # --- grad_on_s ---
    inner_lr_mode="per_channel",  # fixed | per_channel | per_element (Meta-SGD)
    inner_lr=None,           # None -> calibrated default (see DEFAULT_INNER_LR)
    first_order=False,       # True -> FO-CAVIA (no second-order terms; less memory)
    s_init="zeros",          # zeros (CAVIA) | uniform (as full) | learned
    # --- nca_protonet ---
    proto_embed="pooled",    # pooled (masked mean, 32-D) | spatial (masked 32x28x28)
    proto_program="learned", # learned task-agnostic program | zeros
)

# Initial inner step size for grad_on_s. The support loss is a mean over all
# foreground pixels of the support set, so dL/dS per grid cell is small
# (RMS ~1.5e-3 at S=0). Calibrated on a released checkpoint (seed 55, MNIST):
# lr=10 lowers the support loss monotonically over 3 steps (2.89 -> 0.99),
# lr=100 diverges after the first step. The step size is then meta-learned
# (log-parameterised) unless inner_lr_mode == "fixed".
DEFAULT_INNER_LR = 10.0


# ======================================================================
# 2. NCA BUILDING BLOCKS (state-dict compatible with the original scripts)
# ======================================================================
def _perception_filters(kind: str) -> torch.Tensor:
    ident = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    if kind == "identity":
        return ident.unsqueeze(0)                                   # (1, 3, 3)
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    lap = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    return torch.stack([ident, sobel_x, sobel_x.T, lap])           # (4, 3, 3)


class BaseNCA(nn.Module):
    def __init__(self, c_in: int, c_out: int, hidden: int,
                 perception: str = "sobel", w2_init_scale: float = 0.2):
        super().__init__()
        self.c_in = c_in
        self.perception = perception
        base = _perception_filters(perception)
        self.n_filters = base.shape[0]
        self.register_buffer("filters", base.repeat(c_in, 1, 1).unsqueeze(1))
        self.w1 = nn.Conv2d(c_in * self.n_filters, hidden, 1)
        self.w2 = nn.Conv2d(hidden, c_out, 1, bias=False)
        self.w2.weight.data.mul_(w2_init_scale)

    # Speed-up (exact up to float rounding): perception is depthwise and W1 is linear, so for an
    # input [static, dyn] we have W1 P([static, dyn]) + b = (W1_s P(static) + b) + W1_d P(dyn).
    # The static part (image + program for the Active-NCA, error signal for the Meta-NCA) does not
    # change during a call, so it is computed once per call instead of at every step: the per-step
    # 1x1 conv shrinks from 228 to 128 input channels (Active) / 168 to 128 (Meta), and fewer
    # activations are stored for backward. Set PRECOMPUTE_STATIC = False for the literal form.
    PRECOMPUTE_STATIC = True

    def perceive(self, x, first_channel=0):
        """Perception of input channels [first_channel, first_channel + x.shape[1])."""
        if self.perception == "identity":
            return x
        c = x.shape[1]
        f = self.filters[first_channel * self.n_filters:(first_channel + c) * self.n_filters]
        x_pad = F.pad(x, [1, 1, 1, 1], mode="circular")
        return F.conv2d(x_pad, f, groups=c)

    def grid_update(self, x):
        """Literal update rule (Eq. 1) on the full concatenated input."""
        return self.w2(F.relu(self.w1(self.perceive(x)), inplace=True))

    def static_term(self, static):
        n = static.shape[1] * self.n_filters
        return F.conv2d(self.perceive(static, 0), self.w1.weight[:, :n], self.w1.bias)

    def dyn_update(self, static_term, dyn, n_static: int):
        n = n_static * self.n_filters
        h = static_term + F.conv2d(self.perceive(dyn, n_static), self.w1.weight[:, n:])
        return self.w2(F.relu(h, inplace=True))

    def run(self, static, dyn, steps):
        B, _, H, W = dyn.shape
        if self.PRECOMPUTE_STATIC:
            st = self.static_term(static.to(dyn.dtype))   # same promotion as torch.cat below
            n_static = static.shape[1]
            for _ in range(steps):
                ds = self.dyn_update(st, dyn, n_static)
                dyn = dyn + ds * _stochastic_mask(B, H, W, dyn.device)
        else:
            for _ in range(steps):
                ds = self.grid_update(torch.cat([static, dyn], dim=1))
                dyn = dyn + ds * _stochastic_mask(B, H, W, dyn.device)
        return dyn


def _stochastic_mask(B, H, W, device):
    return (torch.rand(B, 1, H, W, device=device) > 0.5).float()


class ActiveNCA(BaseNCA):
    def __init__(self, hidden=256, w2_init_scale=0.2):
        super().__init__(1 + STATIC_CHN + ACTIVE_DYN_CHN, ACTIVE_DYN_CHN, hidden,
                         "sobel", w2_init_scale)

    def forward(self, img, program, steps, dyn=None):
        B, _, H, W = img.shape
        if dyn is None:
            dyn = torch.rand(B, ACTIVE_DYN_CHN, H, W, device=img.device)
        return self.run(torch.cat([img, program], dim=1), dyn, steps)


class MetaNCA(BaseNCA):
    def __init__(self, hidden=256, perception="sobel", w2_init_scale=0.2):
        super().__init__(LOSS_CHN + META_DYN_CHN, META_DYN_CHN, hidden,
                         perception, w2_init_scale)

    def forward(self, signal, dyn=None, steps=16):
        B, _, H, W = signal.shape
        if dyn is None:
            dyn = torch.rand(B, META_DYN_CHN, H, W, device=signal.device)
        return self.run(signal, dyn, steps)


def matched_identity_hidden(hidden: int) -> int:
    """Hidden width for a 1x1-perception Meta-NCA with the same #params as the 3x3 one."""
    c_in = LOSS_CHN + META_DYN_CHN
    full = 4 * c_in * hidden + hidden + hidden * META_DYN_CHN
    return int(round(full / (c_in + 1 + META_DYN_CHN)))


# ======================================================================
# 3. HELPERS SHARED BY ALL VARIANTS
# ======================================================================
def spatial_onehot(y, H, W, n_cls=LOSS_CHN):
    return F.one_hot(y, n_cls).float()[:, :, None, None].expand(-1, -1, H, W)


def pixel_ce(logits, y, mask):
    """Per-pixel CE (B,H,W) of logits (B,C,H,W) against image label y, times mask."""
    B, _, H, W = logits.shape
    ce = F.cross_entropy(logits.float(), y.view(B, 1, 1).expand(B, H, W), reduction="none")
    return ce * mask[:, 0]


def masked_readout(logits, mask):
    """Eq. 5: average logits over foreground pixels -> (B, C)."""
    return (logits * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp(min=1.0)


def broadcast_program(S, n_per_task):
    """(T, C, H, W) -> (T * n_per_task, C, H, W), task-major like the original."""
    T, C, H, W = S.shape
    return S.unsqueeze(1).expand(T, n_per_task, C, H, W).reshape(T * n_per_task, C, H, W)


# ======================================================================
# 4. LEARNERS
# ======================================================================
class ProgramLearner(nn.Module):
    """full / global_program / no_mask / grad_on_s / no_feedback /
    noncellular_updater / short_unroll."""

    def __init__(self, spec: VariantSpec, opts: dict):
        super().__init__()
        self.spec = spec
        self.opts = opts
        self.active = ActiveNCA(opts["hidden"], opts["w2_init_scale"])
        self.meta = None
        if spec.updater in ("meta_nca", "label_only"):
            hidden = opts["hidden"]
            if spec.updater_perception == "identity" and opts.get("match_params", True):
                hidden = matched_identity_hidden(hidden)
            self.meta = MetaNCA(hidden, spec.updater_perception, opts["w2_init_scale"])
        elif spec.updater == "gradient":
            assert spec.program == "spatial"
            lr0 = math.log(opts["inner_lr"] if opts.get("inner_lr") else DEFAULT_INNER_LR)
            mode = opts["inner_lr_mode"]
            if mode == "fixed":
                self.register_buffer("log_inner_lr", torch.tensor(lr0))
            elif mode == "per_channel":
                self.log_inner_lr = nn.Parameter(torch.full((1, STATIC_CHN, 1, 1), lr0))
            elif mode == "per_element":
                self.log_inner_lr = nn.Parameter(torch.full((1, STATIC_CHN, GRID, GRID), lr0))
            else:
                raise ValueError(mode)
            if opts["s_init"] == "learned":
                self.S0 = nn.Parameter(torch.zeros(1, STATIC_CHN, GRID, GRID))
        else:
            raise ValueError(f"ProgramLearner cannot run updater '{spec.updater}'")

    # ---------------- small helpers ----------------
    def alive_mask(self, x):
        if self.spec.use_mask:
            return (x > MASK_THRESHOLD).float()
        return torch.ones_like(x)

    def inner_lr(self):
        return self.log_inner_lr.exp()

    def init_program(self, n_tasks, H, W, device):
        if self.spec.updater == "gradient":
            mode = self.opts["s_init"]
            if mode == "zeros":
                return torch.zeros(n_tasks, STATIC_CHN, H, W, device=device)
            if mode == "uniform":
                return torch.rand(n_tasks, STATIC_CHN, H, W, device=device)
            return self.S0.expand(n_tasks, -1, -1, -1)
        if self.spec.program == "global":
            return torch.rand(n_tasks, STATIC_CHN, 1, 1, device=device).expand(-1, -1, H, W)
        return torch.rand(n_tasks, STATIC_CHN, H, W, device=device)

    def program_from_meta(self, D, n_tasks):
        B, C, H, W = D.shape
        D = D.view(n_tasks, B // n_tasks, C, H, W)[:, :, :STATIC_CHN]
        if self.spec.program == "global":
            return D.mean(dim=(1, 3, 4), keepdim=False)[:, :, None, None].expand(-1, -1, H, W)
        return D.mean(dim=1)                       # batch mean-pooling (Alg. 1, l.14)

    # ---------------- adaptation (shared by train and test) ----------------
    def adapt(self, supp_x, supp_y, n_tasks: int, K: int, N: int,
              meta_train: bool = False,
              callback: Optional[Callable[[int, torch.Tensor], None]] = None):
        """Returns the adapted program S of shape (n_tasks, 24, H, W).
        `callback(k, S_k)` is called for k = 0..K (used for trajectories)."""
        B, _, H, W = supp_x.shape
        n_supp = B // n_tasks
        mask = self.alive_mask(supp_x)
        S = self.init_program(n_tasks, H, W, supp_x.device)
        if callback is not None:
            callback(0, S.detach())
        if self.spec.updater == "gradient":
            return self._adapt_gradient(supp_x, supp_y, mask, S, n_tasks, n_supp,
                                        K, N, meta_train, callback)
        y_sp = spatial_onehot(supp_y, H, W)
        D = None
        for k in range(1, K + 1):
            if self.spec.updater == "label_only":
                signal = y_sp * mask                     # open loop: no predictions
            else:
                dyn = self.active(supp_x, broadcast_program(S, n_supp), steps=2 * N)
                probs = F.softmax(dyn[:, :LOSS_CHN].float(), dim=1)
                signal = (y_sp - probs) * mask          # Eq. 3
            D = self.meta(signal, D, steps=N)
            S = self.program_from_meta(D, n_tasks)
            if callback is not None:
                callback(k, S.detach())
        return S

    def _adapt_gradient(self, supp_x, supp_y, mask, S, n_tasks, n_supp, K, N,
                        meta_train, callback):
        create_graph = meta_train and not self.opts["first_order"]
        with torch.enable_grad():
            lr = self.inner_lr()
            if not S.requires_grad:
                S = S.detach().requires_grad_(True)
            den = mask.view(n_tasks, -1).sum(1).clamp(min=1.0)
            chunk = int(self.opts.get("eval_grad_chunk", 25))
            for k in range(1, K + 1):
                if not meta_train and n_tasks == 1 and supp_x.shape[0] > chunk:
                    # Test time: S is shared by all support images, so dL/dS is a sum of
                    # per-image terms; accumulating it over chunks gives the same gradient
                    # with bounded memory (10-shot = 100 images x 38 steps otherwise).
                    g = torch.zeros_like(S)
                    for i in range(0, supp_x.shape[0], chunk):
                        xs = supp_x[i:i + chunk]
                        dyn = self.active(xs, S.expand(xs.shape[0], -1, -1, -1), steps=2 * N)
                        ce = pixel_ce(dyn[:, :LOSS_CHN], supp_y[i:i + chunk], mask[i:i + chunk])
                        g = g + torch.autograd.grad(ce.sum() / den[0], S)[0]
                else:
                    dyn = self.active(supp_x, broadcast_program(S, n_supp), steps=2 * N)
                    ce = pixel_ce(dyn[:, :LOSS_CHN], supp_y, mask)
                    # per-task masked mean (so the step size does not depend on #tasks)
                    num = ce.view(n_tasks, -1).sum(1)
                    loss = (num / den).sum()
                    (g,) = torch.autograd.grad(loss, S, create_graph=create_graph)
                if not create_graph:
                    g = g.detach()
                S = S - lr * g
                if not meta_train:
                    S = S.detach().requires_grad_(k < K)
                if callback is not None:
                    callback(k, S.detach())
        return S if meta_train else S.detach()

    # ---------------- query side ----------------
    def query_logits_map(self, qry_x, S, n_tasks, N):
        n_qry = qry_x.shape[0] // n_tasks
        dyn = self.active(qry_x, broadcast_program(S, n_qry), steps=2 * N)
        return dyn[:, :LOSS_CHN]

    @torch.no_grad()
    def predict(self, x, state, N):
        """Class scores (B, 10) for one adapted task (state: (1, 24, H, W))."""
        logits = self.query_logits_map(x, state, 1, N)
        return masked_readout(logits.float(), self.alive_mask(x))

    def episode(self, supp_x, supp_y, qry_x, qry_y, n_tasks, K, N):
        """Meta-training episode(s): returns (query loss, query accuracy)."""
        S = self.adapt(supp_x, supp_y, n_tasks, K, N, meta_train=True)
        logits = self.query_logits_map(qry_x, S, n_tasks, N)
        mask = self.alive_mask(qry_x)
        # pooled over all query pixels of the micro-batch (as in the original)
        loss = pixel_ce(logits, qry_y, mask).sum() / mask.sum().clamp(min=1.0)
        with torch.no_grad():
            acc = (masked_readout(logits.float(), mask).argmax(1) == qry_y).float().mean()
        return loss, acc


class NCAProtoNet(nn.Module):
    """The Active-NCA as an embedding network + prototype classification."""

    def __init__(self, spec: VariantSpec, opts: dict):
        super().__init__()
        self.spec = spec
        self.opts = opts
        self.active = ActiveNCA(opts["hidden"], opts["w2_init_scale"])
        if opts["proto_program"] == "learned":
            self.S0 = nn.Parameter(torch.rand(1, STATIC_CHN, GRID, GRID))
        else:
            self.register_buffer("S0", torch.zeros(1, STATIC_CHN, GRID, GRID))
        # spatial embeddings are 32x28x28-D: use the per-pixel mean squared distance so
        # the logits have the same scale as the pooled 32-D embedding (else softmax saturates)
        self.dist_scale = 1.0 / (GRID * GRID) if opts["proto_embed"] == "spatial" else 1.0

    def alive_mask(self, x):
        return (x > MASK_THRESHOLD).float() if self.spec.use_mask else torch.ones_like(x)

    def embed(self, x, N):
        dyn = self.active(x, self.S0.expand(x.shape[0], -1, -1, -1), steps=2 * N).float()
        mask = self.alive_mask(x)
        if self.opts["proto_embed"] == "spatial":
            return (dyn * mask).flatten(1)
        return masked_readout(dyn, mask)                           # (B, 32)

    def adapt(self, supp_x, supp_y, n_tasks, K, N, meta_train=False, callback=None):
        emb = self.embed(supp_x, N)
        D = emb.shape[1]
        emb = emb.view(n_tasks, -1, D)
        y = supp_y.view(n_tasks, -1)
        protos = torch.zeros(n_tasks, N_WAY, D, device=emb.device, dtype=emb.dtype)
        protos = protos.scatter_add(1, y[..., None].expand(-1, -1, D), emb)
        counts = torch.zeros(n_tasks, N_WAY, device=emb.device, dtype=emb.dtype).scatter_add(
            1, y, torch.ones_like(y, dtype=emb.dtype))
        return protos / counts.clamp(min=1.0)[..., None]

    def _neg_sqdist(self, q, p):
        # q: (T, Q, D), p: (T, C, D) -> (T, Q, C)
        d = q.pow(2).sum(-1, keepdim=True) + p.pow(2).sum(-1)[:, None, :] - 2.0 * q @ p.transpose(1, 2)
        return -self.dist_scale * d.clamp(min=0.0)

    @torch.no_grad()
    def predict(self, x, state, N):
        return self._neg_sqdist(self.embed(x, N)[None], state)[0]

    def episode(self, supp_x, supp_y, qry_x, qry_y, n_tasks, K, N):
        protos = self.adapt(supp_x, supp_y, n_tasks, K, N, meta_train=True)
        q = self.embed(qry_x, N).view(n_tasks, -1, protos.shape[-1])
        logits = self._neg_sqdist(q, protos).reshape(-1, N_WAY)
        loss = F.cross_entropy(logits, qry_y)
        with torch.no_grad():
            acc = (logits.argmax(1) == qry_y).float().mean()
        return loss, acc


def build_learner(variant: str, opts: Optional[dict] = None) -> nn.Module:
    spec = VARIANTS[variant]
    full_opts = dict(DEFAULT_OPTS)
    full_opts.update(opts or {})
    if spec.updater == "gradient" and not full_opts.get("inner_lr"):
        full_opts["inner_lr"] = DEFAULT_INNER_LR
    if spec.updater == "prototype":
        return NCAProtoNet(spec, full_opts)
    return ProgramLearner(spec, full_opts)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_time_schedule(spec: VariantSpec) -> Tuple[int, int]:
    """(K_test, N_test) = top of the training jitter range, as in the original
    (train K' in [3,5], N' in [16,19]; test K=5, N=19)."""
    return spec.k_repeats + 2, spec.n_steps + 3


# ======================================================================
# 5. CHECKPOINT I/O
# ======================================================================
def save_run_checkpoint(path, learner, variant, opts, extra=None):
    payload = {"variant": variant, "opts": opts,
               "state_dict": {k: v.detach().cpu() for k, v in learner.state_dict().items()}}
    if extra:
        payload.update(extra)
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_run_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    learner = build_learner(payload["variant"], payload["opts"])
    learner.load_state_dict(payload["state_dict"])
    return learner.to(device), payload


def load_original_checkpoint(active_path, meta_path, device="cpu", hidden=256):
    """Wrap a checkpoint pair from checkpoints/ as a `full` ProgramLearner."""
    learner = build_learner("full", {"hidden": hidden})
    learner.active.load_state_dict(torch.load(active_path, map_location=device))
    learner.meta.load_state_dict(torch.load(meta_path, map_location=device))
    return learner.to(device)


def find_original_checkpoints(ckpt_dir, seeds=None):
    """{seed_str: (active_path, meta_path)} for files active_nca_seed<S>.pth."""
    import re
    pat = re.compile(r"active_nca_seed(\d+)\.pth$")
    pairs = {}
    for f in os.listdir(ckpt_dir):
        m = pat.match(f)
        if not m:
            continue
        meta = os.path.join(ckpt_dir, f"meta_nca_seed{m.group(1)}.pth")
        if os.path.exists(meta):
            pairs[m.group(1)] = (os.path.join(ckpt_dir, f), meta)
    if seeds and seeds != ["all"]:
        missing = [s for s in seeds if s not in pairs]
        if missing:
            raise KeyError(f"seeds {missing} not found in {ckpt_dir}")
        pairs = {s: pairs[s] for s in seeds}
    return dict(sorted(pairs.items(), key=lambda kv: int(kv[0])))


# ======================================================================
# 6. DATA
# ======================================================================
DATASETS = ["omniglot", "mnist", "usps", "fashion", "kmnist", "medmnist"]
OOD_DATASETS = ["mnist", "usps", "fashion", "kmnist", "medmnist"]
DATASET_LABELS = {"omniglot": "Omniglot", "mnist": "MNIST", "usps": "USPS",
                  "fashion": "Fashion-MNIST", "kmnist": "KMNIST",
                  "medmnist": "MedMNIST", "fake": "Fake"}


def _transforms():
    import torchvision.transforms as T
    from PIL import ImageFilter
    standard = T.Compose([T.Resize((28, 28)), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    edges = T.Compose([T.Resize((28, 28)), T.Lambda(lambda x: x.filter(ImageFilter.FIND_EDGES)),
                       T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    omni = T.Compose([T.Resize((28, 28)), T.ToTensor(), T.Lambda(lambda x: 1.0 - x),
                      T.Normalize((0.5,), (0.5,))])
    return standard, edges, omni


def _materialise(dataset, label_fn=None):
    imgs, labels = [], []
    for i in range(len(dataset)):
        img, y = dataset[i]
        if label_fn is not None:
            y = label_fn(i)
        if isinstance(y, (np.ndarray, list)):
            y = int(np.asarray(y).flatten()[0])
        elif torch.is_tensor(y):
            y = int(y.item())
        imgs.append(img)
        labels.append(int(y))
    return torch.stack(imgs).float(), np.asarray(labels, dtype=np.int64)


def load_eval_dataset(name, data_root="./data", cache=True):
    """(images [N,1,28,28] float32 in [-1,1], targets int64[N]) with exactly the
    preprocessing of ML-NCA-Testing.py. Cached under <data_root>/ablation_cache."""
    if name == "fake":
        return make_fake_dataset(n_classes=12, n_per_class=40, seed=1234)
    cache_path = os.path.join(data_root, "ablation_cache", f"eval_{name}.pt")
    if cache and os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        return d["images"], d["targets"]
    import torchvision
    standard, edges, omni = _transforms()
    if name == "mnist":
        ds = torchvision.datasets.MNIST(data_root, train=False, download=True, transform=standard)
    elif name == "fashion":
        ds = torchvision.datasets.FashionMNIST(data_root, train=False, download=True, transform=edges)
    elif name == "kmnist":
        ds = torchvision.datasets.KMNIST(data_root, train=False, download=True, transform=standard)
    elif name == "usps":
        ds = torchvision.datasets.USPS(data_root, train=False, download=True, transform=standard)
    elif name == "medmnist":
        import medmnist
        info = medmnist.INFO["organamnist"]   # 11 organ classes; 10-way tasks sample 10 of them
        # default root (~/.medmnist), as in the original scripts
        ds = getattr(medmnist, info["python_class"])(split="test", download=True, transform=edges)
    elif name == "omniglot":
        ds = torchvision.datasets.Omniglot(data_root, background=False, download=True, transform=omni)
    else:
        raise ValueError(name)
    print(f"[data] materialising '{name}' ({len(ds)} images) ...", flush=True)
    images, targets = _materialise(ds)
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"images": images, "targets": targets}, cache_path)
    return images, targets


def load_omniglot_train_bank(data_root="./data", cache=True):
    """Omniglot background set grouped by class: (964, 20, 1, 28, 28)."""
    cache_path = os.path.join(data_root, "ablation_cache", "train_omniglot_background.pt")
    if cache and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    import torchvision
    _, _, omni = _transforms()
    ds = torchvision.datasets.Omniglot(data_root, background=True, download=True, transform=omni)
    print(f"[data] materialising Omniglot background ({len(ds)} images) ...", flush=True)
    images, targets = _materialise(ds, label_fn=lambda i: ds._flat_character_images[i][1])
    bank = torch.stack([images[targets == c] for c in sorted(set(targets.tolist()))])
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(bank, cache_path)
    return bank


def _draw_segment(img, p0, p1):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) * 2 + 2
    for t in np.linspace(0.0, 1.0, n):
        r = int(round(p0[0] + t * (p1[0] - p0[0])))
        c = int(round(p0[1] + t * (p1[1] - p0[1])))
        img[max(r - 1, 0):r + 1, max(c - 1, 0):c + 1] = 1.0


def make_fake_dataset(n_classes=30, n_per_class=20, seed=0):
    """Synthetic stroke 'characters' for smoke tests (no download needed)."""
    rng = np.random.RandomState(seed)
    imgs, labels = [], []
    for c in range(n_classes):
        segs = [(rng.randint(5, 23, size=2), rng.randint(5, 23, size=2)) for _ in range(3)]
        for _ in range(n_per_class):
            img = np.zeros((28, 28), np.float32)
            off = rng.randint(-1, 2, size=2)
            for p0, p1 in segs:
                _draw_segment(img, p0 + off, p1 + off)
            img = np.clip(img + 0.1 * rng.randn(28, 28) * img, 0, 1)
            imgs.append(img * 2.0 - 1.0)
            labels.append(c)
    return torch.from_numpy(np.stack(imgs).astype(np.float32))[:, None], np.asarray(labels, dtype=np.int64)


def make_fake_train_bank(n_classes=60, n_per_class=20, seed=0):
    images, targets = make_fake_dataset(n_classes, n_per_class, seed)
    return torch.stack([images[targets == c] for c in range(n_classes)])


@torch.no_grad()
def sample_meta_batch(bank, n_tasks, n_way=10, n_supp=1, n_qry=5):
    """Same sampler as ML-NCA-Faster-trainign.py (shared example permutation per task)."""
    num_classes, num_ex, C, H, W = bank.shape
    dev = bank.device
    supp, qry = [], []
    for _ in range(n_tasks):
        cls = torch.randperm(num_classes, device=dev)[:n_way]
        perm = torch.randperm(num_ex, device=dev)[:n_supp + n_qry]
        sel = bank[cls][:, perm]
        supp.append(sel[:, :n_supp].reshape(n_way * n_supp, C, H, W))
        qry.append(sel[:, n_supp:].reshape(n_way * n_qry, C, H, W))
    supp_y = torch.arange(n_way, device=dev).repeat_interleave(n_supp).repeat(n_tasks)
    qry_y = torch.arange(n_way, device=dev).repeat_interleave(n_qry).repeat(n_tasks)
    return torch.cat(supp), supp_y, torch.cat(qry), qry_y


@dataclass
class TaskSplit:
    supp_idx: np.ndarray
    supp_y: np.ndarray
    test_idx: np.ndarray
    test_y: np.ndarray
    classes: List[int] = field(default_factory=list)


def make_task_split(targets, n_way, k_shot, eval_seed, max_test_per_class=None):
    """Global few-shot split of ML-NCA-Testing.py: pick n_way classes, k_shot support
    images per class, ALL remaining images of those classes are the test set.
    A private RandomState(eval_seed) reproduces np.random.seed(eval_seed) draws."""
    rng = np.random.RandomState(eval_seed)
    classes = sorted(set(targets.tolist()))
    rng.shuffle(classes)
    chosen = classes[:n_way]
    s_idx, s_y, t_idx, t_y = [], [], [], []
    for surrogate, c in enumerate(chosen):
        pos = np.where(targets == c)[0]
        rng.shuffle(pos)
        test = pos[k_shot:]
        if max_test_per_class is not None:
            test = test[:max_test_per_class]
        s_idx.extend(pos[:k_shot]); s_y.extend([surrogate] * len(pos[:k_shot]))
        t_idx.extend(test); t_y.extend([surrogate] * len(test))
    return TaskSplit(np.asarray(s_idx), np.asarray(s_y), np.asarray(t_idx), np.asarray(t_y),
                     [int(c) for c in chosen])


# ======================================================================
# 7. EVALUATION OF ONE TASK
# ======================================================================
def amp_context(device, amp_dtype):
    """autocast(amp_dtype) or a no-op context when amp_dtype is None."""
    import contextlib
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(torch.device(device).type, dtype=amp_dtype)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_task(learner, images, targets, k_shot, eval_seed, K, N,
                  device, batch_size=256, max_test_per_class=None,
                  k_eval: Optional[List[int]] = None, amp_dtype=None,
                  n_way=N_WAY):
    """Adapt on the support set of one task and score all its test images.
    Returns dict(acc={k: acc%}, adapt_time_s, peak_mem_mb, n_test).
    `k_eval` (ProgramLearner only): also score intermediate programs S_k."""
    import time
    split = make_task_split(targets, n_way, k_shot, eval_seed, max_test_per_class)
    supp_x = images[split.supp_idx].to(device)
    supp_y = torch.as_tensor(split.supp_y, device=device)
    test_x = images[split.test_idx]
    test_y = torch.as_tensor(split.test_y)

    torch.manual_seed(eval_seed)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    states = {}
    cb = None
    if k_eval and isinstance(learner, ProgramLearner):
        wanted = set(k_eval)
        cb = lambda k, S: states.__setitem__(k, S.clone()) if k in wanted else None

    dev_type = torch.device(device).type
    _sync(device); t0 = time.perf_counter()
    with torch.no_grad(), amp_context(device, amp_dtype):
        final = learner.adapt(supp_x, supp_y, 1, K, N, meta_train=False, callback=cb)
    _sync(device); adapt_time = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated(device) / 2**20) if dev_type == "cuda" else float("nan")

    to_score = {K: final}
    for k, S in states.items():
        to_score.setdefault(k, S)
    accs = {}
    for k, state in sorted(to_score.items()):
        correct = 0
        for i in range(0, test_x.shape[0], batch_size):
            xb = test_x[i:i + batch_size].to(device)
            with amp_context(device, amp_dtype):
                scores = learner.predict(xb, state, N)
            correct += (scores.argmax(1).cpu() == test_y[i:i + batch_size]).sum().item()
        accs[k] = 100.0 * correct / max(1, test_x.shape[0])
    return dict(acc=accs, adapt_time_s=adapt_time, peak_mem_mb=peak,
                n_test=int(test_x.shape[0]))


# ======================================================================
# 8. STATISTICS
# ======================================================================
def t_crit95(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
                 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        return table.get(int(df), 1.96)


def mean_sd_ci(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(a.mean())
    if n < 2:
        return m, 0.0, 0.0, n
    sd = float(a.std(ddof=1))
    return m, sd, t_crit95(n - 1) * sd / math.sqrt(n), n


def welch_diff_ci(a, b):
    """mean(a) - mean(b) with a Welch 95% CI half-width."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float(a.mean() - b.mean()) if len(a) and len(b) else float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return float(a.mean() - b.mean()), 0.0
    df = (va + vb) ** 2 / ((va ** 2) / (len(a) - 1) + (vb ** 2) / (len(b) - 1))
    return float(a.mean() - b.mean()), t_crit95(df) * se


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
