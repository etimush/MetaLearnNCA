import csv
import os
import re
import time
import argparse
import random
import numpy as np
from PIL import ImageFilter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Optional Scipy for exact t-distribution critical values
try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. CONFIG & DETERMINISM
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==========================================
# 1. SHARED CANONICAL BACKBONE (Conv4-64)
# ==========================================
class Conv4(nn.Module):
    """
    Standard 4-layer CNN (64 channels) with 28x28 inputs.
    track_running_stats=False ensures compatibility with MAML's per-task inner updates.
    """

    def __init__(self, in_channels=1, hidden_dim=64, track_running_stats=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7 -> 3
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 64]
        )

    def forward(self, x):
        return self.encoder(x).flatten(1)


# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================

# --- A. Prototypical Networks (Snell et al., 2017) ---
class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Conv4(track_running_stats=True)

    def forward(self, x_supp, y_supp, x_qry, n_way):
        feat_supp = self.backbone(x_supp)
        feat_qry = self.backbone(x_qry)

        prototypes = torch.stack([
            feat_supp[y_supp == c].mean(0) for c in range(n_way)
        ])

        dists = torch.cdist(feat_qry, prototypes)
        return -dists


# --- B. Matching Networks (Vinyals et al., 2016) ---
class MatchingNet(nn.Module):
    """Cosine-attention Matching Network (no full-context embeddings).

    The Conv4 features are post-ReLU, so every cosine similarity lies in [0, 1];
    an unscaled softmax over them is nearly uniform (with 10 support images a
    perfect match gets at most e/(e+9) ~ 23% of the attention), which starves
    training. As in the reference implementation of Chen et al., "A Closer Look
    at Few-shot Classification" (ICLR 2019), similarities are multiplied by
    `scale` (default 100) before the softmax. scale=1 reproduces the old model.
    """

    def __init__(self, scale=100.0):
        super().__init__()
        self.backbone = Conv4(track_running_stats=True)
        self.scale = scale

    def forward(self, x_supp, y_supp, x_qry, n_way):
        feat_supp = F.normalize(self.backbone(x_supp), dim=-1)
        feat_qry = F.normalize(self.backbone(x_qry), dim=-1)

        log_attn = F.log_softmax(self.scale * torch.mm(feat_qry, feat_supp.t()), dim=-1)  # (Q, S)

        # log p(c | x) = log sum_{i : y_i = c} attn_i, computed stably with logsumexp
        in_class = F.one_hot(y_supp, num_classes=n_way).t().bool()                          # (C, S)
        masked = log_attn.unsqueeze(1).masked_fill(~in_class.unsqueeze(0), float("-inf"))   # (Q, C, S)
        return torch.logsumexp(masked, dim=-1)                                               # (Q, C)


# --- C. Relation Networks (Sung et al., 2018) ---
class RelationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Conv4(track_running_stats=True)
        self.relation_head = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_supp, y_supp, x_qry, n_way):
        feat_supp = self.backbone(x_supp)
        feat_qry = self.backbone(x_qry)

        prototypes = torch.stack([
            feat_supp[y_supp == c].mean(0) for c in range(n_way)
        ])

        Q = feat_qry.size(0)
        q_exp = feat_qry.unsqueeze(1).expand(Q, n_way, 64)
        p_exp = prototypes.unsqueeze(0).expand(Q, n_way, 64)
        pairs = torch.cat([q_exp, p_exp], dim=-1)

        return self.relation_head(pairs.view(-1, 128)).view(Q, n_way)


# --- D. MAML (Finn et al., 2017) with Fixed Optimization & Inner Clipping ---
class MAMLNet(nn.Module):
    def __init__(self, n_way=10):
        super().__init__()
        self.backbone = Conv4(track_running_stats=False)
        self.classifier = nn.Linear(64, n_way)
        self.n_way = n_way

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def adapt(self, x_supp, y_supp, inner_steps=5, inner_lr=0.05, grad_clip=1.0, create_graph=False):
        fast_params = {name: param for name, param in self.named_parameters()}
        for _ in range(inner_steps):
            logits = torch.func.functional_call(self, fast_params, x_supp)
            loss = F.cross_entropy(logits, y_supp)
            grads = torch.autograd.grad(loss, fast_params.values(), create_graph=create_graph)

            if grad_clip is not None and grad_clip > 0:
                grads = [torch.clamp(g, -grad_clip, grad_clip) for g in grads]

            fast_params = {
                name: param - inner_lr * grad
                for (name, param), grad in zip(fast_params.items(), grads)
            }
        return fast_params


# ==========================================
# 3. EXACT FLOP COUNTER (Conv4-64)
# ==========================================
def calculate_conv4_flops(batch_size):
    return batch_size * 19.63e6


# ==========================================
# 4. EXACT DATASETS & TRANSFORMS
# ==========================================
def get_transforms():
    standard_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    fashion_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda x: x.filter(ImageFilter.FIND_EDGES)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    omniglot_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return standard_transform, fashion_transform, omniglot_transform


def get_omniglot_train_dataset():
    _, _, omniglot_transform = get_transforms()
    dataset = torchvision.datasets.Omniglot('./data', background=True, download=True, transform=omniglot_transform)
    dataset.targets = torch.tensor([target for _, target in dataset._flat_character_images])
    return dataset


def sample_train_task(dataset, n_way=10, n_support=1, n_query=5):
    targets = dataset.targets.numpy()
    available_classes = list(np.unique(targets))
    np.random.shuffle(available_classes)
    chosen_classes = available_classes[:n_way]
    class_to_surrogate = {true_cls: surr for surr, true_cls in enumerate(chosen_classes)}

    support_idx, query_idx = [], []
    for true_cls in chosen_classes:
        pos = np.where(targets == true_cls)[0]
        chosen = np.random.choice(pos, n_support + n_query, replace=False)
        support_idx.extend(chosen[:n_support])
        query_idx.extend(chosen[n_support:])

    supp_set = Subset(dataset, support_idx)
    qry_set = Subset(dataset, query_idx)

    supp_loader = DataLoader(supp_set, batch_size=len(supp_set), shuffle=False)
    qry_loader = DataLoader(qry_set, batch_size=len(qry_set), shuffle=False)

    supp_x, supp_y_orig = next(iter(supp_loader))
    qry_x, qry_y_orig = next(iter(qry_loader))

    supp_y = torch.zeros_like(supp_y_orig)
    qry_y = torch.zeros_like(qry_y_orig)

    for true_cls, surr in class_to_surrogate.items():
        supp_y[supp_y_orig == true_cls] = surr
        qry_y[qry_y_orig == true_cls] = surr

    return supp_x.to(DEVICE), supp_y.to(DEVICE), qry_x.to(DEVICE), qry_y.to(DEVICE)


def get_global_splits(dataset_name, n_way=10, k_shot=10, eval_seed=42):
    standard_transform, fashion_transform, omniglot_transform = get_transforms()

    if dataset_name == "mnist":
        dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=standard_transform)
        targets = dataset.targets.numpy()
    elif dataset_name == "fashion":
        dataset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=fashion_transform)
        targets = dataset.targets.numpy()
    elif dataset_name == "kmnist":
        dataset = torchvision.datasets.KMNIST('./data', train=False, download=True, transform=standard_transform)
        targets = dataset.targets.numpy()
    elif dataset_name == "usps":
        dataset = torchvision.datasets.USPS('./data', train=False, download=True, transform=standard_transform)
        targets = np.array(dataset.targets)
    elif dataset_name == "medmnist":
        import medmnist
        info = medmnist.INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', download=True, transform=fashion_transform)
        targets = dataset.labels.flatten()
    elif dataset_name == "omniglot":
        dataset = torchvision.datasets.Omniglot('./data', background=False, download=True, transform=omniglot_transform)
        targets = np.array([y for _, y in dataset])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # The task (which classes, which support images) is a function of eval_seed only.
    # A private RandomState(eval_seed) draws exactly what np.random.seed(eval_seed) +
    # np.random.shuffle draw in ML-NCA-Testing.py, so for the same eval_seed the
    # baselines are scored on the same tasks as MetaLearnNCA (Table 1), independently
    # of how much global RNG the preceding meta-training consumed.
    rng = np.random.RandomState(eval_seed)
    support_idx, test_idx = [], []
    available_classes = sorted(set(np.asarray(targets).tolist()))
    rng.shuffle(available_classes)
    chosen_classes = available_classes[:n_way]
    class_to_surrogate = {true_cls: surr for surr, true_cls in enumerate(chosen_classes)}

    for true_cls in chosen_classes:
        pos = np.where(targets == true_cls)[0]
        rng.shuffle(pos)
        support_idx.extend(pos[:k_shot])
        test_idx.extend(pos[k_shot:])

    class RemappedSubset(torch.utils.data.Dataset):
        def __init__(self, ds, indices, class_map):
            self.ds = ds
            self.indices = indices
            self.class_map = class_map

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, true_label = self.ds[self.indices[idx]]
            if isinstance(true_label, (np.ndarray, list)):
                true_label = int(true_label[0])
            elif torch.is_tensor(true_label):
                true_label = int(true_label.item())
            return img, self.class_map[true_label]

    supp_set = RemappedSubset(dataset, support_idx, class_to_surrogate)
    test_set = RemappedSubset(dataset, test_idx, class_to_surrogate)

    supp_loader = DataLoader(supp_set, batch_size=len(supp_set), shuffle=False)

    gen = torch.Generator()
    gen.manual_seed(eval_seed)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, generator=gen)

    return supp_loader, test_loader


# ==========================================
# 5. TRAINING ROUTINE
# ==========================================
def train_model(model_name, omni_dataset, meta_epochs=1500, meta_batch_size=32, matching_scale=100.0):
    print(f"\n=======================================================")
    print(f" TRAINING BASELINE: {model_name} (Meta-Batch Size: {meta_batch_size})")
    print(f"=======================================================")

    if model_name == "ProtoNet":
        model = ProtoNet().to(DEVICE)
    elif model_name == "MatchingNet":
        model = MatchingNet(scale=matching_scale).to(DEVICE)
    elif model_name == "RelationNet":
        model = RelationNet().to(DEVICE)
    elif model_name == "MAML":
        model = MAMLNet(n_way=10).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()

    pbar = tqdm(range(1, meta_epochs + 1), desc=f"Training {model_name}")
    for epoch in pbar:
        optimizer.zero_grad()
        epoch_loss = 0.0

        for task_idx in range(meta_batch_size):
            supp_x, supp_y, qry_x, qry_y = sample_train_task(omni_dataset, n_way=10, n_support=1, n_query=5)

            if model_name in ["ProtoNet", "MatchingNet"]:
                logits = model(supp_x, supp_y, qry_x, 10)
                loss = F.cross_entropy(logits, qry_y) / meta_batch_size
            elif model_name == "RelationNet":
                scores = model(supp_x, supp_y, qry_x, 10)
                y_qry_onehot = F.one_hot(qry_y, num_classes=10).float()
                loss = F.mse_loss(scores, y_qry_onehot) / meta_batch_size
            elif model_name == "MAML":
                fast_weights = model.adapt(supp_x, supp_y, inner_steps=3, inner_lr=0.1, grad_clip=1.0,
                                           create_graph=False)
                qry_logits = torch.func.functional_call(model, fast_weights, qry_x)
                loss = F.cross_entropy(qry_logits, qry_y) / meta_batch_size

            loss.backward()
            epoch_loss += loss.item() * meta_batch_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 50 == 0:
            pbar.set_postfix({"MetaLoss": f"{epoch_loss / meta_batch_size:.4f}"})

    return model


# ==========================================
# 6. GLOBAL EVALUATION (EXHAUSTIVE TEST SET)
# ==========================================
def evaluate_global_dataset(model, model_name, dataset_name, n_way=10, k_shot=10, eval_seed=42):
    supp_loader, test_loader = get_global_splits(dataset_name, n_way=n_way, k_shot=k_shot, eval_seed=eval_seed)
    supp_x, supp_y = next(iter(supp_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    B_supp = supp_x.size(0)

    total_flops = 0.0
    start_time = time.perf_counter()

    if model_name == "MAML":
        # Omniglot needs larger inner steps; complex OOD datasets need smaller, regularized steps
        if dataset_name == "omniglot":
            inner_lr = 0.2 if k_shot <= 5 else 0.1
            grad_clip = None  # Don't choke gradients on clean Omniglot strokes
            inner_steps = 5
        else:
            # OOD transfer (MNIST, Fashion, USPS, etc.)
            if k_shot <= 1:
                inner_lr = 0.1
            elif k_shot <= 5:
                inner_lr = 0.05
            else:
                inner_lr = 0.03
            grad_clip = 1.0
            inner_steps = 5

        with torch.enable_grad():
            fast_weights = model.adapt(
                supp_x, supp_y,
                inner_steps=inner_steps,
                inner_lr=inner_lr,
                grad_clip=grad_clip,
                create_graph=False
            )



        total_flops += 5 * 3 * calculate_conv4_flops(B_supp)
    elif model_name in ["ProtoNet", "MatchingNet", "RelationNet"]:
        total_flops += calculate_conv4_flops(B_supp)

    correct = 0
    total = 0

    with torch.no_grad():
        for qry_x, qry_y in test_loader:
            qry_x, qry_y = qry_x.to(DEVICE), qry_y.to(DEVICE)
            B_qry = qry_x.size(0)
            total_flops += calculate_conv4_flops(B_qry)

            if model_name in ["ProtoNet", "MatchingNet"]:
                logits = model(supp_x, supp_y, qry_x, n_way)
                preds = logits.argmax(dim=-1)
            elif model_name == "RelationNet":
                scores = model(supp_x, supp_y, qry_x, n_way)
                preds = scores.argmax(dim=-1)
            elif model_name == "MAML":
                qry_logits = torch.func.functional_call(model, fast_weights, qry_x)
                preds = qry_logits.argmax(dim=-1)

            correct += (preds == qry_y).sum().item()
            total += qry_y.size(0)

    elapsed_time = time.perf_counter() - start_time
    acc = (correct / total) * 100.0
    return acc, elapsed_time, total_flops


def compute_mean_and_95ci(data_list):
    arr = np.array(data_list)
    n = len(arr)
    mean = np.mean(arr)
    if n < 2:
        return mean, 0.0
    std_err = np.std(arr, ddof=1) / np.sqrt(n)
    if HAS_SCIPY:
        t_crit = stats.t.ppf(0.975, df=n - 1)
    else:
        t_crit = 1.96 if n > 30 else 2.0
    return mean, std_err * t_crit


# ==========================================
# 7. SUMMARY TABLE FORMATTERS (ASCII & LATEX)
# ==========================================
def print_summary_tables(results, datasets, shots, models):
    """
    results format: results[dataset][shot][model_name] = (mean, ci)
    """
    col_w = 22

    # ---------------- ASCII Table ----------------
    print("\n" + "=" * 115)
    print("                      ALL-MODELS COMPLETE BENCHMARK SUMMARY TABLE")
    print("=" * 115)
    header = f"{'DATASET':<14} | {'SHOT':<5} | " + " | ".join([f"{m:^{col_w}}" for m in models])
    print(header)
    print("-" * len(header))

    for d_name in datasets:
        for s_idx, k_shot in enumerate(shots):
            ds_str = d_name.upper() if s_idx == 0 else ""
            cells = []
            for m in models:
                if m in results[d_name][k_shot]:
                    mean, ci = results[d_name][k_shot][m]
                    cells.append(f"{mean:6.2f}% ± {ci:4.2f}%".center(col_w))
                else:
                    cells.append("N/A".center(col_w))
            print(f"{ds_str:<14} | {k_shot:<5} | " + " | ".join(cells))
        print("-" * len(header))

    # ---------------- LaTeX Table ----------------
    print("\n" + "=" * 115)
    print("                   LATEX CODE (COPY-PASTE READY FOR YOUR ICLR PAPER)")
    print("=" * 115)

    col_spec = "ll" + "c" * len(models)
    latex_header = "Dataset & Shot & " + " & ".join(models) + " \\\\"

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\small")
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    print(latex_header)
    print("\\midrule")

    for d_idx, d_name in enumerate(datasets):
        for s_idx, k_shot in enumerate(shots):
            ds_str = f"\\textbf{{{d_name.title()}}}" if s_idx == 0 else ""
            cells = []
            for m in models:
                if m in results[d_name][k_shot]:
                    mean, ci = results[d_name][k_shot][m]
                    cells.append(f"${mean:.2f} \\pm {ci:.2f}\\%$")
                else:
                    cells.append("--")
            print(f"{ds_str:<20} & {k_shot:<2} & " + " & ".join(cells) + " \\\\")
        if d_idx < len(datasets) - 1:
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Few-shot classification accuracy across 1-shot, 5-shot, and 10-shot cross-domain transfers.}")
    print("\\label{tab:few_shot_full_results}")
    print("\\end{table}\n")


# ==========================================
# 8. MAIN BENCHMARK RUNNER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Meta-Learning Baselines aligned with MetaNCA.")
    parser.add_argument("--epochs", type=int, default=5000, help="Meta-epochs on Omniglot (default: 5000)")
    parser.add_argument("--meta_batch", type=int, default=32, help="Meta-batch size (default: 32)")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10],
                        help="Support shot counts to test (e.g. --shots 1 5 10)")
    parser.add_argument("--test_runs", type=int, default=10,
                        help="Independent evaluation runs for 95% CI (default: 10)")
    parser.add_argument("--models", nargs="+", default=["MAML","ProtoNet", "MatchingNet", "RelationNet"], #,
                        help="Models to run: ProtoNet MatchingNet RelationNet MAML")
    parser.add_argument("--eval_seed_start", type=int, default=43,
                        help="Evaluation run i uses eval_seed = eval_seed_start + i. The default (43..52 for "
                             "10 runs) matches ML-NCA-Testing.py, so baselines and MetaLearnNCA are scored on "
                             "identical tasks.")
    parser.add_argument("--matching_scale", type=float, default=100.0,
                        help="MatchingNet: cosine-similarity scale before the softmax (Chen et al., 2019). "
                             "1.0 reproduces the submitted version.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed, re-applied before each model so a model's result does not "
                             "depend on which models were trained before it.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Several training seeds (overrides --seed), e.g. --seeds 0 1 2 3 4")
    parser.add_argument("--datasets", nargs="+", default=["omniglot", "mnist", "fashion", "kmnist", "usps", "medmnist"])
    parser.add_argument("--results_dir", default="baseline_results",
                        help="Per-task results are appended to <results_dir>/results_raw.csv and trained "
                             "models saved to <results_dir>/checkpoints/. Combine seeds with "
                             "ablations/make_table1.py.")
    parser.add_argument("--no_skip", action="store_true",
                        help="Retrain (model, seed) pairs whose results are already in results_raw.csv")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else [args.seed]

    os.makedirs("./data", exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "checkpoints"), exist_ok=True)
    raw_path = os.path.join(args.results_dir, "results_raw.csv")
    raw_fields = ["model", "train_seed", "dataset", "shot", "eval_seed", "acc", "time_s", "n_params",
                  "epochs", "meta_batch", "matching_scale"]

    def config_of(model_name):
        return dict(epochs=str(args.epochs), meta_batch=str(args.meta_batch),
                    matching_scale=str(args.matching_scale) if model_name == "MatchingNet" else "")

    def already_done(model_name, seed):
        if args.no_skip or not os.path.exists(raw_path):
            return False
        cfg = config_of(model_name)
        need = {(d, str(k), str(args.eval_seed_start + r)) for d in args.datasets for k in args.shots
                for r in range(args.test_runs)}
        with open(raw_path) as f:
            have = {(r["dataset"], r["shot"], r["eval_seed"]) for r in csv.DictReader(f)
                    if r["model"] == model_name and r["train_seed"] == str(seed)
                    and all(r[c] == v for c, v in cfg.items())}
        return need <= have

    set_determinism(seeds[0])
    print("Loading Omniglot Meta-Training Background Dataset...")
    omni_train_dataset = get_omniglot_train_dataset()

    datasets = args.datasets
    new_file = not os.path.exists(raw_path)
    raw_fh = open(raw_path, "a", newline="")
    raw_writer = csv.DictWriter(raw_fh, fieldnames=raw_fields)
    if new_file:
        raw_writer.writeheader()

    for seed in seeds:
        # Structure to hold results: results[dataset][shot][model] = (mean, ci) for this training seed
        results = {d: {s: {} for s in args.shots} for d in datasets}

        for model_name in args.models:
            if already_done(model_name, seed):
                print(f"[skip] {model_name} seed {seed}: results already in {raw_path} (use --no_skip)")
                continue
            set_determinism(seed)
            trained_model = train_model(
                model_name, omni_train_dataset, meta_epochs=args.epochs, meta_batch_size=args.meta_batch,
                matching_scale=args.matching_scale
            )
            trained_model.eval()
            n_params = sum(p.numel() for p in trained_model.parameters())
            torch.save(trained_model.state_dict(),
                       os.path.join(args.results_dir, "checkpoints", f"{model_name}_seed{seed}.pth"))

            for k_shot in args.shots:
                print(f"\nEvaluating {model_name} seed {seed} ({k_shot}-shot) over {args.test_runs} independent test runs...")
                print("=" * 95)
                print(
                    f"{'DATASET':<12} | {'SHOT':<6} | {'FINAL MEAN ACC ± 95% CI':<25} | {'AVG TIME (s)':<14} | {'BEST SEED':<10}")
                print("=" * 95)

                for d_name in datasets:
                    accs, wall_times = [], []
                    for run_idx in range(args.test_runs):
                        eval_seed = args.eval_seed_start + run_idx
                        acc, elap_t, _ = evaluate_global_dataset(
                            trained_model, model_name, d_name, n_way=10, k_shot=k_shot, eval_seed=eval_seed
                        )
                        accs.append(acc)
                        wall_times.append(elap_t)
                        raw_writer.writerow(dict(model=model_name, train_seed=seed, dataset=d_name, shot=k_shot,
                                                 eval_seed=eval_seed, acc=f"{acc:.4f}", time_s=f"{elap_t:.4f}",
                                                 n_params=n_params, **config_of(model_name)))
                    raw_fh.flush()

                    mean_acc, ci = compute_mean_and_95ci(accs)
                    avg_time = np.mean(wall_times)
                    best_acc = np.max(accs)

                    results[d_name][k_shot][model_name] = (mean_acc, ci)

                    print(
                        f"{d_name.upper():<12} | {k_shot:<6} | {mean_acc:6.2f}% ± {ci:4.2f}%             | {avg_time:10.2f}s   | {best_acc:6.2f}%")

                print("=" * 95)

        # Per-seed tables (CI over evaluation tasks only, i.e. ignoring training variance)
        trained_now = [m for m in args.models if any(m in results[d][k] for d in datasets for k in args.shots)]
        if trained_now:
            print(f"\n##### Training seed {seed} #####")
            print_summary_tables(results, datasets, args.shots, trained_now)
    raw_fh.close()

    print(f"\nPer-task results: {raw_path}")
    print("Combine training seeds (mean ± SD) and add MetaLearnNCA with:\n"
          f"  python ablations/make_table1.py --baseline_csv {raw_path} --nca_csv ablation_results/results_raw.csv")
