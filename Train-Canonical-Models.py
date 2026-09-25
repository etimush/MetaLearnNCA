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
    def __init__(self):
        super().__init__()
        self.backbone = Conv4(track_running_stats=True)

    def forward(self, x_supp, y_supp, x_qry, n_way):
        feat_supp = F.normalize(self.backbone(x_supp), dim=-1)
        feat_qry = F.normalize(self.backbone(x_qry), dim=-1)

        sims = torch.mm(feat_qry, feat_supp.t())
        attn = F.softmax(sims, dim=-1)

        one_hot = F.one_hot(y_supp, num_classes=n_way).float()
        probs = torch.mm(attn, one_hot)
        return torch.log(probs + 1e-8)


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

    support_idx, test_idx = [], []
    available_classes = list(set(targets))
    np.random.shuffle(available_classes)
    chosen_classes = available_classes[:n_way]
    class_to_surrogate = {true_cls: surr for surr, true_cls in enumerate(chosen_classes)}

    for true_cls in chosen_classes:
        pos = np.where(targets == true_cls)[0]
        np.random.shuffle(pos)
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
def train_model(model_name, omni_dataset, meta_epochs=1500, meta_batch_size=32):
    print(f"\n=======================================================")
    print(f" TRAINING BASELINE: {model_name} (Meta-Batch Size: {meta_batch_size})")
    print(f"=======================================================")

    if model_name == "ProtoNet":
        model = ProtoNet().to(DEVICE)
    elif model_name == "MatchingNet":
        model = MatchingNet().to(DEVICE)
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
    args = parser.parse_args()

    set_determinism(42)
    os.makedirs("./data", exist_ok=True)

    print("Loading Omniglot Meta-Training Background Dataset...")
    omni_train_dataset = get_omniglot_train_dataset()

    datasets = ["omniglot", "mnist", "fashion", "kmnist", "usps", "medmnist"]

    # Structure to hold results: results[dataset][shot][model] = (mean, ci)
    results = {d: {s: {} for s in args.shots} for d in datasets}

    for model_name in args.models:
        trained_model = train_model(
            model_name, omni_train_dataset, meta_epochs=args.epochs, meta_batch_size=args.meta_batch
        )
        trained_model.eval()

        for k_shot in args.shots:
            print(f"\nEvaluating {model_name} ({k_shot}-shot) over {args.test_runs} independent test runs...")
            print("=" * 95)
            print(
                f"{'DATASET':<12} | {'SHOT':<6} | {'FINAL MEAN ACC ± 95% CI':<25} | {'AVG TIME (s)':<14} | {'BEST SEED':<10}")
            print("=" * 95)

            for d_name in datasets:
                accs, wall_times = [], []
                for run_idx in range(args.test_runs):
                    acc, elap_t, _ = evaluate_global_dataset(
                        trained_model, model_name, d_name, n_way=10, k_shot=k_shot, eval_seed=42 + run_idx
                    )
                    accs.append(acc)
                    wall_times.append(elap_t)

                mean_acc, ci = compute_mean_and_95ci(accs)
                avg_time = np.mean(wall_times)
                best_acc = np.max(accs)

                results[d_name][k_shot][model_name] = (mean_acc, ci)

                print(
                    f"{d_name.upper():<12} | {k_shot:<6} | {mean_acc:6.2f}% ± {ci:4.2f}%             | {avg_time:10.2f}s   | {best_acc:6.2f}%")

            print("=" * 95)

    # Print the aggregate summary tables at the very end
    print_summary_tables(results, datasets, args.shots, args.models)