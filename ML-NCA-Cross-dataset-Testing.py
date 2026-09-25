import os
import csv
import time
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import ImageFilter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Exact Student-t critical values
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. HYPERPARAMETERS & CONFIG
# ==========================================
ACTIVE_DYN_CHN = 32
META_DYN_CHN = 32
STATIC_CHN = 24
LOSS_CHN = 10
HIDDEN_DIM = 256

N_EVAL_SEEDS = 50   # 50 random evaluation seeds per cell
K_REPEATS = 5       # Evaluation repeats (k = 5)
N_STEPS = 19        # Base NCA steps per interaction

CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluated datasets and their exact protocol k_shot
DATASET_CONFIGS = [
    ("omniglot", 10),
    ("mnist", 100),
    ("fashion", 100),
    ("kmnist", 100),
    ("usps", 10),
    ("medmnist", 100),
]
DATASET_NAMES = [d[0] for d in DATASET_CONFIGS]

# Omniglot explicitly takes seed0; others take dataset-specific checkpoints
CHECKPOINT_MAP = {
    "omniglot": ("active_nca_seed0.pth", "meta_nca_seed0.pth"),
    "mnist": ("active_nca_mnist.pth", "meta_nca_mnist.pth"),
    "fashion": ("active_nca_fashion.pth", "meta_nca_fashion.pth"),
    "kmnist": ("active_nca_kmnist.pth", "meta_nca_kmnist.pth"),
    "usps": ("active_nca_usps.pth", "meta_nca_usps.pth"),
    "medmnist": ("active_nca_medmnist.pth", "meta_nca_medmnist.pth"),
}


def set_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================
class BaseNCA(nn.Module):
    def __init__(self, c_in, c_out, hidden_size):
        super().__init__()
        self.c_in = c_in
        ident = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        lap = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
        base_filters = torch.stack([ident, sobel_x, sobel_x.T, lap])

        grouped_filters = base_filters.repeat(c_in, 1, 1).unsqueeze(1)
        self.register_buffer('filters', grouped_filters)

        self.w1 = nn.Conv2d(c_in * 4, hidden_size, 1)
        self.w2 = nn.Conv2d(hidden_size, c_out, 1, bias=False)
        self.w2.weight.data.zero_()

    def perceive(self, x):
        x_pad = F.pad(x, [1, 1, 1, 1], mode='circular')
        return F.conv2d(x_pad, self.filters, groups=self.c_in)

    def grid_update(self, x):
        return self.w2(F.relu(self.w1(self.perceive(x)), inplace=True))


class ActiveNCA(BaseNCA):
    def __init__(self):
        c_in = 1 + STATIC_CHN + ACTIVE_DYN_CHN
        super().__init__(c_in, ACTIVE_DYN_CHN, hidden_size=HIDDEN_DIM)

    def forward(self, img, static_channels, steps=N_STEPS):
        B, _, H, W = img.shape
        dyn = torch.rand(B, ACTIVE_DYN_CHN, H, W, device=img.device)

        for _ in range(steps):
            x = torch.cat([img, static_channels, dyn], dim=1)
            ds = self.grid_update(x)
            mask = (torch.rand(B, 1, H, W, device=img.device) > 0.5).float()
            dyn = dyn + ds * mask
        return dyn


class MetaNCA(BaseNCA):
    def __init__(self):
        c_in = LOSS_CHN + META_DYN_CHN
        super().__init__(c_in, META_DYN_CHN, hidden_size=HIDDEN_DIM)

    def forward(self, img, pixel_loss, dyn_state=None, steps=N_STEPS):
        B, _, H, W = img.shape
        if dyn_state is None:
            dyn_state = torch.rand(B, META_DYN_CHN, H, W, device=img.device)

        for _ in range(steps):
            x = torch.cat([pixel_loss, dyn_state], dim=1)
            ds = self.grid_update(x)
            mask = (torch.rand(B, 1, H, W, device=img.device) > 0.5).float()
            dyn_state = dyn_state + ds * mask
        return dyn_state


def get_pixel_loss(pred_dyn, targets, images):
    alive_mask = (images > -0.8).float()
    pred_logits = pred_dyn[:, :LOSS_CHN, :, :]
    probs = F.softmax(pred_logits, dim=1)

    spatial_targets = torch.zeros_like(pred_logits)
    for b in range(images.size(0)):
        spatial_targets[b, targets[b], :, :] = 1.0

    pixel_error = (spatial_targets - probs) * alive_mask
    return pixel_error, pred_logits, alive_mask


# ==========================================
# 2. EXACT ANALYTICAL FLOP COUNTER
# ==========================================
def calculate_step_flops(c_in, c_out, hidden_size, batch_size, H=28, W=28):
    elements = batch_size * H * W
    flops_perceive = elements * (4 * c_in) * (2 * 3 * 3)
    flops_w1 = elements * (2 * (4 * c_in) * hidden_size + hidden_size)
    flops_relu = elements * hidden_size
    flops_w2 = elements * (2 * hidden_size * c_out)
    flops_update = elements * (2 * c_out)
    return flops_perceive + flops_w1 + flops_relu + flops_w2 + flops_update


# ==========================================
# 3. DATASET CACHING & EXACT GLOBAL SPLITS
# ==========================================
_DATASET_CACHE = {}


def get_cached_raw_dataset(dataset_name):
    """Loads and caches the raw torchvision dataset in RAM once."""
    dname = dataset_name.lower()
    if dname in _DATASET_CACHE:
        return _DATASET_CACHE[dname]

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

    if dname == "mnist":
        dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=standard_transform)
        targets = dataset.targets.numpy()
    elif dname == "fashion":
        dataset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=fashion_transform)
        targets = dataset.targets.numpy()
    elif dname == "kmnist":
        dataset = torchvision.datasets.KMNIST('./data', train=False, download=True, transform=standard_transform)
        targets = dataset.targets.numpy()
    elif dname == "usps":
        dataset = torchvision.datasets.USPS('./data', train=False, download=True, transform=standard_transform)
        targets = np.array(dataset.targets)
    elif dname == "medmnist":
        try:
            import medmnist
        except ImportError:
            raise ImportError("Please install medmnist: pip install medmnist")
        info = medmnist.INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', download=True, transform=fashion_transform)
        targets = dataset.labels.flatten()
    elif dname == "omniglot":
        dataset = torchvision.datasets.Omniglot('./data', background=False, download=True, transform=omniglot_transform)
        targets = np.array([y for _, y in dataset])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    _DATASET_CACHE[dname] = (dataset, targets)
    return dataset, targets


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


def get_global_splits(dataset_name, n_way=10, k_shot=100):
    """
    Exact protocol: splits dataset into k_shot support set and evaluates
    on the entire remaining test set.
    """
    dataset, targets = get_cached_raw_dataset(dataset_name)

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

    supp_set = RemappedSubset(dataset, support_idx, class_to_surrogate)
    test_set = RemappedSubset(dataset, test_idx, class_to_surrogate)

    supp_loader = DataLoader(supp_set, batch_size=len(supp_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    return supp_loader, test_loader


# ==========================================
# 4. SINGLE EVALUATION RUNNER (FULL TEST SET)
# ==========================================
def evaluate_model_on_dataset(active_nca, meta_nca, dataset_name, k_shot=100, k_repeats=K_REPEATS):
    supp_loader, test_loader = get_global_splits(dataset_name, n_way=10, k_shot=k_shot)

    supp_x, supp_y = next(iter(supp_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    B_supp = supp_x.size(0)

    S_task = torch.rand(1, STATIC_CHN, supp_x.size(2), supp_x.size(3), device=DEVICE)
    D_meta = None

    # Support adaptation over K_REPEATS
    with torch.no_grad():
        for _ in range(k_repeats):
            S_expanded = S_task.expand(B_supp, -1, -1, -1)
            supp_dyn = active_nca(supp_x, S_expanded, steps=N_STEPS * 2)
            pixel_loss, _, _ = get_pixel_loss(supp_dyn, supp_y, supp_x)

            D_meta = meta_nca(supp_x, pixel_loss, dyn_state=D_meta, steps=N_STEPS)
            S_task = D_meta[:, :STATIC_CHN, :, :].mean(dim=0, keepdim=True)

        # Full Test Set Evaluation
        correct = 0
        total = 0
        for qry_x, qry_y in test_loader:
            qry_x, qry_y = qry_x.to(DEVICE), qry_y.to(DEVICE)
            B_qry = qry_x.size(0)

            S_expanded_qry = S_task.expand(B_qry, -1, -1, -1)
            qry_dyn = active_nca(qry_x, S_expanded_qry, steps=N_STEPS * 2)

            _, qry_pred, alive_mask = get_pixel_loss(qry_dyn, qry_y, qry_x)
            logits = (qry_pred * alive_mask).sum(dim=(2, 3))
            preds = logits.argmax(dim=1)

            correct += (preds == qry_y).sum().item()
            total += qry_y.size(0)

    return (correct / total) * 100.0


# ==========================================
# 5. STATISTICAL UTILITY (95% CONFIDENCE INTERVAL)
# ==========================================
def compute_mean_and_95ci(scores):
    arr = np.array(scores)
    n = len(arr)
    mean = np.mean(arr)
    if n < 2:
        return mean, 0.0

    sem = np.std(arr, ddof=1) / np.sqrt(n)
    if HAS_SCIPY:
        t_crit = stats.t.ppf(0.975, df=n - 1)
    else:
        # Standard Student-t lookup table for 95% two-tailed confidence
        t_table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 49: 2.0096, 50: 2.0086
        }
        df = n - 1
        t_crit = t_table.get(df, 1.96 if df > 30 else 2.0)

    ci = sem * t_crit
    return mean, ci


# ==========================================
# 6. MAIN BENCHMARK SCRIPT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MetaNCA Full-Dataset Cross-Domain Transfer Benchmark")
    parser.add_argument("--seeds", type=int, default=N_EVAL_SEEDS, help="Number of random evaluation seeds (default: 50)")
    parser.add_argument("--k_repeats", type=int, default=K_REPEATS, help="Support optimization repeats (default: 5)")
    args = parser.parse_args()

    num_eval_seeds = args.seeds
    k_reps = args.k_repeats

    print("=" * 95)
    print(f" FULL-DATASET CROSS-DOMAIN BENCHMARK: {len(DATASET_NAMES)}x{len(DATASET_NAMES)} TRANSFER MATRIX")
    print(f" Protocol: 10-Way, Dataset-Specific n-Shot Support -> Test on ENTIRE Remaining Dataset")
    print(f" Evaluation Seeds: {num_eval_seeds} per pair | Device: {DEVICE}")
    print("=" * 95)

    # Pre-cache all datasets in RAM once
    print("Pre-caching datasets into memory...")
    for dname, _ in DATASET_CONFIGS:
        get_cached_raw_dataset(dname)
    print("All datasets loaded.")

    active_nca = ActiveNCA().to(DEVICE)
    meta_nca = MetaNCA().to(DEVICE)
    active_nca.eval()
    meta_nca.eval()

    num_ds = len(DATASET_NAMES)
    mean_matrix = np.zeros((num_ds, num_ds))
    ci_matrix = np.zeros((num_ds, num_ds))
    string_matrix = [["" for _ in range(num_ds)] for _ in range(num_ds)]
    times_matrix = np.zeros((num_ds, num_ds))

    # Evaluate each trained model against all target datasets
    for i, train_d in enumerate(DATASET_NAMES):
        act_file, meta_file = CHECKPOINT_MAP[train_d]
        act_path = os.path.join(CHECKPOINT_DIR, act_file)
        meta_path = os.path.join(CHECKPOINT_DIR, meta_file)

        # Fallback check
        if not os.path.exists(act_path) or not os.path.exists(meta_path):
            alt_act = os.path.join(CHECKPOINT_DIR, f"active_nca_{train_d}.pth")
            alt_meta = os.path.join(CHECKPOINT_DIR, f"meta_nca_{train_d}.pth")
            if os.path.exists(alt_act) and os.path.exists(alt_meta):
                act_path, meta_path = alt_act, alt_meta
            else:
                print(f"\n[WARNING] Checkpoint '{act_file}' not found in '{CHECKPOINT_DIR}'. Skipping row {train_d.upper()}.")
                for j in range(num_ds):
                    string_matrix[i][j] = "N/A"
                continue

        print(f"\n[{i+1}/{num_ds}] Evaluating Model Trained On: {train_d.upper()} ({act_file})")
        active_nca.load_state_dict(torch.load(act_path, map_location=DEVICE))
        meta_nca.load_state_dict(torch.load(meta_path, map_location=DEVICE))

        for j, (test_d, k_shot) in enumerate(DATASET_CONFIGS):
            pair_start = time.perf_counter()
            scores = []

            pbar = tqdm(range(num_eval_seeds), desc=f"  -> Testing on {test_d.upper():<9} ({k_shot}-shot)", leave=False)
            for eval_seed in pbar:
                set_determinism(eval_seed + 1000)
                acc = evaluate_model_on_dataset(active_nca, meta_nca, test_d, k_shot=k_shot, k_repeats=k_reps)
                scores.append(acc)

            mean_acc, ci_acc = compute_mean_and_95ci(scores)
            elapsed = time.perf_counter() - pair_start

            mean_matrix[i, j] = mean_acc
            ci_matrix[i, j] = ci_acc
            times_matrix[i, j] = elapsed / num_eval_seeds
            string_matrix[i][j] = f"{mean_acc:5.1f} ± {ci_acc:3.1f}%"

            tag = " (In-Domain)" if i == j else ""
            print(f"  [{test_d.upper():<9}] Full Test Set: {mean_acc:5.1f} ± {ci_acc:3.1f}%{tag} (avg {times_matrix[i, j]:.2f}s/run)")

    # ==========================================
    # 7. PRINT FINAL ASCII TABLE
    # ==========================================
    col_headers = [d.upper() for d in DATASET_NAMES]
    header_str = f"{'TRAINED ON v \\ EVAL ON >':<26} | " + " | ".join([f"{h:^15}" for h in col_headers])
    divider = "=" * len(header_str)

    print("\n" + divider)
    print(header_str)
    print(divider)

    for i, train_d in enumerate(DATASET_NAMES):
        row_str = f"{train_d.upper():<26} | " + " | ".join([f"{string_matrix[i][j]:^15}" for j in range(num_ds)])
        print(row_str)

    print(divider)

    # ==========================================
    # 8. EXPORT CSV
    # ==========================================
    csv_file = "cross_dataset_transfer_matrix_full.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Trained On \\ Evaluated On"] + col_headers)
        for i, train_d in enumerate(DATASET_NAMES):
            writer.writerow([train_d.upper()] + string_matrix[i])

    print(f"\nSaved CSV results to '{csv_file}'.")

    # ==========================================
    # 9. PUBLICATION HEATMAP
    # ==========================================
    plt.figure(figsize=(11, 8.5))
    plt.imshow(mean_matrix, cmap="YlGnBu", vmin=10, vmax=100)
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label("Full Test Set Accuracy (%)", fontsize=11, fontweight="bold")

    plt.xticks(np.arange(num_ds), col_headers, fontsize=10, rotation=20)
    plt.yticks(np.arange(num_ds), col_headers, fontsize=10)
    plt.xlabel("Evaluated On (Target Domain)", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("Trained On (Source Domain)", fontsize=12, fontweight="bold", labelpad=10)
    plt.title(
        f"MetaNCA Cross-Domain Transfer Matrix (Full Test Set)\n"
        f"(Mean ± 95% CI across {num_eval_seeds} Evaluation Seeds)",
        fontsize=13, fontweight="bold", pad=15
    )

    for i in range(num_ds):
        for j in range(num_ds):
            if string_matrix[i][j] != "N/A":
                txt = f"{mean_matrix[i, j]:.1f}%\n±{ci_matrix[i, j]:.1f}%"
                text_color = "white" if mean_matrix[i, j] > 55 else "black"
                font_weight = "bold" if i == j else "normal"
                plt.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=9.5, fontweight=font_weight)
            else:
                plt.text(j, i, "N/A", ha="center", va="center", color="gray", fontsize=9.5)

    plt.tight_layout()
    plot_name = "cross_dataset_transfer_heatmap_full.png"
    plt.savefig(plot_name, dpi=300)
    print(f"Saved publication heatmap to '{plot_name}'.")
    plt.show()


if __name__ == '__main__':
    main()