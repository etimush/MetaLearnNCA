import os
import time
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import ImageFilter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
"""Script to compute optimization surface of ML-NCA"""
# ==========================================
# 0. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTIVE_DYN_CHN = 32
META_DYN_CHN = 32
STATIC_CHN = 24
LOSS_CHN = 10
HIDDEN_DIM = 256

N_STEPS = 19
K_VALS = [1, 2, 3, 4, 5]  # k repeats from 1 to 5
N_EVAL_SEEDS = 3  # Average over 3 seeds per (shot, k) point for smooth 3D surfaces

# Dataset shot schedules
DATASET_SCHEDULES = [
    ("omniglot", [1, 2, 3, 4, 5, 7, 10]),
    ("kmnist", [1, 5, 10, 20, 30, 50, 100]),
    ("mnist", [1, 5, 10, 20, 30, 50, 100]),
    ("fashion", [1, 5, 10, 20, 30, 50, 100]),
    ("usps", [1, 2, 3, 4, 5, 7, 10]),
    ("medmnist", [1, 5, 10, 20, 30, 50, 100]),
]

CHECKPOINT_DIR = "checkpoints"
OMNIGLOT_CHECKPOINTS = ("active_nca_seed0.pth", "meta_nca_seed0.pth")


def set_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==========================================
# 1. MODEL DEFINITIONS (DEPTHWISE GROUPED)
# ==========================================
class BaseNCA(nn.Module):
    def __init__(self, c_in, c_out, hidden_size):
        super().__init__()
        self.c_in = c_in
        ident = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        lap = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
        base_filters = torch.stack([ident, sobel_x, sobel_y, lap])

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
        c_in = 1 + STATIC_CHN + ACTIVE_DYN_CHN  # 1 + 24 + 32 = 57
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
        c_in = LOSS_CHN + META_DYN_CHN  # 10 + 32 = 42
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
# 2. DATASET CACHING & SPLITTING
# ==========================================
_DATASET_CACHE = {}


def get_cached_raw_dataset(dataset_name):
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


def get_global_splits(dataset_name, n_way=10, k_shot=10):
    dataset, targets = get_cached_raw_dataset(dataset_name)

    support_idx, test_idx = [], []
    available_classes = list(set(targets))
    np.random.shuffle(available_classes)
    chosen_classes = available_classes[:n_way]
    class_to_surrogate = {true_cls: surr for surr, true_cls in enumerate(chosen_classes)}

    for true_cls in chosen_classes:
        pos = np.where(targets == true_cls)[0]
        np.random.shuffle(pos)
        actual_shot = min(k_shot, len(pos) - 1)  # Ensure at least 1 image remains for test
        support_idx.extend(pos[:actual_shot])
        test_idx.extend(pos[actual_shot:])

    supp_set = RemappedSubset(dataset, support_idx, class_to_surrogate)
    test_set = RemappedSubset(dataset, test_idx, class_to_surrogate)

    supp_loader = DataLoader(supp_set, batch_size=len(supp_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    return supp_loader, test_loader


# ==========================================
# 3. TRAJECTORY EVALUATOR (k = 1 to 5)
# ==========================================
def evaluate_trajectory(active_nca, meta_nca, dataset_name, k_shot, max_k=5):
    """
    Runs support adaptation iteratively from k=1 to max_k.
    Evaluates on the test set at each k repeat.
    Returns: dict {k: accuracy} for k in 1..max_k
    """
    supp_loader, test_loader = get_global_splits(dataset_name, n_way=10, k_shot=k_shot)
    supp_x, supp_y = next(iter(supp_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    B_supp = supp_x.size(0)

    S_task = torch.rand(1, STATIC_CHN, supp_x.size(2), supp_x.size(3), device=DEVICE)
    D_meta = None
    k_accuracies = {}

    def test_acc(S_current):
        correct, total = 0, 0
        with torch.no_grad():
            for qry_x, qry_y in test_loader:
                qry_x, qry_y = qry_x.to(DEVICE), qry_y.to(DEVICE)
                B_qry = qry_x.size(0)

                S_expanded_qry = S_current.expand(B_qry, -1, -1, -1)
                qry_dyn = active_nca(qry_x, S_expanded_qry, steps=N_STEPS * 2)

                _, qry_pred, alive_mask = get_pixel_loss(qry_dyn, qry_y, qry_x)
                logits = (qry_pred * alive_mask).sum(dim=(2, 3))
                preds = logits.argmax(dim=1)

                correct += (preds == qry_y).sum().item()
                total += qry_y.size(0)
        return (correct / total) * 100.0

    with torch.no_grad():
        for k in range(1, max_k + 1):
            # Adapt S_task on support set
            S_expanded = S_task.expand(B_supp, -1, -1, -1)
            supp_dyn = active_nca(supp_x, S_expanded, steps=N_STEPS * 2)
            pixel_loss, _, _ = get_pixel_loss(supp_dyn, supp_y, supp_x)

            D_meta = meta_nca(supp_x, pixel_loss, dyn_state=D_meta, steps=N_STEPS)
            S_task = D_meta[:, :STATIC_CHN, :, :].mean(dim=0, keepdim=True)

            # Evaluate test set accuracy at repeat k
            k_accuracies[k] = test_acc(S_task)

    return k_accuracies


# ==========================================
# 4. MAIN BENCHMARK & SURFACE PLOT GENERATOR
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Omniglot-Trained MetaNCA across Shots & k Repeats (3D Surfaces)")
    parser.add_argument("--checkpoints", type=str, default=CHECKPOINT_DIR, help="Checkpoint directory.")
    parser.add_argument("--seeds", type=int, default=N_EVAL_SEEDS,
                        help="Number of random seeds per (shot, k) to average.")
    args = parser.parse_args()

    set_determinism(0)

    # 1. Resolve Checkpoints
    act_file, meta_file = OMNIGLOT_CHECKPOINTS
    act_path = os.path.join(args.checkpoints, act_file)
    meta_path = os.path.join(args.checkpoints, meta_file)

    if not os.path.exists(act_path) or not os.path.exists(meta_path):
        # Fallback to local root or alternative name
        fallback_act = "active_nca_latest10.pth"
        fallback_meta = "meta_nca_latest10.pth"
        if os.path.exists(fallback_act) and os.path.exists(fallback_meta):
            act_path, meta_path = fallback_act, fallback_meta
        else:
            raise FileNotFoundError(f"Could not find Omniglot checkpoints at '{act_path}'.")

    print("=" * 85)
    print(f" EVALUATING OMNIGLOT-TRAINED MODEL ACROSS SHOTS AND REPEATS (k=1..5)")
    print(f" Active Checkpoint : {act_path}")
    print(f" Meta Checkpoint   : {meta_path}")
    print(f" Device            : {DEVICE}")
    print("=" * 85)

    active_nca = ActiveNCA().to(DEVICE)
    meta_nca = MetaNCA().to(DEVICE)
    active_nca.load_state_dict(torch.load(act_path, map_location=DEVICE))
    meta_nca.load_state_dict(torch.load(meta_path, map_location=DEVICE))
    active_nca.eval()
    meta_nca.eval()

    # Pre-cache raw datasets
    print("Pre-caching datasets into memory...")
    for dname, _ in DATASET_SCHEDULES:
        get_cached_raw_dataset(dname)
    print("Datasets ready.")

    # Storage: surface_grids[dataset] = 2D numpy array of shape (len(K_VALS), len(shots))
    surface_grids = {}

    total_start = time.perf_counter()

    # 2. Compute Surface Grids
    for dname, shot_list in DATASET_SCHEDULES:
        print(f"\n>>> Benchmarking Dataset: {dname.upper()} | Shots: {shot_list} | Repeats: {K_VALS} <<<")

        # Grid shape: (num_k, num_shots) = (5, 7)
        z_grid = np.zeros((len(K_VALS), len(shot_list)))

        for s_idx, shot in enumerate(shot_list):
            seed_runs = []
            for seed in range(args.seeds):
                set_determinism(seed + 500)
                accs_at_k = evaluate_trajectory(active_nca, meta_nca, dname, k_shot=shot, max_k=K_VALS[-1])
                seed_runs.append([accs_at_k[k] for k in K_VALS])

            mean_k = np.mean(seed_runs, axis=0)  # Shape: (5,)
            z_grid[:, s_idx] = mean_k

            print(f"  [Shot = {shot:3d}] Accuracies k=1..5: " + ", ".join(
                [f"k{k}={mean_k[k - 1]:5.1f}%" for k in K_VALS]))

        surface_grids[dname] = z_grid

    elapsed_total = time.perf_counter() - total_start
    print(f"\nAll evaluations complete in {elapsed_total:.1f}s.")

    # ==========================================
    # 3. RENDER 2x3 TILED 3D SURFACE PLOT
    # ==========================================
    print("\nGenerating tiled 3D surface plot...")
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "MetaNCA Few-Shot Generalization Landscape: Accuracy vs. Support Shots & Optimization Repeats (k)\n"
        "(Evaluated using Omniglot-Trained Dual-NCA Model)",
        fontsize=16, fontweight="bold", y=0.98
    )

    colormap_options = {
        "omniglot": "viridis",
        "kmnist": "plasma",
        "mnist": "crest" if "crest" in plt.colormaps() else "viridis",
        "fashion": "magma",
        "usps": "cividis",
        "medmnist": "coolwarm"
    }

    for idx, (dname, shot_list) in enumerate(DATASET_SCHEDULES, 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        Z = surface_grids[dname]  # Shape: (len(K_VALS), len(shot_list))

        # Create 2D Meshgrids
        X, Y = np.meshgrid(shot_list, K_VALS)

        cmap_name = colormap_options.get(dname, "viridis")
        surf = ax.plot_surface(
            X, Y, Z,
            cmap=cmap_name,
            edgecolor='k',
            linewidth=0.4,
            alpha=0.88,
            antialiased=True
        )

        # Wireframe overlay for clear grid structure
        ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.3, alpha=0.3)

        # Domain title
        tag = "(In-Domain Training Source)" if dname == "omniglot" else "(Out-of-Distribution Transfer)"
        ax.set_title(f"{dname.upper()}\n{tag}", fontsize=12, fontweight="bold", pad=12)

        # Axis labeling
        ax.set_xlabel("Support Shots (n)", fontsize=10, fontweight="bold", labelpad=9)
        ax.set_ylabel("Repeats (k)", fontsize=10, fontweight="bold", labelpad=9)
        ax.set_zlabel("Accuracy (%)", fontsize=10, fontweight="bold", labelpad=9)

        # Tick settings
        ax.set_xticks(shot_list)
        ax.set_yticks(K_VALS)
        ax.set_zlim(0, 105)

        # Standardized viewing angle for optimal 3D perspective
        ax.view_init(elev=24, azim=-125)

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08)
        cbar.set_label("Accuracy (%)", fontsize=8)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.05, wspace=0.25, hspace=0.28)

    out_plot = "surface_plot_k_vs_shots_all_datasets.png"
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    print(f"Saved publication-quality 3D surface diagram to '{out_plot}'.")
    plt.show()


if __name__ == '__main__':
    main()