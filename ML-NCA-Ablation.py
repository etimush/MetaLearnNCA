import os
import random
import argparse
import numpy as np
from PIL import ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 0. HYPERPARAMETERS & CONFIG
# ==========================================
ACTIVE_DYN_CHN = 32  # Channels 0-9: Logits, 10-31: Hidden reasoning memory
META_DYN_CHN = 32  # Meta-NCA dynamic reasoning channels
STATIC_CHN = 24  # Spatial communication grid (S_task: Meta-NCA -> Active-NCA)
LOSS_CHN = 10  # Class prediction logits (10 classes)

N_STEPS = 19
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, hidden_size=256):
        c_in = 1 + STATIC_CHN + ACTIVE_DYN_CHN  # 1 + 24 + 32 = 57
        super().__init__(c_in, ACTIVE_DYN_CHN, hidden_size=hidden_size)

    def forward(self, img, static_channels, steps=N_STEPS, dyna=None, ablate_dynamic_channel=None):
        B, _, H, W = img.shape
        dyn = torch.rand(B, ACTIVE_DYN_CHN, H, W, device=img.device) if dyna is None else dyna.clone()

        def apply_dynamic_ablation(state):
            if ablate_dynamic_channel is not None:
                state = state.clone()
                state[:, ablate_dynamic_channel:ablate_dynamic_channel + 1, :, :] = torch.rand(
                    B, 1, H, W, device=state.device
                )
            return state

        dyn = apply_dynamic_ablation(dyn)

        for _ in range(steps):
            x = torch.cat([img, static_channels, dyn], dim=1)
            ds = self.grid_update(x)
            mask = (torch.rand(B, 1, H, W, device=img.device) > 0.5).float()
            dyn = dyn + ds * mask
            dyn = apply_dynamic_ablation(dyn)

        return dyn


class MetaNCA(BaseNCA):
    def __init__(self, hidden_size=256):
        c_in = LOSS_CHN + META_DYN_CHN  # 10 + 32 = 42
        super().__init__(c_in, META_DYN_CHN, hidden_size=hidden_size)

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
# 2. DATASET SPLITTING (SINGLE DATASET)
# ==========================================
def get_dataset_splits(dataset_name, n_way=10, k_shot=10):
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

    dname = dataset_name.lower()
    if dname == "fashion":
        dataset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=fashion_transform)
        targets = dataset.targets.numpy()
    elif dname == "omniglot":
        dataset = torchvision.datasets.Omniglot('./data', background=False, download=True, transform=omniglot_transform)
        targets = np.array([y for _, y in dataset])
    elif dname == "mnist":
        dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=standard_transform)
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
    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'.")

    available_classes = list(set(targets))
    np.random.shuffle(available_classes)
    chosen_classes = available_classes[:n_way]
    class_to_surrogate = {true_cls: surr for surr, true_cls in enumerate(chosen_classes)}

    support_idx, test_idx = [], []
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
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    return supp_loader, test_loader


# ==========================================
# 3. SPATIAL DAMAGE & ADDITIVE NOISE OPERATORS
# ==========================================
def apply_spatial_damage(grid, num_circles, radius=4):
    """Circular contiguous damage patches (replacement noise)."""
    if num_circles <= 0:
        return grid.clone()

    damaged = grid.clone()
    B, C, H, W = damaged.shape
    device = damaged.device

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    for _ in range(num_circles):
        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        circle_mask = (dist_sq <= radius ** 2)

        noise = torch.rand((B, C, H, W), device=device)
        mask_expanded = circle_mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
        damaged = torch.where(mask_expanded, noise, damaged)

    return damaged


def apply_random_spatial_noise(grid, noise_ratio, noise_scale=0.5, noise_type="gaussian"):
    """
    ADDS random noise directly onto the existing, well-formed program channels S_task
    at `noise_ratio` percentage of spatial coordinates (H x W).
    """
    if noise_ratio <= 0.0:
        return grid.clone()

    damaged = grid.clone()
    B, C, H, W = damaged.shape
    total_pixels = H * W
    num_corrupt = int(round(noise_ratio * total_pixels))

    perm = torch.randperm(total_pixels, device=damaged.device)[:num_corrupt]
    mask_flat = torch.zeros(total_pixels, dtype=torch.bool, device=damaged.device)
    mask_flat[perm] = True
    spatial_mask = mask_flat.view(1, 1, H, W).expand(B, C, -1, -1).float()

    if noise_type == "gaussian":
        noise = torch.randn_like(damaged) * noise_scale
    elif noise_type == "uniform":
        noise = (torch.rand_like(damaged) * 2.0 - 1.0) * noise_scale
    else:
        noise = torch.rand_like(damaged) * noise_scale

    damaged = damaged + spatial_mask * noise
    return damaged


# ==========================================
# 4. CORE ADAPTATION & EVALUATION
# ==========================================
def adapt_support_rules(active_nca, meta_nca, supp_x, supp_y, k_repeats=5):
    S_task = torch.rand(1, STATIC_CHN, supp_x.size(2), supp_x.size(3), device=DEVICE)
    D_meta = None

    for _ in range(k_repeats):
        with torch.no_grad():
            S_expanded = S_task.expand(supp_x.size(0), -1, -1, -1)
            supp_dyn = active_nca(supp_x, S_expanded, steps=N_STEPS * 2)
            pixel_loss, _, _ = get_pixel_loss(supp_dyn, supp_y, supp_x)

            D_meta = meta_nca(supp_x, pixel_loss, dyn_state=D_meta, steps=N_STEPS)
            S_task = D_meta[:, :STATIC_CHN, :, :].mean(dim=0, keepdim=True)

    return S_task


def compute_accuracy(active_nca, test_loader, S_task, ablate_dynamic_channel=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for qry_x, qry_y in test_loader:
            qry_x, qry_y = qry_x.to(DEVICE), qry_y.to(DEVICE)
            S_expanded = S_task.expand(qry_x.size(0), -1, -1, -1)
            qry_dyn = active_nca(qry_x, S_expanded, steps=N_STEPS * 2, ablate_dynamic_channel=ablate_dynamic_channel)

            _, qry_pred, alive_mask = get_pixel_loss(qry_dyn, qry_y, qry_x)
            logits = (qry_pred * alive_mask).sum(dim=(2, 3))
            preds = logits.argmax(dim=1)

            correct += (preds == qry_y).sum().item()
            total += qry_y.size(0)

    return (correct / total) * 100.0


# ==========================================
# 5. ABLATION EXPERIMENT RUNNERS
# ==========================================
def run_hidden_channel_ablation(active_nca, test_loader, S_task, dataset_name):
    print(f"\n[1/4] Running Hidden Dynamic Channel Ablation ({dataset_name.upper()})...")
    baseline_acc = compute_accuracy(active_nca, test_loader, S_task, ablate_dynamic_channel=None)
    print(f"  --> Baseline Accuracy (No Ablation): {baseline_acc:.2f}%")

    hidden_channels = list(range(LOSS_CHN, ACTIVE_DYN_CHN))  # Channels 10 to 31
    ablation_accuracies = []

    for ch in tqdm(hidden_channels, desc="Ablating Hidden Reasoning Channels"):
        acc = compute_accuracy(active_nca, test_loader, S_task, ablate_dynamic_channel=ch)
        ablation_accuracies.append(acc)

    plt.figure(figsize=(10, 5))
    plt.bar(hidden_channels, ablation_accuracies, color='#2b5c8f', edgecolor='black', alpha=0.85)
    plt.axhline(baseline_acc, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.1f}%)')
    plt.title(f"{dataset_name.upper()}: Hidden Reasoning Channel Ablation (10–31)", fontsize=13, fontweight='bold')
    plt.xlabel("Ablated Dynamic Channel Index", fontsize=11)
    plt.ylabel("Global Test Accuracy (%)", fontsize=11)
    plt.xticks(hidden_channels)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    fname = f"ablation_hidden_channels_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  --> Saved: '{fname}'")

    return baseline_acc, hidden_channels, ablation_accuracies


def run_program_channel_ablation(active_nca, test_loader, S_task, dataset_name):
    print(f"\n[2/4] Running Spatial Program Channel Ablation ({dataset_name.upper()})...")
    baseline_acc = compute_accuracy(active_nca, test_loader, S_task, ablate_dynamic_channel=None)

    program_channels = list(range(STATIC_CHN))  # Channels 0 to 23
    ablation_accuracies = []

    for ch in tqdm(program_channels, desc="Ablating Program Channels"):
        corrupted_S = S_task.clone()
        corrupted_S[:, ch:ch + 1, :, :] = torch.rand_like(corrupted_S[:, ch:ch + 1, :, :])
        acc = compute_accuracy(active_nca, test_loader, corrupted_S, ablate_dynamic_channel=None)
        ablation_accuracies.append(acc)

    plt.figure(figsize=(11, 5))
    plt.bar(program_channels, ablation_accuracies, color='#4575b4', edgecolor='black', alpha=0.85)
    plt.axhline(baseline_acc, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.1f}%)')
    plt.title(f"{dataset_name.upper()}: Spatial Program Channel Ablation (S_task 0–23)", fontsize=13, fontweight='bold')
    plt.xlabel("Ablated Program Channel Index", fontsize=11)
    plt.ylabel("Global Test Accuracy (%)", fontsize=11)
    plt.xticks(program_channels)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    fname = f"ablation_program_channels_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  --> Saved: '{fname}'")

    return baseline_acc, program_channels, ablation_accuracies


def run_spatial_damage_study(active_nca, test_loader, S_task, dataset_name, max_circles=6, trials=4, radius=4):
    print(f"\n[3/4] Running Localized Circular Damage Study ({dataset_name.upper()})...")
    circle_counts = list(range(max_circles + 1))
    mean_accuracies = []
    std_accuracies = []

    for num_c in circle_counts:
        trial_accs = []
        for _ in range(trials if num_c > 0 else 1):
            damaged_S = apply_spatial_damage(S_task, num_circles=num_c, radius=radius)
            acc = compute_accuracy(active_nca, test_loader, damaged_S, ablate_dynamic_channel=None)
            trial_accs.append(acc)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs) if num_c > 0 else 0.0
        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        circle_counts, mean_accuracies, yerr=std_accuracies,
        marker='o', capsize=5, capthick=1.5, linewidth=2, markersize=6,
        color='#d95f02', ecolor='gray', label=f'{dataset_name.upper()} Resilience'
    )
    plt.title(f"{dataset_name.upper()}: Circular Spatial Damage Resilience", fontsize=13, fontweight='bold')
    plt.xlabel(f"Number of Circular Patches (Radius = {radius} px, All Channels)", fontsize=11)
    plt.ylabel("Global Test Accuracy (%)", fontsize=11)
    plt.xticks(circle_counts)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=11)
    plt.tight_layout()

    fname = f"spatial_damage_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  --> Saved: '{fname}'")

    return circle_counts, mean_accuracies, std_accuracies


def run_random_spatial_noise_study(active_nca, test_loader, S_task, dataset_name, trials=4, noise_scale=0.5,
                                   noise_type="gaussian"):
    """
    Evaluates accuracy when ADDING random noise to 0% -> 100% of spatial coordinates
    across all channels of S_task in 10% increments.
    """
    print(
        f"\n[4/4] Running Additive Spatial Noise Study (0% -> 100% in 10% steps, scale={noise_scale}) on {dataset_name.upper()}...")
    percentages = list(range(0, 101, 10))
    mean_accuracies = []
    std_accuracies = []

    baseline_acc = compute_accuracy(active_nca, test_loader, S_task, ablate_dynamic_channel=None)

    for p in tqdm(percentages, desc="Testing Additive Noise Levels"):
        ratio = p / 100.0
        trial_accs = []
        for _ in range(trials if p > 0 else 1):
            noisy_S = apply_random_spatial_noise(
                S_task,
                noise_ratio=ratio,
                noise_scale=noise_scale,
                noise_type=noise_type
            )
            acc = compute_accuracy(active_nca, test_loader, noisy_S, ablate_dynamic_channel=None)
            trial_accs.append(acc)

        mean_acc = np.mean(trial_accs)
        std_acc = np.std(trial_accs) if p > 0 else 0.0
        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)

    plt.figure(figsize=(8.5, 5))
    plt.errorbar(
        percentages, mean_accuracies, yerr=std_accuracies,
        marker='s', capsize=5, capthick=1.5, linewidth=2, markersize=6,
        color='#7570b3', ecolor='gray',
        label=f'{dataset_name.upper()} Additive Noise ({noise_type.capitalize()}, $\\sigma={noise_scale}$)'
    )
    plt.axhline(baseline_acc, color='red', linestyle='--', linewidth=1.8, label=f'Clean Baseline ({baseline_acc:.1f}%)')
    plt.title(f"{dataset_name.upper()}: Program Resilience to Additive Spatial Noise", fontsize=13, fontweight='bold')
    plt.xlabel("Percentage of Grid Locations Receiving Additive Noise (%)", fontsize=11)
    plt.ylabel("Global Test Accuracy (%)", fontsize=11)
    plt.xticks(percentages)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()

    fname = f"spatial_noise_additive_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  --> Saved: '{fname}'")

    return percentages, mean_accuracies, std_accuracies


# ==========================================
# 6. VISUALIZATION OPERATORS
# ==========================================
def visualize_all_hidden_channels(active_nca, test_loader, S_task, dataset_name):
    """Full 4x6 grid of all 22 hidden reasoning channels (10-31) + input and prediction."""
    qry_x, qry_y = next(iter(test_loader))
    qry_x = qry_x[:1].to(DEVICE)
    qry_y = qry_y[:1].to(DEVICE)

    with torch.no_grad():
        S_expanded = S_task.expand(1, -1, -1, -1)
        qry_dyn = active_nca(qry_x, S_expanded, steps=N_STEPS * 2)

    hidden_maps = qry_dyn[0, LOSS_CHN:ACTIVE_DYN_CHN].cpu()
    num_hidden = hidden_maps.size(0)

    fig, axes = plt.subplots(4, 6, figsize=(14, 9))
    fig.suptitle(f"{dataset_name.upper()}: Complete Active-NCA Hidden Reasoning Channels (10–31)", fontsize=14,
                 fontweight='bold')
    axes = axes.flatten()

    axes[0].imshow(qry_x[0, 0].cpu(), cmap='gray')
    axes[0].set_title("Input Image", fontsize=10, fontweight='bold', color='navy')
    axes[0].axis('off')

    pred_map = qry_dyn[0, :LOSS_CHN].argmax(dim=0).cpu() * (qry_x[0, 0].cpu() > -0.8)
    axes[1].imshow(pred_map, cmap='jet')
    axes[1].set_title(f"Target: {qry_y[0].item()} | Pred", fontsize=10, fontweight='bold', color='darkgreen')
    axes[1].axis('off')

    for idx in range(num_hidden):
        ch_num = LOSS_CHN + idx
        ax = axes[idx + 2]
        ax.imshow(hidden_maps[idx], cmap='viridis')
        ax.set_title(f"Hidden Ch {ch_num}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    fname = f"hidden_channels_full_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  --> Saved Full Hidden Channels Grid: '{fname}'")


def visualize_spatial_program_channels(S_task, dataset_name):
    """Full 4x6 grid of all 24 spatial program channels (0-23)."""
    fig, axes = plt.subplots(4, 6, figsize=(14, 9))
    fig.suptitle(f"{dataset_name.upper()}: Complete Spatial Program Field S_task (All 24 Channels)", fontsize=14,
                 fontweight='bold')
    axes = axes.flatten()

    program_maps = S_task[0].cpu()
    for ch in range(STATIC_CHN):
        ax = axes[ch]
        ax.imshow(program_maps[ch], cmap='magma')
        ax.set_title(f"Program Ch {ch}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    fname = f"program_channels_full_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  --> Saved Full Program Channels Grid: '{fname}'")


def visualize_compressed_1x8(active_nca, test_loader, S_task, dataset_name,
                             hidden_indices=(10, 11, 12, 13),
                             program_indices=(5, 13, 21, 23)):
    """Generates a publication-ready 1x8 strip figure."""
    qry_x, _ = next(iter(test_loader))
    qry_x = qry_x[:1].to(DEVICE)

    with torch.no_grad():
        S_expanded = S_task.expand(1, -1, -1, -1)
        qry_dyn = active_nca(qry_x, S_expanded, steps=N_STEPS * 2)

    fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
    fig.suptitle(
        f"{dataset_name.upper()}: Compressed Feature Representation "
        f"[Left: 4 Active Hidden Channels | Right: 4 Spatial Program Channels]",
        fontsize=12, fontweight='bold', y=1.05
    )

    for i, h_ch in enumerate(hidden_indices):
        ax = axes[i]
        feature = qry_dyn[0, h_ch].cpu()
        ax.imshow(feature, cmap='viridis')
        ax.set_title(f"Hidden Ch {h_ch}", fontsize=10, color='navy', fontweight='bold')
        ax.axis('off')

    for j, p_ch in enumerate(program_indices):
        ax = axes[4 + j]
        feature = S_task[0, p_ch].cpu()
        ax.imshow(feature, cmap='magma')
        ax.set_title(f"Program Ch {p_ch}", fontsize=10, color='darkred', fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    fname = f"compressed_1x8_channels_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  --> Saved Compressed 1x8 Strip: '{fname}'")


def visualize_program_difference_heatmap(active_nca, meta_nca, dataset_name, k_shot, k_repeats=5, seed1=42, seed2=43):
    """
    Adapts S_task across two different seeds, computes signed spatial difference
    Delta S = S_task(seed1) - S_task(seed2) across all 24 channels, and plots a 4x6 diverging heatmap.
    """
    print(f"\n--- Comparing Program Representations: Seed {seed1} vs. Seed {seed2} ({dataset_name.upper()}) ---")

    # Run 1: Seed 1
    set_determinism(seed1)
    supp_loader_1, test_loader_1 = get_dataset_splits(dataset_name, n_way=10, k_shot=k_shot)
    supp_x1, supp_y1 = next(iter(supp_loader_1))
    supp_x1, supp_y1 = supp_x1.to(DEVICE), supp_y1.to(DEVICE)
    S_task_1 = adapt_support_rules(active_nca, meta_nca, supp_x1, supp_y1, k_repeats=k_repeats)
    acc_1 = compute_accuracy(active_nca, test_loader_1, S_task_1)

    # Run 2: Seed 2
    set_determinism(seed2)
    supp_loader_2, test_loader_2 = get_dataset_splits(dataset_name, n_way=10, k_shot=k_shot)
    supp_x2, supp_y2 = next(iter(supp_loader_2))
    supp_x2, supp_y2 = supp_x2.to(DEVICE), supp_y2.to(DEVICE)
    S_task_2 = adapt_support_rules(active_nca, meta_nca, supp_x2, supp_y2, k_repeats=k_repeats)
    acc_2 = compute_accuracy(active_nca, test_loader_2, S_task_2)

    # Reset determinism back to seed1
    set_determinism(seed1)

    diff = (S_task_1[0] - S_task_2[0]).cpu()
    abs_diff = torch.abs(diff)
    global_mad = abs_diff.mean().item()

    vmax = abs_diff.max().item()
    if vmax == 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(4, 6, figsize=(15, 9.5))
    fig.suptitle(
        f"{dataset_name.upper()}: Program Channel Divergence $\\Delta S = S_{{task}}^{{(Seed\\ {seed1})}} - S_{{task}}^{{(Seed\\ {seed2})}}$\n"
        f"Accuracies: Seed {seed1} = {acc_1:.2f}% | Seed {seed2} = {acc_2:.2f}% | Global Mean Absolute Difference (MAD): {global_mad:.4f}",
        fontsize=13, fontweight='bold', y=0.98
    )
    axes = axes.flatten()

    im = None
    for ch in range(STATIC_CHN):
        ax = axes[ch]
        ch_diff = diff[ch].numpy()
        ch_mad = abs_diff[ch].mean().item()
        im = ax.imshow(ch_diff, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax.set_title(f"Ch {ch} (MAD: {ch_mad:.3f})", fontsize=9, fontweight='bold')
        ax.axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Signed Difference ($S_{1} - S_{2}$)", fontsize=10, fontweight='bold')

    plt.subplots_adjust(left=0.03, right=0.90, top=0.89, bottom=0.03, wspace=0.25, hspace=0.35)
    fname = f"program_channels_diff_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f"  --> Saved Program Difference Heatmap: '{fname}'")
    print(f"      • Seed {seed1} Test Acc : {acc_1:.2f}%")
    print(f"      • Seed {seed2} Test Acc : {acc_2:.2f}%")
    print(f"      • Global Mean Abs Diff  : {global_mad:.4f}")

    return S_task_1, S_task_2, acc_1, acc_2


def visualize_program_noise_comparison(S_task, dataset_name, noise_scale=0.5, noise_type="gaussian",
                                       representative_channels=(0, 3, 6, 9, 12, 15, 18, 21)):
    """
    Renders a 2x8 comparison grid:
      - Top row: 8 representative program channels with NO noise (0% noise)
      - Bottom row: The same 8 channels with 100% additive noise
    Color scales are column-locked to show contrast and signal retention directly.
    """
    print(f"\n--- Generating 2x8 Clean vs. 100% Additive Noise Comparison ({dataset_name.upper()}) ---")

    # 1. Generate 100% additive noise program
    S_noisy_100 = apply_random_spatial_noise(
        S_task,
        noise_ratio=1.0,
        noise_scale=noise_scale,
        noise_type=noise_type
    )

    clean_grid = S_task[0].cpu().numpy()
    noisy_grid = S_noisy_100[0].cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
    fig.suptitle(
        f"{dataset_name.upper()}: Spatial Program Resilience Under 100% Additive Noise\n"
        f"[Top Row: Clean $S_{{task}}$ (0% Noise) | Bottom Row: Perturbed $S_{{task}}$ (100% Additive Noise, $\\sigma={noise_scale}$)]",
        fontsize=12, fontweight='bold', y=0.98
    )

    for col, ch in enumerate(representative_channels):
        clean_ch = clean_grid[ch]
        noisy_ch = noisy_grid[ch]

        # Column-matched min/max bounds so intensity is directly comparable
        vmin = min(clean_ch.min(), noisy_ch.min())
        vmax = max(clean_ch.max(), noisy_ch.max())

        # Top row: Clean channel
        ax_top = axes[0, col]
        ax_top.imshow(clean_ch, cmap='magma', vmin=vmin, vmax=vmax)
        ax_top.set_title(f"Clean Ch {ch}", fontsize=10, color='navy', fontweight='bold')
        ax_top.axis('off')

        # Bottom row: 100% additive noise channel
        ax_bot = axes[1, col]
        ax_bot.imshow(noisy_ch, cmap='magma', vmin=vmin, vmax=vmax)
        ax_bot.set_title(f"Noise Ch {ch}", fontsize=10, color='darkred', fontweight='bold')
        ax_bot.axis('off')

    plt.tight_layout()
    fname = f"program_noise_visual_comparison_{dataset_name.lower()}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  --> Saved 2x8 Noise Comparison Figure: '{fname}'")


# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="MetaNCA Single-Dataset Ablation, Difference Heatmap & Visualization Suite")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["omniglot", "fashion", "mnist", "kmnist", "usps", "medmnist"],
                        help="The single dataset to evaluate and ablate.")
    parser.add_argument("--active_ckpt", type=str, default=None,
                        help="Explicit path to active_nca checkpoint (.pth).")
    parser.add_argument("--meta_ckpt", type=str, default=None,
                        help="Explicit path to meta_nca checkpoint (.pth).")
    parser.add_argument("--seed", type=str, default=None,
                        help="Shortcut seed identifier (e.g. --seed 0 looks in checkpoints/ for matching pair).")
    parser.add_argument("--seed1", type=int, default=42,
                        help="First evaluation seed for representation difference analysis (default: 42).")
    parser.add_argument("--seed2", type=int, default=43,
                        help="Second evaluation seed for representation difference analysis (default: 43).")
    parser.add_argument("--k_shot", type=int, default=None,
                        help="Support shot count. Defaults: 10 for omniglot/usps; 100 for others.")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="MLP hidden dimension (default: 256).")
    parser.add_argument("--k_repeats", type=int, default=5,
                        help="Number of support optimization repeats (default: 5).")
    parser.add_argument("--noise_scale", type=float, default=0.5,
                        help="Additive noise scale / standard deviation (default: 0.5).")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "uniform"],
                        help="Additive noise distribution type (default: gaussian).")

    args = parser.parse_args()
    set_determinism(args.seed1)

    # 1. Resolve Checkpoint Paths
    active_path = args.active_ckpt
    meta_path = args.meta_ckpt

    if args.seed is not None:
        active_path = f"checkpoints/active_nca_seed{args.seed}.pth"
        meta_path = f"checkpoints/meta_nca_seed{args.seed}.pth"
    elif active_path is None or meta_path is None:
        default_act = "active_nca_latest0.pth"
        default_meta = "meta_nca_latest0.pth"
        if os.path.exists(default_act) and os.path.exists(default_meta):
            active_path, meta_path = default_act, default_meta
        else:
            raise FileNotFoundError(
                "Please specify model checkpoints via '--active_ckpt' & '--meta_ckpt', or via '--seed <id>'."
            )

    print(f"\n=======================================================")
    print(f" TARGET DATASET : {args.dataset.upper()}")
    print(f" ACTIVE CKPT    : {active_path}")
    print(f" META CKPT      : {meta_path}")
    print(f" DIFF SEEDS     : Seed {args.seed1} vs. Seed {args.seed2}")
    print(f" DEVICE         : {DEVICE}")
    print(f"=======================================================")

    # 2. Determine k_shot
    k_shot = args.k_shot
    if k_shot is None:
        k_shot = 10 if args.dataset in ["omniglot", "usps"] else 100
    print(f"Episode Setup: 10-Way, {k_shot}-Shot Optimization")

    # 3. Instantiate and Load Models
    active_nca = ActiveNCA(hidden_size=args.hidden_dim).to(DEVICE)
    meta_nca = MetaNCA(hidden_size=args.hidden_dim).to(DEVICE)

    try:
        active_nca.load_state_dict(torch.load(active_path, map_location=DEVICE))
        meta_nca.load_state_dict(torch.load(meta_path, map_location=DEVICE))
        print("Successfully loaded pre-trained Dual-NCA weights.")
    except Exception as e:
        print(f"Error loading checkpoint weights: {e}")
        return

    active_nca.eval()
    meta_nca.eval()

    # 4. Data Loading for Baseline Visualizations & Ablations (Seed 1)
    supp_loader, test_loader = get_dataset_splits(args.dataset, n_way=10, k_shot=k_shot)
    supp_x, supp_y = next(iter(supp_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)

    # 5. Adapt S_task on Support Set
    print(f"\nAdapting spatial program S_task over {args.k_repeats} interaction repeats...")
    S_task = adapt_support_rules(active_nca, meta_nca, supp_x, supp_y, k_repeats=args.k_repeats)

    # 6. Generate Standard Visualizations
    print("\n--- Generating Spatial Visualizations ---")
    visualize_all_hidden_channels(active_nca, test_loader, S_task, args.dataset)
    visualize_spatial_program_channels(S_task, args.dataset)
    visualize_compressed_1x8(active_nca, test_loader, S_task, args.dataset)

    # 7. Generate Program Difference Heatmap (Seed 1 vs. Seed 2)
    visualize_program_difference_heatmap(
        active_nca, meta_nca, args.dataset, k_shot=k_shot,
        k_repeats=args.k_repeats, seed1=args.seed1, seed2=args.seed2
    )

    # 8. Visual 2x8 Clean vs. 100% Additive Noise Comparison
    visualize_program_noise_comparison(
        S_task, args.dataset,
        noise_scale=args.noise_scale, noise_type=args.noise_type,
        representative_channels=(0, 3, 6, 9, 12, 15, 18, 21)
    )

    # 9. Run Full Ablation Suite
    print("\n--- Running Ablation Suite ---")
    run_hidden_channel_ablation(active_nca, test_loader, S_task, args.dataset)
    run_program_channel_ablation(active_nca, test_loader, S_task, args.dataset)
    run_spatial_damage_study(active_nca, test_loader, S_task, args.dataset, max_circles=6, trials=4, radius=4)
    run_random_spatial_noise_study(
        active_nca, test_loader, S_task, args.dataset,
        trials=4, noise_scale=args.noise_scale, noise_type=args.noise_type
    )

    print(f"\nAll tasks finished for {args.dataset.upper()}! Generated files:")
    print(f"  1. hidden_channels_full_{args.dataset.lower()}.png")
    print(f"  2. program_channels_full_{args.dataset.lower()}.png")
    print(f"  3. compressed_1x8_channels_{args.dataset.lower()}.png")
    print(f"  4. program_channels_diff_{args.dataset.lower()}.png")
    print(f"  5. program_noise_visual_comparison_{args.dataset.lower()}.png  <-- [2x8 Clean vs 100% Noise Comparison]")
    print(f"  6. ablation_hidden_channels_{args.dataset.lower()}.png")
    print(f"  7. ablation_program_channels_{args.dataset.lower()}.png")
    print(f"  8. spatial_damage_{args.dataset.lower()}.png")
    print(f"  9. spatial_noise_additive_{args.dataset.lower()}.png")


if __name__ == '__main__':
    main()