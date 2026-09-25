import os
import re
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

# Optional Scipy for exact t-distribution critical values
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. HYPERPARAMETERS & CONFIG
# ==========================================
ACTIVE_DYN_CHN = 32  # Dynamic reasoning channels for Active-NCA
META_DYN_CHN = 32    # Dynamic reasoning channels for Meta-NCA
STATIC_CHN = 24      # Spatial communication grid (S_task)
LOSS_CHN = 10        # Class prediction logits
HIDDEN_DIM = 256     # Hidden MLP channel dimension

K_REPEATS = 5        # Evaluation repeats (k = 0, 1, 2, 3, 4, 5)
N_STEPS = 19         # Base NCA steps per interaction
SHIFTS_TO_PROG = 4   #Test time spatial shift to the program after generation

CHECKPOINT_DIR = "checkpoints"
RUN_SINGLE_SEED = 55         # e.g., 0, "0", or None
RUN_EVAL_LOOP = True    # <-- Set to True to run NUM_EVAL_LOOPS test runs on ONE training seed
NUM_EVAL_LOOPS = 5     # Number of test evaluation runs (independent support/test splits)

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
        base_filters = torch.stack([ident, sobel_x, sobel_y, lap])  # (4, 3, 3)

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

    def forward(self, img, static_channels, steps=N_STEPS, dyna=None):
        B, _, H, W = img.shape
        dyn = torch.rand(B, ACTIVE_DYN_CHN, H, W, device=img.device) if dyna is None else dyna

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
    B, _, H, W = images.shape
    alive_mask = (images > -0.8).float()
    pred_logits = pred_dyn[:, :LOSS_CHN, :, :]
    probs = F.softmax(pred_logits, dim=1)

    spatial_targets = torch.zeros_like(pred_logits)
    for b in range(B):
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
# 3. DATA SPLITTING & BENCHMARK SUITE
# ==========================================
def get_global_splits(dataset_name, n_way=10, k_shot=1, eval_seed=None):
    if eval_seed is not None:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)

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
        try:
            import medmnist
        except ImportError:
            raise ImportError("Please install medmnist: 'pip install medmnist'")
        info = medmnist.INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', download=True, transform=fashion_transform)
        targets = dataset.labels.flatten()
    elif dataset_name == "omniglot":
        dataset = torchvision.datasets.Omniglot('./data', background=False, download=True, transform=omniglot_transform)
        targets = np.array([y for _, y in dataset])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    return supp_loader, test_loader


# ==========================================
# 4. SINGLE MODEL EVALUATION PIPELINE
# ==========================================
def evaluate_global_oneshot(active_nca, meta_nca, dataset_name, n_way=10, k_shot=1, max_k_repeats=K_REPEATS, eval_seed=None):
    supp_loader, test_loader = get_global_splits(dataset_name, n_way, k_shot, eval_seed=eval_seed)

    supp_x, supp_y = next(iter(supp_loader))
    supp_x, supp_y = supp_x.to(DEVICE), supp_y.to(DEVICE)
    B_supp = supp_x.size(0)

    S_task = torch.rand(1, STATIC_CHN, supp_x.size(2), supp_x.size(3), device=DEVICE)
    D_meta = None

    accuracies = []
    total_flops = 0.0

    flops_active_step_supp = calculate_step_flops(1 + STATIC_CHN + ACTIVE_DYN_CHN, ACTIVE_DYN_CHN, HIDDEN_DIM, B_supp)
    flops_meta_step_supp = calculate_step_flops(LOSS_CHN + META_DYN_CHN, META_DYN_CHN, HIDDEN_DIM, B_supp)

    def evaluate_test_set(S_current):
        nonlocal total_flops
        correct = 0
        total = 0
        S_current = torch.roll(S_current, (SHIFTS_TO_PROG, SHIFTS_TO_PROG), (2,3))
        with torch.no_grad():
            for qry_x, qry_y in test_loader:
                qry_x, qry_y = qry_x.to(DEVICE), qry_y.to(DEVICE)
                B_qry = qry_x.size(0)

                flops_active_step_qry = calculate_step_flops(1 + STATIC_CHN + ACTIVE_DYN_CHN, ACTIVE_DYN_CHN, HIDDEN_DIM, B_qry)
                total_flops += (N_STEPS * 2) * flops_active_step_qry

                S_expanded_qry = S_current.expand(B_qry, -1, -1, -1)
                qry_dyn = active_nca(qry_x, S_expanded_qry, steps=N_STEPS * 2)

                _, qry_pred, alive_mask = get_pixel_loss(qry_dyn, qry_y, qry_x)
                logits = (qry_pred * alive_mask).sum(dim=(2, 3))
                preds = logits.argmax(dim=1)

                correct += (preds == qry_y).sum().item()
                total += qry_y.size(0)

        return (correct / total) * 100.0

    start_time = time.perf_counter()

    acc_0 = evaluate_test_set(S_task)
    accuracies.append(acc_0)

    for k in range(1, max_k_repeats + 1):
        with torch.no_grad():
            total_flops += (N_STEPS * 2) * flops_active_step_supp
            S_expanded = S_task.expand(B_supp, -1, -1, -1)
            supp_dyn = active_nca(supp_x, S_expanded, steps=N_STEPS * 2)
            pixel_loss, _, _ = get_pixel_loss(supp_dyn, supp_y, supp_x)

            total_flops += N_STEPS * flops_meta_step_supp
            D_meta = meta_nca(supp_x, pixel_loss, dyn_state=D_meta, steps=N_STEPS)
            S_update = D_meta[:, :STATIC_CHN, :, :].mean(dim=0, keepdim=True)
            S_task = S_update

        acc_k = evaluate_test_set(S_task)
        accuracies.append(acc_k)

    elapsed_wall_time = time.perf_counter() - start_time
    return accuracies, elapsed_wall_time, total_flops


# ==========================================
# 5. CHECKPOINT SCANNER & DISCOVERY
# ==========================================
def find_checkpoints(checkpoint_dir, target_seed=None):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

    files = os.listdir(checkpoint_dir)
    pattern = re.compile(r"active_nca_seed([a-zA-Z0-9_\-]+)\.pth", re.IGNORECASE)

    pairs = {}
    for f in files:
        match = pattern.match(f)
        if match:
            seed_str = match.group(1)
            meta_pattern = re.compile(rf"meta_nca_seed{seed_str}\.pth", re.IGNORECASE)
            matching_meta = [mf for mf in files if meta_pattern.match(mf)]

            if matching_meta:
                pairs[seed_str] = (
                    os.path.join(checkpoint_dir, f),
                    os.path.join(checkpoint_dir, matching_meta[0])
                )

    if not pairs:
        raise FileNotFoundError(f"No matching active/meta checkpoint pairs found in '{checkpoint_dir}'.")

    if target_seed is not None:
        target_str = str(target_seed)
        if target_str in pairs:
            return {target_str: pairs[target_str]}
        else:
            raise KeyError(f"Specified seed '{target_str}' not found. Available seeds: {list(pairs.keys())}")

    return dict(sorted(pairs.items(), key=lambda x: (len(x[0]), x[0])))


# ==========================================
# 6. STATISTICAL UTILITY (95% CONFIDENCE INTERVAL)
# ==========================================
def compute_mean_and_95ci(data_matrix):
    data = np.array(data_matrix)
    n = data.shape[0]
    mean = np.mean(data, axis=0)

    if n < 2:
        ci = np.zeros_like(mean)
        return mean, ci

    std_err = np.std(data, axis=0, ddof=1) / np.sqrt(n)

    if HAS_SCIPY:
        t_crit = stats.t.ppf(0.975, df=n - 1)
    else:
        t_table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042
        }
        df = n - 1
        t_crit = t_table.get(df, 1.96 if df > 30 else 2.0)

    ci = std_err * t_crit
    return mean, ci


# ==========================================
# 7. MAIN EXECUTION & BENCHMARK RUNNER
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaNCA checkpoints with 95% CI & Compute Benchmarks.")
    parser.add_argument("--checkpoints", type=str, default=CHECKPOINT_DIR, help="Path to checkpoint folder.")
    parser.add_argument("--seed", type=str, default=str(RUN_SINGLE_SEED) if RUN_SINGLE_SEED is not None else None,
                        help="Specify single seed to evaluate, or leave unset for all.")
    parser.add_argument("--run_eval_loop", action="store_true", default=RUN_EVAL_LOOP,
                        help="Run multiple independent test splits on a single checkpoint.")
    parser.add_argument("--eval_loops", type=int, default=NUM_EVAL_LOOPS,
                        help="Number of test evaluation loops per dataset.")
    parser.add_argument("--k_shot", type=int, default=5,
                        help="Support shot count to evaluate (default: 1).")
    args = parser.parse_args()

    # 1. Discover Checkpoints
    try:
        model_pairs = find_checkpoints(args.checkpoints, target_seed=args.seed)
    except Exception as e:
        print(f"Error locating checkpoints: {e}")
        return

    dataset_configs = [
        ("omniglot", args.k_shot),
        ("mnist", args.k_shot),
        ("fashion", args.k_shot),
        ("kmnist", args.k_shot),
        ("usps", args.k_shot),
        ("medmnist", args.k_shot),
    ]

    active_nca = ActiveNCA().to(DEVICE)
    meta_nca = MetaNCA().to(DEVICE)
    active_nca.eval()
    meta_nca.eval()

    all_params = list(active_nca.parameters()) + list(meta_nca.parameters())


    pytorch_total_params = sum(p.numel() for p in all_params)
    print(f"Number of parameters: {pytorch_total_params}")

    raw_results = {d_name: {} for d_name, _ in dataset_configs}
    wall_times = {d_name: [] for d_name, _ in dataset_configs}
    dataset_flops = {d_name: [] for d_name, _ in dataset_configs}

    # =========================================================================
    # MODE A: EVALUATION LOOP OVER A SINGLE CHECKPOINT (Apples-to-Apples with Baselines)
    # =========================================================================
    if args.run_eval_loop:
        target_seed, (act_path, meta_path) = list(model_pairs.items())[0]
        print(f"\n==========================================================================")
        print(f" MODE: EVALUATION LOOP ON SINGLE CHECKPOINT (SEED {target_seed})")
        print(f" Running {args.eval_loops} independent test splits per dataset ({args.k_shot}-shot)")
        print(f"==========================================================================")

        active_nca.load_state_dict(torch.load(act_path, map_location=DEVICE))
        meta_nca.load_state_dict(torch.load(meta_path, map_location=DEVICE))

        for loop_idx in range(1, args.eval_loops + 1):
            eval_seed = 42 + loop_idx
            set_determinism(eval_seed)
            print(f"\n--- [Evaluation Run {loop_idx:2d}/{args.eval_loops}] (Split Seed: {eval_seed}) ---")

            for d_name, k_shot in dataset_configs:
                accs, elapsed_time, flops = evaluate_global_oneshot(
                    active_nca, meta_nca, d_name, n_way=10, k_shot=k_shot,
                    max_k_repeats=K_REPEATS, eval_seed=eval_seed
                )
                raw_results[d_name][f"run_{loop_idx}"] = accs
                wall_times[d_name].append(elapsed_time)
                dataset_flops[d_name].append(flops)

                print(f"  [{d_name.upper():<9}] k=0: {accs[0]:5.2f}% -> k={K_REPEATS}: {accs[-1]:5.2f}% | Time: {elapsed_time:5.2f}s | GFLOPs: {flops / 1e9:6.1f}")

        num_samples = args.eval_loops
        sample_label = f"{args.eval_loops} Test Runs on Seed {target_seed}"

    # =========================================================================
    # MODE B: CHECKPOINT SCAN ACROSS INDEPENDENT TRAINING SEEDS
    # =========================================================================
    else:
        total_seeds = len(model_pairs)
        print(f"\nDiscovered {total_seeds} model checkpoint pair(s) in '{args.checkpoints}':")
        for s, (act_p, meta_p) in model_pairs.items():
            print(f"  • Seed {s:>5}: {os.path.basename(act_p)} & {os.path.basename(meta_p)}")

        for model_idx, (seed_name, (act_path, meta_path)) in enumerate(model_pairs.items(), 1):
            print(f"\n=======================================================")
            print(f" EVALUATING CHECKPOINT [{model_idx}/{total_seeds}]: SEED {seed_name}")
            print(f"=======================================================")
            set_determinism(model_idx + 50)
            active_nca.load_state_dict(torch.load(act_path, map_location=DEVICE))
            meta_nca.load_state_dict(torch.load(meta_path, map_location=DEVICE))

            for d_name, k_shot in dataset_configs:
                accs, elapsed_time, flops = evaluate_global_oneshot(
                    active_nca, meta_nca, d_name, n_way=10, k_shot=k_shot, max_k_repeats=K_REPEATS
                )
                raw_results[d_name][seed_name] = accs
                wall_times[d_name].append(elapsed_time)
                dataset_flops[d_name].append(flops)

                print(f"  [{d_name.upper():<9}] k=0: {accs[0]:5.2f}% -> k={K_REPEATS}: {accs[-1]:5.2f}% | Time: {elapsed_time:5.2f}s | GFLOPs: {flops / 1e9:6.1f}")

        num_samples = total_seeds
        sample_label = f"{total_seeds} Training Seeds"

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" FINAL BENCHMARK SUMMARY TABLE ({sample_label})")
    print("=" * 95)
    print(f"{'DATASET':<12} | {'SHOT':<5} | {'FINAL MEAN ACC ± 95% CI':<25} | {'AVG TIME (s)':<14} | {'BEST RUN ACC':<12}")
    print("-" * 95)

    dataset_summary = {}
    for d_name, k_shot in dataset_configs:
        runs = list(raw_results[d_name].values())
        mean_curve, ci_curve = compute_mean_and_95ci(runs)
        avg_time = np.mean(wall_times[d_name])
        avg_flops = np.mean(dataset_flops[d_name]) / 1e9
        best_acc = np.max([r[-1] for r in runs])

        dataset_summary[d_name] = {
            'mean_curve': mean_curve,
            'ci_curve': ci_curve,
            'avg_time': avg_time,
            'avg_flops': avg_flops
        }

        ci_str = f"± {ci_curve[-1]:.2f}%" if num_samples > 1 else "(single run)"
        print(f"{d_name.upper():<12} | {k_shot:<5} | {mean_curve[-1]:6.2f}% {ci_str:<17} | {avg_time:10.2f}s   | {best_acc:6.2f}%")

    print("=" * 95)

    # =========================================================================
    # PLOT GENERATION
    # =========================================================================
    plt.figure(figsize=(11, 7))
    k_vals = list(range(K_REPEATS + 1))

    palette = {
        "mnist": "#1f77b4",
        "fashion": "#ff7f0e",
        "kmnist": "#2ca02c",
        "usps": "#d62728",
        "medmnist": "#9467bd",
        "omniglot": "#8c564b"
    }

    for d_name, _ in dataset_configs:
        summary = dataset_summary[d_name]
        mean = summary['mean_curve']
        ci = summary['ci_curve']
        color = palette.get(d_name, None)

        plt.plot(k_vals, mean, marker='o', linewidth=2.2, label=f"{d_name.upper()}", color=color)

        if num_samples > 1:
            plt.fill_between(k_vals, mean - ci, mean + ci, color=color, alpha=0.18)

    title_suffix = f" (Averaged across {sample_label} with 95% CI)" if num_samples > 1 else ""
    plt.title(f"MetaLearnNCA Few-Shot Adaptation vs. Optimization Repeats (k){title_suffix}", fontsize=12, fontweight='bold')
    plt.xlabel("Optimization repeats on support set (k steps)", fontsize=11)
    plt.ylabel("Global Test Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(k_vals)
    plt.ylim(0, 105)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.95)
    plt.tight_layout()

    out_plot_name = "nca_eval_ci_comparison.png"
    plt.savefig(out_plot_name, dpi=300)
    print(f"\nSaved 95% CI graph as '{out_plot_name}'.")
    plt.show()


if __name__ == '__main__':
    main()