import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFilter
import matplotlib.pyplot as plt

# ==========================================
# 0. HYPERPARAMETERS (HIGH SPEED, ~4GB VRAM)
# ==========================================
ACTIVE_DYN_CHN = 32
META_DYN_CHN = 32
STATIC_CHN = 24
LOSS_CHN = 10

META_EPOCHS = 4000

# 2 tasks parallel x 8 accumulation passes = 16 tasks per optimizer step
MICRO_BATCH_SIZE = 2
ACCUM_STEPS = 8

META_LR = 1e-3
K_REPEATS = 3
N_STEPS = 16

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
# 1. MULTI-DATASET GPU VRAM PRE-LOADER & SAMPLER
# ==========================================
def load_dataset_vram(dataset_name):
    """
    Downloads and pre-processes any supported dataset once, grouping images by class
    and storing them directly in GPU VRAM for fast sampling.
    """
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
        transforms.Lambda(lambda x: 1.0 - x),  # Invert: background=0.0, stroke=1.0
        transforms.Normalize((0.5,), (0.5,))
    ])

    dname = dataset_name.lower()
    print(f"\nLoading '{dname.upper()}' dataset into GPU VRAM...")

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
            raise ImportError("Please install medmnist: 'pip install medmnist'")
        info = medmnist.INFO['organamnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split='test', download=True, transform=fashion_transform)
        targets = dataset.labels.flatten()
    elif dname == "omniglot":
        dataset = torchvision.datasets.Omniglot('./data', background=True, download=True, transform=omniglot_transform)
        targets = np.array([y for _, y in dataset])
    else:
        raise ValueError(f"Unsupported dataset: '{dataset_name}'")

    # Group all images by class into a GPU-resident dictionary
    class_dict = {}
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        raw_target = targets[idx]
        if isinstance(raw_target, (np.ndarray, list)):
            t = int(raw_target[0])
        elif torch.is_tensor(raw_target):
            t = int(raw_target.item())
        else:
            t = int(raw_target)

        if t not in class_dict:
            class_dict[t] = []
        class_dict[t].append(img)

    # Stack into GPU tensors per class
    vram_class_dict = {}
    total_elements = 0
    for t in sorted(class_dict.keys()):
        stacked = torch.stack(class_dict[t]).to(DEVICE)  # Shape: (N_c, 1, 28, 28)
        vram_class_dict[t] = stacked
        total_elements += stacked.nelement()

    vram_mb = (total_elements * 4) / 1e6
    print(f"Loaded '{dname.upper()}': {len(vram_class_dict)} classes, ~{vram_mb:.1f} MB in GPU VRAM.")
    return vram_class_dict


@torch.no_grad()
def sample_vectorized_meta_batch(class_dict, meta_batch_size=2, n_way=10, n_supp=1, n_qry=5):
    """
    Vectorized GPU-side meta-batch sampling supporting datasets with any class count.
    """
    available_classes = list(class_dict.keys())
    num_classes = len(available_classes)
    actual_n_way = min(n_way, num_classes)

    supp_imgs, qry_imgs = [], []

    for b in range(meta_batch_size):
        # Pick actual_n_way random classes
        chosen_indices = torch.randperm(num_classes, device=DEVICE)[:actual_n_way]
        task_classes = [available_classes[i] for i in chosen_indices]

        task_supp, task_qry = [], []

        for cls in task_classes:
            cls_imgs = class_dict[cls]
            num_available = cls_imgs.size(0)
            needed = n_supp + n_qry

            if num_available >= needed:
                perm = torch.randperm(num_available, device=DEVICE)[:needed]
                selected = cls_imgs[perm]
            else:
                # Sample with replacement if class has fewer images than needed
                perm = torch.randint(0, num_available, (needed,), device=DEVICE)
                selected = cls_imgs[perm]

            task_supp.append(selected[:n_supp])
            task_qry.append(selected[n_supp:])

        supp_imgs.append(torch.cat(task_supp, dim=0))
        qry_imgs.append(torch.cat(task_qry, dim=0))

    supp_x = torch.stack(supp_imgs)  # (B_meta, B_supp, 1, H, W)
    qry_x = torch.stack(qry_imgs)    # (B_meta, B_qry, 1, H, W)

    supp_y = torch.arange(actual_n_way, device=DEVICE).repeat_interleave(n_supp).repeat(meta_batch_size, 1)
    qry_y = torch.arange(actual_n_way, device=DEVICE).repeat_interleave(n_qry).repeat(meta_batch_size, 1)

    return supp_x, supp_y, qry_x, qry_y


# ==========================================
# 2. MODEL DEFINITIONS (DEPTHWISE GROUPED)
# ==========================================
class BaseNCA(nn.Module):
    def __init__(self, c_in, c_out, hidden_size):
        super().__init__()
        self.c_in = c_in
        ident = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        lap = torch.tensor([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
        base_filters = torch.stack([ident, sobel_x, sobel_x.T, lap])  # (4, 3, 3)

        grouped_filters = base_filters.repeat(c_in, 1, 1).unsqueeze(1)
        self.register_buffer('filters', grouped_filters)

        self.w1 = nn.Conv2d(c_in * 4, hidden_size, 1)
        self.w2 = nn.Conv2d(hidden_size, c_out, 1, bias=False)
        self.w2.weight.data.div_(5)

    def perceive(self, x):
        x_pad = F.pad(x, [1, 1, 1, 1], mode='circular')
        return F.conv2d(x_pad, self.filters, groups=self.c_in)

    def grid_update(self, x):
        return self.w2(F.relu(self.w1(self.perceive(x)), inplace=True))


class ActiveNCA(BaseNCA):
    def __init__(self):
        c_in = 1 + STATIC_CHN + ACTIVE_DYN_CHN
        super().__init__(c_in, ACTIVE_DYN_CHN, hidden_size=256)

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
        c_in = LOSS_CHN + META_DYN_CHN
        super().__init__(c_in, META_DYN_CHN, hidden_size=256)

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


def get_pixel_loss_vectorized(pred_dyn, targets, images):
    alive_mask = (images > -0.8).float()
    pred_logits = pred_dyn[:, :LOSS_CHN, :, :]
    probs = F.softmax(pred_logits, dim=1)

    B_tot, _, H, W = images.shape
    spatial_targets = torch.zeros_like(pred_logits)
    spatial_targets.scatter_(1, targets.reshape(B_tot, 1, 1, 1).expand(-1, 1, H, W), 1.0)

    pixel_error = (spatial_targets - probs) * alive_mask
    return pixel_error, pred_logits, alive_mask


# ==========================================
# 3. HIGH-SPEED TRAINING FUNCTION
# ==========================================
plt.ion()
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
fig_acc, ax_acc = plt.subplots(1, 1, figsize=(6, 3))


def run_training(dataset_name="omniglot", seed=42):
    set_determinism(seed)
    class_dict = load_dataset_vram(dataset_name)

    active_nca = ActiveNCA().to(DEVICE)
    meta_nca = MetaNCA().to(DEVICE)
    accs = []

    fig.suptitle(f"Active-NCA & Meta-NCA Live Interaction [{dataset_name.upper()}]", fontsize=14, fontweight='bold')
    ax_acc.set_title(f"Accuracy Curve [{dataset_name.upper()}]", fontsize=11, fontweight='bold')

    try:
        active_nca.grid_update = torch.compile(active_nca.grid_update)
        meta_nca.grid_update = torch.compile(meta_nca.grid_update)
        print("Successfully compiled NCA grid_update engines with PyTorch 2.x.")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    optimizer = optim.AdamW(
        list(active_nca.parameters()) + list(meta_nca.parameters()),
        lr=META_LR, weight_decay=1e-5
    )

    scaler = torch.amp.GradScaler('cuda')

    print(f"\n>>> Starting Run | Dataset: {dataset_name.upper()} | Seed: {seed} | GPU: {torch.cuda.get_device_name(0)} <<<")
    start_time = time.time()

    for epoch in range(1, META_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_acc = 0.0

        for _ in range(ACCUM_STEPS):
            supp_x, supp_y, qry_x, qry_y = sample_vectorized_meta_batch(
                class_dict, meta_batch_size=MICRO_BATCH_SIZE, n_way=10, n_supp=1, n_qry=5
            )
            B_meta, B_supp, C_img, H, W = supp_x.shape
            _, B_qry, _, _, _ = qry_x.shape

            flat_supp_x = supp_x.reshape(B_meta * B_supp, C_img, H, W)
            flat_supp_y = supp_y.reshape(B_meta * B_supp)
            flat_qry_x = qry_x.reshape(B_meta * B_qry, C_img, H, W)
            flat_qry_y = qry_y.reshape(B_meta * B_qry)

            N_steps_current = random.randint(N_STEPS, N_STEPS + 3)
            K_repeats_current = random.randint(K_REPEATS, K_REPEATS + 2)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                S_task = torch.rand(B_meta, 1, STATIC_CHN, H, W, device=DEVICE)
                D_meta = None

                # Support Adaptation
                for _ in range(K_repeats_current):
                    flat_S = S_task.expand(-1, B_supp, -1, -1, -1).reshape(B_meta * B_supp, STATIC_CHN, H, W)
                    supp_dyn = active_nca(flat_supp_x, flat_S, steps=N_steps_current * 2)

                    pixel_loss, _, _ = get_pixel_loss_vectorized(supp_dyn, flat_supp_y, flat_supp_x)
                    D_meta = meta_nca(flat_supp_x, pixel_loss, dyn_state=D_meta, steps=N_steps_current)

                    D_meta_grouped = D_meta.reshape(B_meta, B_supp, META_DYN_CHN, H, W)
                    S_task = D_meta_grouped[:, :, :STATIC_CHN].mean(dim=1, keepdim=True)

                # Query Evaluation
                flat_S_qry = S_task.expand(-1, B_qry, -1, -1, -1).reshape(B_meta * B_qry, STATIC_CHN, H, W)
                qry_dyn = active_nca(flat_qry_x, flat_S_qry, steps=N_steps_current * 2)

                _, qry_logits, alive_mask = get_pixel_loss_vectorized(qry_dyn, flat_qry_y, flat_qry_x)

                spatial_targets = flat_qry_y.reshape(B_meta * B_qry, 1, 1).expand(-1, H, W)
                ce_loss_raw = F.cross_entropy(qry_logits, spatial_targets, reduction='none')
                loss_micro = (ce_loss_raw * alive_mask.squeeze(1)).sum() / alive_mask.sum().clamp(min=1.0)
                loss_scaled = loss_micro / ACCUM_STEPS

            scaler.scale(loss_scaled).backward()

            with torch.no_grad():
                logits = (qry_logits * alive_mask).sum(dim=(2, 3)) / alive_mask.sum(dim=(2, 3)).clamp(min=1.0)
                acc = (logits.argmax(dim=1) == flat_qry_y).float().mean().item()
                epoch_acc += (acc * 100.0) / ACCUM_STEPS

        epoch_loss += loss_scaled.detach()
        if torch.isnan(epoch_loss).any() and epoch > 10:
            print(f"NaN loss detected on {dataset_name}. Skipping...")
            break

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(active_nca.parameters()) + list(meta_nca.parameters()),
            max_norm=1.0,
            foreach=True
        )
        scaler.step(optimizer)
        scaler.update()



        accs.append(epoch_acc)

        # Live Visualization (Every 25 epochs)
        if epoch % 25 == 0 or epoch == 1:
            axes[0].clear()
            axes[0].imshow(flat_supp_x[0, 0].detach().cpu(), cmap='gray')
            axes[0].set_title(f"Support: {flat_supp_y[0].item()}", fontsize=10)

            supp_pred_map = (supp_dyn[0, :10].argmax(dim=0).float() + 1.0) * (flat_supp_x[0, 0] > -0.8)
            axes[1].clear()
            axes[1].imshow(supp_pred_map.detach().cpu(), cmap='jet', vmin=0, vmax=10)
            axes[1].set_title("Supp Pred", fontsize=10)

            axes[2].clear()
            axes[2].imshow(flat_qry_x[0, 0].detach().cpu(), cmap='gray')
            axes[2].set_title(f"Query: {flat_qry_y[0].item()}", fontsize=10)

            qry_pred_map = (qry_dyn[0, :10].argmax(dim=0).float() + 1.0) * (flat_qry_x[0, 0] > -0.8)
            axes[3].clear()
            axes[3].imshow(qry_pred_map.detach().cpu(), cmap='jet', vmin=0, vmax=10)
            axes[3].set_title("Qry Pred", fontsize=10)

            axes[4].clear()
            axes[4].imshow(S_task[0, 0, 0].detach().cpu().float().numpy(), cmap='viridis')
            axes[4].set_title("S_task Ch 0", fontsize=10)

            for ax in axes:
                ax.axis('off')

            fig.canvas.draw()
            fig.canvas.flush_events()

            ax_acc.clear()
            ax_acc.plot(accs, color='#1f77b4', linewidth=1.8)
            ax_acc.set_title(f"{dataset_name.upper()} Accuracy ({accs[-1]:.1f}%)", fontsize=11, fontweight='bold')
            ax_acc.set_xlabel("Epochs", fontsize=9)
            ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
            ax_acc.set_ylim(0, 105)
            ax_acc.grid(True, linestyle=':', alpha=0.6)

            fig_acc.canvas.draw()
            fig_acc.canvas.flush_events()
            plt.pause(0.001)

            elapsed = time.time() - start_time
            sec_per_ep = elapsed / epoch
            eta_mins = (META_EPOCHS - epoch) * sec_per_ep / 60.0
            print(f"[{dataset_name.upper()}] Epoch {epoch:4d}/{META_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | {sec_per_ep:.2f}s/ep | ETA: {eta_mins:.1f}m")

    if torch.isnan(epoch_loss).any():
        return

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(active_nca.state_dict(), f"checkpoints/active_nca_{dataset_name.lower()}.pth")
    torch.save(meta_nca.state_dict(), f"checkpoints/meta_nca_{dataset_name.lower()}.pth")
    print(f"--> Saved '{dataset_name.upper()}' checkpoints to checkpoints/.")


# ==========================================
# 4. DATASET EXECUTION LOOP
# ==========================================
if __name__ == '__main__':
    # Add, remove, or reorder the datasets you want to train on:
    datasets_to_train = [
        "medmnist",
        "kmnist",
        "mnist",
        "fashion",
        "usps"

    ]

    for dname in datasets_to_train:
        run_training(dataset_name=dname, seed=2)