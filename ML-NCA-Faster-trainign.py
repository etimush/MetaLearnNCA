import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import time
import matplotlib.pyplot as plt
""" Refactored trainign script for MLNCA, functionally the same as the main script, just rins faster through tensor and batch tricks."""
# ==========================================
# 0. HYPERPARAMETERS (HIGH SPEED, ~4GB VRAM)
# ==========================================
ACTIVE_DYN_CHN = 32
META_DYN_CHN = 32
STATIC_CHN = 24
LOSS_CHN = 10

META_EPOCHS = 4000

# 2 tasks parallel x 4 accumulation passes = 8 tasks per optimizer step
# Cuts loop overhead by 4x compared to 16 accumulation steps!
MICRO_BATCH_SIZE = 2
ACCUM_STEPS = 8

META_LR = 2e-3
K_REPEATS = 3
N_STEPS = 16                # 14 steps diffuses 42 px across 28x28 grid

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
# 1. GPU VRAM PRE-LOADER & SAMPLER
# ==========================================
def load_omniglot_vram():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Pre-loading Omniglot into GPU VRAM...")
    dataset = torchvision.datasets.Omniglot('./data', background=True, download=True, transform=transform)

    class_dict = {}
    for img, target in dataset:
        if target not in class_dict:
            class_dict[target] = []
        class_dict[target].append(img)

    classes_tensor = [torch.stack(class_dict[target]) for target in sorted(class_dict.keys())]
    omni_vram = torch.stack(classes_tensor).to(DEVICE)
    print(f"Omniglot loaded: {omni_vram.shape} (~60 MB in VRAM).")
    return omni_vram


@torch.no_grad()
def sample_vectorized_meta_batch(omni_vram, meta_batch_size=2, n_way=10, n_supp=1, n_qry=5):
    num_classes, num_ex, C, H, W = omni_vram.shape

    chosen_classes = torch.stack([
        torch.randperm(num_classes, device=DEVICE)[:n_way] for _ in range(meta_batch_size)
    ])

    supp_imgs, qry_imgs = [], []
    for b in range(meta_batch_size):
        task_classes = chosen_classes[b]
        task_data = omni_vram[task_classes]

        perm = torch.randperm(num_ex, device=DEVICE)[:n_supp + n_qry]
        selected = task_data[:, perm]

        supp = selected[:, :n_supp].reshape(n_way * n_supp, C, H, W)
        qry = selected[:, n_supp:].reshape(n_way * n_qry, C, H, W)

        supp_imgs.append(supp)
        qry_imgs.append(qry)

    supp_x = torch.stack(supp_imgs)
    qry_x = torch.stack(qry_imgs)

    supp_y = torch.arange(n_way, device=DEVICE).repeat_interleave(n_supp).repeat(meta_batch_size, 1)
    qry_y = torch.arange(n_way, device=DEVICE).repeat_interleave(n_qry).repeat(meta_batch_size, 1)

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

        # Depthwise grouped filters: Shape (c_in * 4, 1, 3, 3)
        grouped_filters = base_filters.repeat(c_in, 1, 1).unsqueeze(1)
        self.register_buffer('filters', grouped_filters)

        self.w1 = nn.Conv2d(c_in * 4, hidden_size, 1)
        self.w2 = nn.Conv2d(hidden_size, c_out, 1, bias=False)
        self.w2.weight.data.div_(5)

    def perceive(self, x):
        # Native grouped conv - no reshape, engages Tensor Cores
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
fig.suptitle("Active-NCA & Meta-NCA Live Interaction", fontsize=14, fontweight='bold')
fig_acc, ax_acc = plt.subplots(1, 1, figsize=(6, 3))


def run_training(seed=42):
    set_determinism(seed)
    omni_vram = load_omniglot_vram()

    active_nca = ActiveNCA().to(DEVICE)
    meta_nca = MetaNCA().to(DEVICE)
    accs = []
    # COMPILE THE CELLULAR TRANSITION KERNELS (FAST & MEMORY SAFE)
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=META_EPOCHS, eta_min=5e-4)
    scaler = torch.amp.GradScaler('cuda')

    # Live GUI Windows

    print(f"\n>>> Starting Run | Seed: {seed} | GPU: {torch.cuda.get_device_name(0)} <<<")
    start_time = time.time()

    for epoch in range(1, META_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_acc = 0.0

        # Exactly 4 accumulation steps per epoch (fast & lightweight)
        for _ in range(ACCUM_STEPS):
            supp_x, supp_y, qry_x, qry_y = sample_vectorized_meta_batch(
                omni_vram, meta_batch_size=MICRO_BATCH_SIZE, n_way=10, n_supp=1, n_qry=5
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

                # Support Adaptation (2 tasks = 20 images)
                for _ in range(K_repeats_current):
                    flat_S = S_task.expand(-1, B_supp, -1, -1, -1).reshape(B_meta * B_supp, STATIC_CHN, H, W)
                    supp_dyn = active_nca(flat_supp_x, flat_S, steps=N_steps_current * 2)

                    pixel_loss, _, _ = get_pixel_loss_vectorized(supp_dyn, flat_supp_y, flat_supp_x)
                    D_meta = meta_nca(flat_supp_x, pixel_loss, dyn_state=D_meta, steps=N_steps_current)

                    D_meta_grouped = D_meta.reshape(B_meta, B_supp, META_DYN_CHN, H, W)
                    S_task = D_meta_grouped[:, :, :STATIC_CHN].mean(dim=1, keepdim=True)

                # Query Evaluation (2 tasks = 100 images)
                flat_S_qry = S_task.expand(-1, B_qry, -1, -1, -1).reshape(B_meta * B_qry, STATIC_CHN, H, W)
                qry_dyn = active_nca(flat_qry_x, flat_S_qry, steps=N_steps_current * 2)

                _, qry_logits, alive_mask = get_pixel_loss_vectorized(qry_dyn, flat_qry_y, flat_qry_x)

                spatial_targets = flat_qry_y.reshape(B_meta * B_qry, 1, 1).expand(-1, H, W)
                ce_loss_raw = F.cross_entropy(qry_logits, spatial_targets, reduction='none')
                loss_micro = (ce_loss_raw * alive_mask.squeeze(1)).sum() / alive_mask.sum().clamp(min=1.0)
                loss_scaled = loss_micro / ACCUM_STEPS

            # Backward dynamically frees activations step-by-step
            scaler.scale(loss_scaled).backward()


            with torch.no_grad():
                logits = (qry_logits * alive_mask).sum(dim=(2, 3)) / alive_mask.sum(dim=(2, 3)).clamp(min=1.0)
                acc = (logits.argmax(dim=1) == flat_qry_y).float().mean().item()
                epoch_acc += (acc * 100.0) / ACCUM_STEPS

        epoch_loss += loss_scaled.detach()
        if torch.isnan(epoch_loss).any() and epoch >10:
            break


        # Step optimizer once every 4 micro-batches
        scaler.unscale_(optimizer)
        # Fast single-launch multi-tensor norm clipping
        torch.nn.utils.clip_grad_norm_(
            list(active_nca.parameters()) + list(meta_nca.parameters()),
            max_norm=1.0,
            foreach=True
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        accs.append(epoch_acc)

        # Live Visualization (Every 25 epochs)
        if epoch % 25 == 0 or epoch == 1:
            axes[0].clear()
            axes[0].imshow(flat_supp_x[0, 0].detach().cpu(), cmap='gray')
            axes[0].set_title(f"Support: {flat_supp_y[0].item()}", fontsize=10)

            supp_pred_map = supp_dyn[0, :10].argmax(dim=0).float() * (flat_supp_x[0, 0] > -0.8)
            axes[1].clear()
            axes[1].imshow(supp_pred_map.detach().cpu(), cmap='jet')
            axes[1].set_title("Supp Pred", fontsize=10)

            axes[2].clear()
            axes[2].imshow(flat_qry_x[0, 0].detach().cpu(), cmap='gray')
            axes[2].set_title(f"Query: {flat_qry_y[0].item()}", fontsize=10)

            qry_pred_map = qry_dyn[0, :10].argmax(dim=0).float() * (flat_qry_x[0, 0] > -0.8)
            axes[3].clear()
            axes[3].imshow(qry_pred_map.detach().cpu(), cmap='jet')
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
            ax_acc.set_title(f"Accuracy Curve ({accs[-1]:.1f}%)", fontsize=11, fontweight='bold')
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
            print(f"Epoch {epoch:4d}/{META_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | {sec_per_ep:.2f}s/ep | ETA: {eta_mins:.1f}m")
    if torch.isnan(epoch_loss).any():
        return
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(active_nca.state_dict(), f"checkpoints/active_nca_seed{seed}.pth")
    torch.save(meta_nca.state_dict(), f"checkpoints/meta_nca_seed{seed}.pth")
    print(f"Finished Seed {seed}. Checkpoints saved.")


if __name__ == '__main__':
    seeds = [i for i in range(215,5000)]
    for seed in seeds:
        run_training(seed=seed)