import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import matplotlib.pyplot as plt
""" Main script that implements the training algo found n the paper"""
# ==========================================
# 0. HYPERPARAMETERS & CONFIG
# ==========================================
ACTIVE_DYN_CHN = 32  # Channels 0-9 are classes
META_DYN_CHN = 32  # Meta-NCA's internal reasoning channels
STATIC_CHN = 24  # The size of the spatial communication grid (Meta-NCA -> Active-NCA)
LOSS_CHN = 10    # Spatial loss (10 classes)

META_EPOCHS = 5000
META_BATCH_SIZE = 32
META_LR = 1e-3

K_REPEATS = 3  # Outer interactions (Meta-NCA updates Active-NCA K times)
N_STEPS = 16 # Inner grid updates for both NCAs per interaction
# Note: K_REPEATS * N_STEPS = 48 total steps. This is > 28 (MNIST size),
# allowing "rules" to diffuse fully across the grid!

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- PLOTTING SETUP ---
plt.ion()
# Expanded to 1x6 to include the accuracy graph
fig, axes = plt.subplots(1, 5, figsize=(18, 3))
fig.suptitle("Active-NCA & Meta-NCA Interaction (Spatial)", fontsize=16)


def set_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==========================================
# 1. TASK SAMPLING
# ==========================================
def get_omniglot_dataset():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),  # Invert: now background is 0.0, stroke is 1.0



        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download the background set (964 classes)
    dataset = torchvision.datasets.Omniglot('./data', background=True, download=True, transform=transform)

    # Omniglot doesn't have a .targets attribute natively exposed like MNIST.
    # We quickly extract them so the sample_task function can filter by class.
    print("Extracting Omniglot targets...")
    dataset.targets = torch.tensor([target for _, target in dataset._flat_character_images])
    print("Done.")

    return dataset


def sample_task(dataset, n_way=10, n_support=1, n_query=5):
    targets = dataset.targets.numpy()
    support_idx, query_idx = [], []

    # Dynamically find available classes (Omniglot has 964, not 10)
    available_classes = list(np.unique(targets))
    np.random.shuffle(available_classes)
    chosen_classes = available_classes[:n_way]
    class_to_surrogate = {true_cls: surr for surr, true_cls in enumerate(chosen_classes)}

    for true_cls in chosen_classes:
        pos = np.where(targets == true_cls)[0]
        chosen = np.random.choice(pos, n_support + n_query, replace=False)
        support_idx.extend(chosen[:n_support])
        query_idx.extend(chosen[n_support:])

    np.random.shuffle(support_idx)
    np.random.shuffle(query_idx)

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


# ==========================================
# 2. BASE NCA ARCHITECTURE
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
        self.w2.weight.data.zero_()

    def perceive(self, x):
        # Native grouped conv - no reshape, engages Tensor Cores
        x_pad = F.pad(x, [1, 1, 1, 1], mode='circular')
        return F.conv2d(x_pad, self.filters, groups=self.c_in)

    def grid_update(self, x):
        return self.w2(F.relu(self.w1(self.perceive(x)), inplace=True))


class ActiveNCA(BaseNCA):
    def __init__(self):
        c_in = 1 + STATIC_CHN + ACTIVE_DYN_CHN
        super().__init__(c_in, ACTIVE_DYN_CHN, hidden_size=240)

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
        super().__init__(c_in, META_DYN_CHN, hidden_size=240)

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


# ==========================================
# 5. LOSS HELPER
# ==========================================
def get_pixel_loss(pred_dyn, targets, images):
    B, _, H, W = images.shape
    alive_mask = (images > -0.8).float()

    # 1. Raw logits for the global Cross-Entropy Loss
    pred_logits = pred_dyn[:, :LOSS_CHN, :, :]

    # 2. Convert logits to probabilities (Softmax along class dimension)
    probs = F.softmax(pred_logits, dim=1)

    # 3. Create One-Hot Spatial Targets
    spatial_targets = torch.zeros_like(pred_logits)
    for b in range(B):
        spatial_targets[b, targets[b], :, :] = 1.0

    # 4. The Bounded Local Error Signal for the Meta-NCA (Target - Probabilities)
    # Strictly bounded between -1.0 and 1.0!
    pixel_error = (spatial_targets - probs) * alive_mask

    return pixel_error, pred_logits, alive_mask



# ==========================================
# 6. TRAINING ALGORITHM
# ==========================================
def main():
    set_determinism(42)
    # Changed dataset initialization to Omniglot
    dataset = get_omniglot_dataset()


    active_nca = ActiveNCA().to(DEVICE)
    meta_nca = MetaNCA().to(DEVICE)


    # Both sets of weights are learned CONCURRENTLY
    optimizer = optim.AdamW(list(active_nca.parameters()) + list(meta_nca.parameters()), lr=META_LR, weight_decay=1e-5)



    accs = []  # Store accuracies for graphing

    for epoch in range(1, META_EPOCHS + 1):
        optimizer.zero_grad()
        epoch_query_loss = 0.0
        epoch_query_acc = 0.0

        print(f"\n=== Meta-Epoch {epoch}/{META_EPOCHS} ===")

        for task_idx in range(META_BATCH_SIZE):
            supp_x, supp_y, qry_x, qry_y = sample_task(dataset, n_support=1, n_query=5)

            B_supp, _, H, W = supp_x.shape
            S_task = torch.rand(1, STATIC_CHN, H, W, device=supp_x.device)
            N_STEPS2 = random.randint(N_STEPS, N_STEPS + 3)
            K_REPEATS2 = random.randint(K_REPEATS, K_REPEATS + 2)
            # --- A. INITIALIZE TASK REPRESENTATION (SPATIALLY) ---

            D_meta = None

            # --- B. INTERACTION LOOP (REPEAT K TIMES) ---
            for k in range(K_REPEATS2):
                S_expanded = S_task.expand(B_supp, -1, -1, -1)
                supp_dyn = active_nca(supp_x, S_expanded, steps=N_STEPS2*2)

                pixel_loss, _, _ = get_pixel_loss(supp_dyn, supp_y, supp_x)

                D_meta = meta_nca(supp_x, pixel_loss, dyn_state=D_meta, steps=N_STEPS2)


                S_update = D_meta[:, :STATIC_CHN, :, :].mean(dim=0, keepdim=True)

                S_task = S_update

                # --- C. QUERY SET EVALUATION ---
            S_expanded_qry = S_task.expand(qry_x.size(0), -1, -1, -1)
            qry_dyn = active_nca(qry_x, S_expanded_qry, steps=N_STEPS*2)

            _, qry_logits, alive_mask = get_pixel_loss(qry_dyn, qry_y, qry_x)
            B_qry, C_qry, H_qry, W_qry = qry_logits.shape
            spatial_targets = qry_y.view(B_qry, 1, 1).expand(B_qry, H_qry, W_qry)

            # Pixel-wise Cross Entropy
            ce_loss_raw = F.cross_entropy(qry_logits, spatial_targets, reduction='none')
            ce_loss_masked = ce_loss_raw * alive_mask.squeeze(1)
            loss_qry = ce_loss_masked.sum() / alive_mask.sum().clamp(min=1.0)
            (loss_qry/META_BATCH_SIZE).backward()

            epoch_query_loss += loss_qry.item()

            with torch.no_grad():
                logits = (qry_logits * alive_mask).sum(dim=(2, 3)) / alive_mask.sum(dim=(2, 3)).clamp(min=1.0)
                acc = (logits.argmax(dim=1) == qry_y).float().mean().item()
                epoch_query_acc += acc

        torch.nn.utils.clip_grad_norm_(active_nca.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(meta_nca.parameters(), 1.0)
        optimizer.step()


        # Update Accuracy tracking
        avg_loss = epoch_query_loss / META_BATCH_SIZE
        avg_acc = epoch_query_acc / META_BATCH_SIZE
        accs.append(avg_acc * 100)

        # --- E. LIVE VISUALIZATION ---
        if epoch % 20 == 0:
            axes[0].clear();
            axes[0].imshow(supp_x[0, 0].cpu(), cmap='gray')
            axes[0].set_title(f"Support True: {supp_y[0].item()}")

            supp_pred_map = supp_dyn[0, :10].argmax(dim=0).float() * (supp_x[0, 0] > -0.8)
            axes[1].clear();
            axes[1].imshow(supp_pred_map.cpu(), cmap='jet')
            axes[1].set_title("Support Prediction")

            axes[2].clear();
            axes[2].imshow(qry_x[0, 0].cpu(), cmap='gray')
            axes[2].set_title(f"Query True: {qry_y[0].item()}")

            qry_pred_map = qry_dyn[0, :10].argmax(dim=0).float() * (qry_x[0, 0] > -0.8)
            axes[3].clear();
            axes[3].imshow(qry_pred_map.detach().cpu(), cmap='jet')
            axes[3].set_title("Query Prediction")

            axes[4].clear()
            axes[4].imshow(S_task[0, 0].detach().cpu().numpy(), cmap='viridis')
            axes[4].set_title("Static Channel 0 (Spatial)")

            # Turn off image axes only
            for ax in axes[:5]: ax.axis('off')

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
            plt.figure(3)
            plt.plot(accs)
            plt.show(block=False)
            plt.pause(0.01)

        print(f"Mean Query MSE: {avg_loss:.5f} | Mean Query Acc: {avg_acc * 100:.2f}%")

        # --- F. SAVING MODELS ---
        if epoch % 10 == 0:
            torch.save(active_nca.state_dict(), 'active_nca.pth')
            torch.save(meta_nca.state_dict(), 'meta_nca.pth')
            print(f"--> Saved Active-NCA and Meta-NCA checkpoints at epoch {epoch}.")


if __name__ == '__main__':
    main()