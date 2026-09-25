# MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata

[![ICLR 2027](https://img.shields.io/badge/ICLR%202027-Under%20Review-orange.svg)](https://arxiv.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of **"MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata"**.

---

## 📌 Overview

Few-shot meta-learning traditionally formulates task adaptation either through **analytical gradient descent across unrolled computational graphs** (e.g., MAML, FOMAML), incurring severe test-time backpropagation costs, or through **metric-based distance comparisons over flattened 1D feature vectors** (e.g., ProtoNet, MatchingNet), discarding native 2D spatial inductive biases.

**MetaLearnNCA** introduces a decentralized, non-von Neumann paradigm where few-shot task adaptation emerges through the dynamical interaction of two coupled Neural Cellular Automata (NCAs) operating with **zero analytical gradients at adaptation and test time**:

1. **Active-NCA ($f_\theta$):** Executes the perceptual recognition task on a 2D canvas, conditioned on a continuous 2D spatial memory grid termed the **spatial program** ($S_{\text{task}} \in \mathbb{R}^{24 \times H \times W}$).
2. **Meta-NCA ($g_\phi$):** Operates as a decentralized cellular optimizer. Instead of computing chain-rule backpropagation, it reads stroke-masked residual spatial error fields ($E_{\text{pixel}}$) on the support set and diffuses corrective updates directly into $S_{\text{task}}$.

Once adaptation concludes on the support set, $S_{\text{task}}$ is frozen and broadcast to query samples for forward-only classification.

```
       Support Images (X_supp) ──┐
                                 ▼
                     ┌───────────────────────┐
                     │   Active-NCA (f_θ)    │ ◄─── Spatial Program (S_task)
                     │ (Perception & Logits) │          ▲
                     └──────────┬────────────┘          │
                                │ Logits                │ Dynamic Program
                                ▼                       │ Updates
                     ┌───────────────────────┐          │
                     │  Spatial Error Mask   │          │
                     │ (E_pixel = Y - Probs) │          │
                     └──────────┬────────────┘          │
                                │ Residual Error        │
                                ▼                       │
                     ┌───────────────────────┐          │
                     │    Meta-NCA (g_ϕ)     │──────────┘
                     │  (Cellular Optimizer) │
                     └───────────────────────┘
```



---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `ML-NCA-Training.py` | Reference meta-training pipeline on Omniglot with live interactive visualization. |
| `ML-NCA-Faster-trainign.py` | High-speed, VRAM-cached meta-training script utilizing `torch.compile`, mixed-precision AMP (`fp16`), and micro-batch gradient accumulation. |
| `ML-NCA-Train-othe-datasets.py` | Multi-dataset training script to train specialized NCA checkpoints across MNIST, Fashion-MNIST, KMNIST, USPS, and MedMNIST. |
| `ML-NCA-Testing.py` | Main evaluation script for few-shot adaptation trajectories ($k=0 \dots 5$), exact FLOP counting, coordinate shift tests, and multi-seed 95% CI benchmarking. |
| `ML-NCA-Cross-dataset-Testing.py` | Exhaustive $6 \times 6$ cross-dataset transfer benchmark evaluating all source-target dataset pairs across 50 seeds, outputting LaTeX/CSV matrices and publication heatmaps. |
| `ML-NCA-Ablation.py` | Full ablation suite: ablates dynamic channels (10–31) and program channels (0–23), runs circular damage studies, additive noise injections, and visual channel atlas generators. |
| `ML-NCA-Optim-surface.py` | Computes and plots 3D generalization landscape surfaces (Shots vs. Interaction Repeats $k$ vs. Accuracy). |
| `Train-Canonical-Models.py` | Canonical baselines benchmark suite: trains and evaluates **ProtoNet**, **MatchingNet**, **RelationNet**, and **FOMAML** using identical Conv4-64 backbones and data pipelines. |

---

## ⚙️ Architecture & Channel Layout

```
Active-NCA (57 Channels Total):
 ├── [0]       Observation Channel (Raw Grayscale Image)
 ├── [1..24]   Static Spatial Program S_task (Read-only bias from Meta-NCA)
 └── [25..56]  Dynamic Reasoning Channels (32 Channels)
                ├── [25..34] Logits for 10 candidate classes
                └── [35..56] Latent Recurrent Scratchpad Memory

Meta-NCA (42 Channels Total):
 ├── [0..9]    Spatial Error Residuals E_pixel (Target - Predicted Probabilities)
 └── [10..41]  Dynamic Cellular State (32 Channels)
                └── [10..33] Extracted & Mean-Pooled to form updated S_task (24 Channels)
```

---


## 🏋️ Meta-Training

### Fast VRAM-Cached Training on Omniglot (Recommended)
`ML-NCA-Faster-trainign.py` caches all Omniglot characters directly into GPU memory, engages Tensor Cores via grouped depthwise convolutions, uses AMP `float16`, and compiles the grid updates via `torch.compile`:

```bash
python ML-NCA-Faster-trainign.py
```
*Checkpoints are saved automatically to `checkpoints/active_nca_seed<SEED>.pth` and `checkpoints/meta_nca_seed<SEED>.pth`.*

### Reference Training with Live Inspection GUI
To observe real-time spatial predictions and program field updates during training:

```bash
python ML-NCA-Training.py
```

### Training Dedicated Models on Other Domains
To train MetaLearnNCA directly on other datasets (MNIST, Fashion-MNIST, KMNIST, USPS, MedMNIST):

```bash
python ML-NCA-Train-othe-datasets.py
```

---

## 🧪 Evaluation & Benchmarking

### 1. Comprehensive Out-of-Distribution Transfer
To evaluate an Omniglot-trained checkpoint across all benchmark domains (10-way, $k$-shot) with 95% Student-$t$ Confidence Intervals:

```bash
python ML-NCA-Testing.py \
    --checkpoints checkpoints \
    --seed 0 \
    --run_eval_loop \
    --eval_loops 10 \
    --k_shot 10
```

### 2. Full $6 \times 6$ Cross-Dataset Transfer Matrix
To evaluate cross-domain adaptation across all 36 combinations of source and target datasets:

```bash
python ML-NCA-Cross-dataset-Testing.py --seeds 50 --k_repeats 5
```
*Outputs `cross_dataset_transfer_matrix_full.csv` and generates `cross_dataset_transfer_heatmap_full.png`.*

### 3. Canonical Baselines Comparison (ProtoNet, MatchingNet, RelationNet, FOMAML)
To train and evaluate the Conv4-64 baselines under the exact same few-shot evaluation protocol:

```bash
python Train-Canonical-Models.py \
    --epochs 5000 \
    --meta_batch 32 \
    --shots 1 5 10 \
    --test_runs 10 \
    --models FOMAML ProtoNet MatchingNet RelationNet
```

---

## 🔬 Ablations, Fault Tolerance & Visualization

`ML-NCA-Ablation.py` provides an end-to-end ablation and analysis suite for any target dataset:

```bash
python ML-NCA-Ablation.py \
    --dataset mnist \
    --seed 0 \
    --k_shot 10 \
    --noise_scale 0.5 \
    --noise_type gaussian
```

This single command automatically executes:
1. **Dynamic Hidden Channel Ablation (10–31):** Tests sensitivity when individual reasoning channels are corrupted with Gaussian noise.
2. **Program Channel Ablation (0–23):** Zeroes out individual static program channels post-adaptation to evaluate pre-activation redundancy.
3. **Physical Circular Damage Study:** Injects circular punctures ($r = 4\text{ px}$) into $S_{\text{task}}$ across all channels and plots degradation curves.
4. **Additive Spatial Noise Sweep:** Perturbs $0\% \to 100\%$ of spatial coordinates in $S_{\text{task}}$ to verify robustness against high thermal noise.
5. **Channel Atlas Generation:** Outputs complete 46-channel visual atlases (`hidden_channels_full_<dataset>.png`, `program_channels_full_<dataset>.png`, and `compressed_1x8_channels_<dataset>.png`).
6. **Solution Manifold Difference Heatmaps:** Compares program representations $S_{\text{task}}^{(\text{Seed } 42)} - S_{\text{task}}^{(\text{Seed } 43)}$ to visualize functional degeneracy.

### 3D Few-Shot Generalization Landscape Surfaces
To render 3D meshes of accuracy across shot schedules and optimization repeats ($k = 1 \dots 5$):

```bash
python ML-NCA-Optim-surface.py --checkpoints checkpoints --seeds 3
```
*Outputs `surface_plot_k_vs_shots_all_datasets.png`.*

---

## 📊 Key Results

### Few-Shot Transfer Performance (Omniglot Meta-Trained Models)
*Evaluated across 10 independent test splits (mean accuracy % $\pm$ 95% Confidence Interval):*

| Target Domain | Shot | MatchingNet | RelationNet | ProtoNet | FOMAML *(5 steps)* | **MetaLearnNCA (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Omniglot** *(In-Dist)* | 1 | $63.47 \pm 7.79$ | $84.84 \pm 3.74$ | $\mathbf{92.26 \pm 3.15}$ | $87.84 \pm 3.67$ | $87.95 \pm 4.16$ |
| | 5 | $75.47 \pm 4.54$ | $92.73 \pm 3.09$ | $96.40 \pm 3.09$ | $\mathbf{96.93 \pm 1.41}$ | $95.20 \pm 2.13$ |
| | 10 | $77.40 \pm 6.37$ | $93.20 \pm 2.65$ | $97.40 \pm 1.27$ | $\mathbf{98.80 \pm 0.94}$ | $96.12 \pm 0.70$ |
| **MNIST** *(OOD)* | 1 | $44.79 \pm 4.93$ | $51.17 \pm 2.66$ | $55.32 \pm 1.87$ | $48.16 \pm 2.98$ | $\mathbf{65.02 \pm 3.07}$ |
| | 5 | $58.93 \pm 1.34$ | $63.74 \pm 1.39$ | $75.17 \pm 1.56$ | $70.80 \pm 1.72$ | $\mathbf{84.12 \pm 1.57}$ |
| | 10 | $58.66 \pm 1.15$ | $67.30 \pm 0.66$ | $79.17 \pm 1.48$ | $72.37 \pm 2.01$ | $\mathbf{89.71 \pm 0.43}$ |
| **Fashion-MNIST** *(OOD)*| 1 | $25.24 \pm 2.84$ | $20.82 \pm 2.59$ | $22.26 \pm 1.75$ | $31.39 \pm 3.21$ | $\mathbf{40.43 \pm 3.11}$ |
| | 5 | $31.68 \pm 0.82$ | $25.43 \pm 0.75$ | $29.66 \pm 0.91$ | $47.41 \pm 1.88$ | $\mathbf{51.34 \pm 2.67}$ |
| | 10 | $32.99 \pm 0.75$ | $25.91 \pm 1.58$ | $31.63 \pm 1.29$ | $50.53 \pm 1.82$ | $\mathbf{54.40 \pm 1.22}$ |
| **KMNIST** *(OOD)* | 1 | $17.15 \pm 1.46$ | $20.04 \pm 1.96$ | $22.57 \pm 2.35$ | $22.76 \pm 2.01$ | $\mathbf{30.28 \pm 3.14}$ |
| | 5 | $23.15 \pm 1.03$ | $26.06 \pm 1.60$ | $30.09 \pm 2.05$ | $33.66 \pm 1.66$ | $\mathbf{41.65 \pm 1.86}$ |
| | 10 | $25.61 \pm 1.38$ | $29.40 \pm 1.13$ | $35.42 \pm 1.42$ | $36.23 \pm 1.18$ | $\mathbf{51.26 \pm 1.27}$ |
| **USPS** *(OOD)* | 1 | $30.23 \pm 3.33$ | $24.14 \pm 3.46$ | $35.50 \pm 3.74$ | $42.40 \pm 4.62$ | $\mathbf{53.26 \pm 5.99}$ |
| | 5 | $34.39 \pm 3.03$ | $35.41 \pm 2.69$ | $51.07 \pm 2.04$ | $\mathbf{64.91 \pm 2.13}$ | $57.10 \pm 4.26$ |
| | 10 | $36.58 \pm 3.02$ | $38.81 \pm 1.19$ | $55.54 \pm 1.89$ | $\mathbf{69.75 \pm 2.16}$ | $65.02 \pm 3.10$ |
| **MedMNIST** *(OOD)* | 1 | $17.47 \pm 3.00$ | $10.79 \pm 3.83$ | $20.86 \pm 2.66$ | $\mathbf{22.82 \pm 4.74}$ | $18.54 \pm 2.16$ |
| | 5 | $26.61 \pm 3.19$ | $12.24 \pm 3.51$ | $24.85 \pm 2.34$ | $\mathbf{35.91 \pm 2.30}$ | $19.76 \pm 3.86$ |
| | 10 | $27.88 \pm 2.80$ | $11.86 \pm 4.70$ | $25.99 \pm 2.14$ | $\mathbf{34.24 \pm 1.31}$ | $26.43 \pm 2.22$ |
| **Test-Time Gradients?** | — | **No** | **No** | **No** | **Yes (5 steps)** | **No (0 gradients)** |
| **Parameters** | — | $\approx 115\text{k}$ | $\approx 225\text{k}$ | $\approx 115\text{k}$ | $\approx 115\text{k}$ | **$\approx 118\text{k}$** |

---

## 📑 Citation

If you find this work or codebase helpful in your research, please cite our paper:

```bibtex
@inproceedings{anonymous2027metalearn_nca,
  title     = {MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2027},
  note      = {Under review}
}
```

---

## 📜 License
This project is open-sourced under the [MIT License](LICENSE).
