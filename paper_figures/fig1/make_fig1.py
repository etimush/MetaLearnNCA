"""Figure 1: the MetaLearnNCA interaction loop, drawn with real tensors from a released checkpoint.

Usage (from the repository root):
  python paper_figures/fig1/run_loop.py 0 43 paper_figures/fig1/loop_s0.npz   # checkpoint seed, eval seed
  python paper_figures/fig1/pick_query.py <loop.npz> <loop_q.npz> 0 43 omniglot          # optional
  python paper_figures/fig1/make_fig1.py paper_figures/fig1/loop_s0.npz paper_figures/fig1/fig1_loop [5,23]
The optional last argument lists the program channels shown (default: 5,23).
Writes <out>.pdf (vector, fonts embedded, for the paper) and <out>.png (preview).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

z = np.load(sys.argv[1])
out = sys.argv[2]
# program channels shown (default: the channels discussed in Section 6.6)
CH = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5, 23]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif"], "mathtext.fontset": "stix",
    "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
})
W, H = 5.5, 2.45
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

BLUE, BLUE_F = "#3b6ea5", "#dfe9f5"
ORANGE, ORANGE_F = "#c8692c", "#fbe6d8"
GREY = "#444444"
PANEL = "#f6f6f6"
TAB = plt.get_cmap("tab10").colors

# ---------------- data -> images ----------------
S, P, E, sx, sy, m = z["S"], z["P"], z["E"], z["sx"], z["sy"], z["mask"]
X = S[-1].reshape(24, -1).T
mu = X.mean(0)
_, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
lo, hi = np.percentile((X - mu) @ Vt[:3].T, [2, 98], axis=0)


def prog_rgb(Sk):
    Z = (Sk.reshape(24, -1).T - mu) @ Vt[:3].T
    return np.clip((Z - lo) / (hi - lo), 0, 1).reshape(28, 28, 3)


def chan_card(x, y, Sk, c, size, lw=0.5, z=3):
    """one program channel, diverging map centred on its own mean and scaled to its own range
    (all channels start from the same U(0,1) noise, so a shared scale would hide early structure)"""
    m = Sk[c]
    ctr = m.mean()
    v = max(np.percentile(np.abs(m - ctr), 99), 1e-6)
    card(x, y, m, size, cmap="RdBu_r", vmin=ctr - v, vmax=ctr + v, lw=lw, z=z)


def votes_rgb(logits, mask):
    """per-pixel argmax class (tab10 colours) on the strokes, white background"""
    cls = logits.argmax(0)
    img = np.ones((28, 28, 3))
    col = np.array(TAB)[cls]
    img[mask > 0] = col[mask > 0]
    return img


# ---------------- drawing helpers ----------------
def card(x, y, img, s, cmap=None, vmin=None, vmax=None, lw=0.5, z=3):
    ax.imshow(img, extent=[x, x + s, y, y + s], cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest", zorder=z)
    ax.add_patch(Rectangle((x, y), s, s, fill=False, lw=lw, ec="#777777", zorder=z + 0.1))


def stack(x, y, imgs, s, **kw):
    off = 0.045
    for i in reversed(range(len(imgs))):
        card(x + off * i, y + off * i, imgs[i], s, z=3 + (len(imgs) - i) * 0.2, **kw)
    return x + s + off * (len(imgs) - 1), y + s + off * (len(imgs) - 1)


def box(x, y, w, h, fc, ec, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.9, zorder=2))
    # 3x3 neighbourhood icon
    g = 0.055
    gx, gy = x + 0.07, y + h / 2 - 1.5 * g
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((gx + i * g, gy + j * g), g, g, lw=0.4, ec=ec,
                                   fc=ec if (i, j) == (1, 1) else "white", zorder=3))
    tx = gx + 3 * g + 0.06
    ax.text(tx, y + h / 2 + 0.07, title, ha="left", va="center", fontsize=7.0, zorder=4)
    ax.text(tx, y + h / 2 - 0.09, sub, ha="left", va="center", fontsize=6.8, color="#333333", zorder=4)


def arrow(p0, p1, color=GREY, lw=0.8, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6.5, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=z))


def path_arrow(pts, color=GREY, lw=0.8, z=5):
    xs, ys = zip(*pts[:-1])
    ax.plot(list(xs) + [pts[-1][0]], list(ys) + [pts[-1][1]], color=color, lw=lw, zorder=z,
            solid_capstyle="butt")
    arrow(pts[-2], pts[-1], color=color, lw=lw, z=z)


def label(x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "top"); kw.setdefault("fontsize", 7)
    ax.text(x, y, s, zorder=6, **kw)


# ---------------- panels ----------------
SPLIT = 3.08
ax.add_patch(FancyBboxPatch((0.03, 1.12), W - 0.06, 1.30, boxstyle="round,pad=0,rounding_size=0.05",
                            fc=PANEL, ec="none", zorder=0))
ax.add_patch(FancyBboxPatch((0.03, 0.03), SPLIT - 0.06, 1.03, boxstyle="round,pad=0,rounding_size=0.05",
                            fc=PANEL, ec="none", zorder=0))
ax.add_patch(FancyBboxPatch((SPLIT + 0.03, 0.03), W - SPLIT - 0.06, 1.03,
                            boxstyle="round,pad=0,rounding_size=0.05", fc=PANEL, ec="none", zorder=0))
ax.text(0.09, 2.37, r"(a) Support adaptation: $K$ rounds, no gradients", fontsize=7.4, fontweight="bold",
        ha="left", va="top")
ax.text(0.09, 1.01, "(b) Query inference", fontsize=7.4, fontweight="bold", ha="left", va="top")
ax.text(SPLIT + 0.09, 1.01, r"(c) Adaptation over rounds $k$", fontsize=7.4, fontweight="bold",
        ha="left", va="top")

# ---------------- (a) adaptation loop ----------------
s = 0.40
yA = 1.36                      # bottom of thumbnails in row (a)
ids = [2, 0, 5]                # support images shown (front first)
k_show = 0                     # round shown inside the loop (k = 1)
xs_r, _ = stack(0.12, yA, [sx[i, 0] for i in ids], s, cmap="gray_r", vmin=-1, vmax=1)
label(0.12 + s / 2 + 0.045, yA - 0.04, r"$\mathbf{X}_{\mathrm{supp}}$")

bw, bh = 0.88, 0.46
by = yA + s / 2 + 0.045 - bh / 2
cyA = by + bh / 2
bx = xs_r + 0.17
arrow((xs_r + 0.03, cyA), (bx, cyA))
box(bx, by, bw, bh, BLUE_F, BLUE, r"Active-NCA $f_\theta$", r"$2N$ steps")

x1 = bx + bw + 0.18
arrow((bx + bw, cyA), (x1 - 0.02, cyA))
xr, ytop_stack = stack(x1, yA, [votes_rgb(P[k_show][i], m[i, 0]) for i in ids], s)
label(x1 + s / 2 + 0.045, yA - 0.04, r"$\hat{\mathbf{Y}}$: pixel votes")

cx, r = xr + 0.20, 0.085
arrow((xr + 0.03, cyA), (cx - r, cyA))
ax.add_patch(plt.Circle((cx, cyA), r, fc="white", ec=GREY, lw=0.8, zorder=5))
ax.text(cx, cyA - 0.005, r"$-$", ha="center", va="center", fontsize=8, zorder=6)
ax.text(cx, ytop_stack + 0.04,
        r"$\mathbf{E}_{\mathrm{pixel}} = (\mathbf{Y}-\mathrm{softmax}(\hat{\mathbf{Y}}))\odot\mathbf{M}$",
        ha="center", va="bottom", fontsize=6.6, zorder=6)

x2 = cx + r + 0.17
arrow((cx + r, cyA), (x2 - 0.02, cyA))
xe, _ = stack(x2, yA, [E[k_show][i, sy[i]] for i in ids], s, cmap="Oranges", vmin=0, vmax=1)
label(x2 + s / 2 + 0.045, yA - 0.04, r"$\mathbf{E}_{\mathrm{pixel}}$")

mx = xe + 0.17
arrow((xe + 0.03, cyA), (mx, cyA))
box(mx, by, bw, bh, ORANGE_F, ORANGE, r"Meta-NCA $g_\phi$", r"$N$ steps")
ax.add_patch(FancyArrowPatch((mx + bw * 0.70, by), (mx + bw * 0.32, by),
                             connectionstyle="arc3,rad=-0.9", arrowstyle="-|>", mutation_scale=6,
                             lw=0.7, color=ORANGE, zorder=5))
label(mx + bw * 0.51, by - 0.19, r"state $\mathbf{D}_{\mathrm{meta}}$ kept across rounds", fontsize=6.3,
      color=ORANGE)

x3 = mx + bw + 0.46
arrow((mx + bw, cyA), (x3 - 0.02, cyA))
label((mx + bw + x3) / 2, cyA - 0.04, "mean over\nsupport", fontsize=6.0)
off = 0.045
for i in reversed(range(len(CH))):
    chan_card(x3 + off * i, yA + off * i, S[k_show + 1], CH[i], s, z=3 + (len(CH) - i) * 0.2)
xS_r, yS_t = x3 + s + off * (len(CH) - 1), yA + s + off * (len(CH) - 1)
xS_c = (x3 + xS_r) / 2
label(x3 + s / 2 + off, yA - 0.04, r"$\mathbf{S}^{(k)}$")

ytop = 2.17
path_arrow([(xS_c, yS_t), (xS_c, ytop), (bx + bw * 0.5, ytop), (bx + bw * 0.5, by + bh)],
           color=ORANGE, lw=0.9)
ax.text(xS_c - 0.06, ytop + 0.03,
        r"next round $k \leftarrow k+1$, repeated $K$ times; $\mathbf{S}^{(0)}, \mathbf{D}_{\mathrm{meta}}^{(0)} \sim \mathcal{U}(0,1)$",
        ha="right", va="bottom", fontsize=6.6, color=ORANGE, zorder=6)
ax.text(bx + bw * 0.5 + 0.04, ytop - 0.05, r"$\mathbf{S}^{(k-1)}$", ha="left", va="top", fontsize=6.8,
        zorder=6)

# ---------------- (b) query inference ----------------
yB = 0.31
card(0.12, yB, z["qx"][0, 0], s, cmap="gray_r", vmin=-1, vmax=1)
label(0.12 + s / 2, yB - 0.04, r"$\mathbf{X}_{\mathrm{qry}}$")
qbx, qby = 0.12 + s + 0.15, yB + s / 2 - bh / 2
arrow((0.12 + s + 0.03, qby + bh / 2), (qbx, qby + bh / 2))
box(qbx, qby, bw, bh, BLUE_F, BLUE, r"Active-NCA $f_\theta$", r"$2N$ steps")
yfz = 1.088
xfz = xS_r + 0.08
path_arrow([(xS_r, cyA), (xfz, cyA), (xfz, yfz), (qbx + bw * 0.5, yfz),
            (qbx + bw * 0.5, qby + bh)], color=BLUE, lw=0.9)
ax.text(qbx + bw * 0.5 + 0.30, yfz - 0.03, r"$\mathbf{S}^{(K)}$, frozen", ha="left", va="top",
        fontsize=6.8, color=BLUE, zorder=6)

xq = qbx + bw + 0.17
arrow((qbx + bw, qby + bh / 2), (xq - 0.02, qby + bh / 2))
card(xq, yB, votes_rgb(z["qlog"][0], z["qmask"][0, 0]), s)
label(xq + s / 2, yB - 0.04, "pixel votes")
sc = z["qscore"][0]
pr = np.exp(sc - sc.max()); pr /= pr.sum()
bax_x = xq + s + 0.30
bax_w = SPLIT - 0.10 - bax_x
arrow((xq + s + 0.03, yB + s / 2), (bax_x - 0.05, yB + s / 2))
label((xq + s + bax_x) / 2 - 0.01, yB + s / 2 - 0.04, "masked\nmean", fontsize=6.0)
bax = fig.add_axes([bax_x / W, (yB + 0.02) / H, bax_w / W, (s + 0.02) / H])
# bar order: task classes 0..9, or the dataset's class ids in increasing order when given (--digits);
# each bar keeps the colour of its task class, as in the pixel-vote maps
if "tick_labels" in z:
    names = [str(t) for t in z["tick_labels"]]
    order = sorted(range(10), key=lambda c: int(names[c]) if names[c].isdigit() else names[c])
else:
    names, order = [str(c) for c in range(10)], list(range(10))
bax.bar(range(10), pr[order], color=[TAB[c] for c in order], width=0.75)
bax.set_xticks(range(10)); bax.set_xticklabels([names[c] for c in order], fontsize=5.3)
bax.tick_params(axis="x", length=0, pad=1); bax.set_yticks([])
for sp in ["top", "right", "left"]:
    bax.spines[sp].set_visible(False)
bax.spines["bottom"].set_linewidth(0.5)
bax.set_facecolor(PANEL); bax.patch.set_alpha(0)
bax.set_ylim(0, 1.05); bax.set_xlim(-0.6, 9.6)
yhat = int(np.argmax(pr))
xh = order.index(yhat)
bax.annotate(r"$\hat{y}$", (xh + 0.45, pr[yhat] * 0.97), xytext=(xh + 1.6, pr[yhat] * 0.97), fontsize=7,
             va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
label(bax_x + bax_w / 2, yB - 0.12, "class scores", fontsize=6.3)

# ---------------- (c) over the rounds ----------------
acc = z["acc"]
ts, gap = 0.195, 0.085
x0 = W - 0.10 - (6 * ts + 5 * gap)
ch_rows = [0.59, 0.375][:len(CH)]
y_votes = 0.15
for c, yr in zip(CH, ch_rows):
    ax.text(x0 - 0.07, yr + ts / 2, rf"$\mathbf{{S}}^{{(k)}}_{{{c}}}$", ha="right", va="center", fontsize=7)
ax.text(x0 - 0.07, y_votes + ts / 2, "query\nvotes", ha="right", va="center", fontsize=6.2, linespacing=0.95)
ax.text(x0 - 0.07, 0.080, "acc.", ha="right", va="center", fontsize=6.4)
for k in range(6):
    xk = x0 + k * (ts + gap)
    for c, yr in zip(CH, ch_rows):
        chan_card(xk, yr, S[k], c, ts, lw=0.4)
    card(xk, y_votes, votes_rgb(z["qlog_k"][k], z["qmask"][0, 0]), ts, lw=0.4)
    ax.text(xk + ts / 2, ch_rows[0] + ts + 0.025, f"$k={k}$", ha="center", va="bottom", fontsize=6.5)
    ax.text(xk + ts / 2, 0.080, f"{acc[k]:.0f}%", ha="center", va="center", fontsize=6.5)

fig.savefig(out + ".pdf", dpi=600)   # dpi: resolution of the resampled 28x28 thumbnails
fig.savefig(out + ".png", dpi=300)
print("saved", out)
