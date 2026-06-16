"""
Visualisations for the 8D flow model.
All plots project the 8D latent space to 2D via PCA so we can still see it.

fig6_pca_space.png
  - PCA scatter of all 32k real names, coloured by length bucket.
  - Per-condition velocity fields (evaluated in 8D, projected to 2D).
  - 8 guided generation trajectories with decoded name labels.

fig7_guidance_8d.png
  - Guidance scale vs avg generated length for all three conditions.
  - Compared against the 2D version to show the improvement.

fig8_compare_2d_vs_8d.png
  - Side-by-side: 20 generated names from 2D vs 8D for each condition,
    with avg length annotated, making the quality jump obvious at a glance.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

from model import AutoEncoder, decode
from flow_8d import VelocityField8D, sample_ode_8d, LATENT_DIM as LATENT_8
from train_flow_cond import (
    CondVelocityField, sample_ode_cfg,
    length_to_cond, NULL_TOKEN, COND_NAMES,
)

DEVICE = "cpu"
COND_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF5722"}
COND_LABELS = {1: "short (≤4)", 2: "medium (5–6)", 3: "long (≥7)"}
DEFAULT_W   = {1: 2.0, 2: 2.0, 3: 2.0}


def load_8d():
    data = torch.load("checkpoints/latents_8d.pt", weights_only=False)
    mean, std = data["mean"], data["std"]
    z_all, names = data["z"], data["names"]

    ae = AutoEncoder(latent_dim=LATENT_8)
    ae.load_state_dict(torch.load("checkpoints/autoencoder_8d.pt", weights_only=False)["state_dict"])
    ae.eval()

    v = VelocityField8D()
    v.load_state_dict(torch.load("checkpoints/flow_cond_8d.pt", weights_only=False)["state_dict"])
    v.eval()
    return ae, v, z_all, mean, std, names


def load_2d():
    data = torch.load("checkpoints/latents.pt", weights_only=False)
    mean, std = data["mean"], data["std"]
    v = CondVelocityField()
    v.load_state_dict(torch.load("checkpoints/flow_cond.pt", weights_only=False))
    v.eval()
    ae = AutoEncoder(latent_dim=2)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", weights_only=False))
    ae.eval()
    return ae, v, mean, std


# ── fig6: PCA scatter + per-condition fields + trajectories ──────────────

def fig6_pca_space(ae, v_field, z_all, mean, std, names):
    z_norm = ((z_all - mean) / std).numpy()
    conds  = np.array([length_to_cond(len(n)) for n in names])

    pca = PCA(n_components=2)
    pca.fit(z_norm)
    z2   = pca.transform(z_norm)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    var = pca.explained_variance_ratio_

    for col, (ct, label) in enumerate(COND_LABELS.items()):
        ax    = axes[col]
        color = COND_COLORS[ct]

        # background scatter
        for other_ct in [1, 2, 3]:
            mask  = conds == other_ct
            alpha = 0.06 if other_ct != ct else 0.25
            sz    = 2    if other_ct != ct else 3
            ax.scatter(z2[mask, 0], z2[mask, 1],
                       c=COND_COLORS[other_ct], s=sz, alpha=alpha, linewidths=0)

        # vector field: evaluate in 8D, project arrows to 2D via PCA
        lo, hi = z2[:, 0].min() - 0.3, z2[:, 0].max() + 0.3
        lo1,hi1= z2[:, 1].min() - 0.3, z2[:, 1].max() + 0.3
        g1 = np.linspace(lo, hi, 14)
        g2 = np.linspace(lo1, hi1, 14)
        gx, gy = np.meshgrid(g1, g2)
        grid2d = np.stack([gx.ravel(), gy.ravel()], axis=1)
        # inverse-project 2D grid -> 8D via PCA
        grid8d = pca.inverse_transform(grid2d)
        pts8   = torch.tensor(grid8d, dtype=torch.float32)
        t_mid  = torch.full((pts8.shape[0], 1), 0.5)
        cond_t = torch.full((pts8.shape[0],), ct, dtype=torch.long)
        with torch.no_grad():
            v8 = v_field(pts8, t_mid, cond_t).numpy()
        v2 = pca.transform(v8)                    # project velocity to 2D
        ax.quiver(gx.ravel(), gy.ravel(), v2[:, 0], v2[:, 1],
                  color=color, alpha=0.5, angles="xy", scale_units="xy",
                  scale=20.0, width=0.003)

        # 5 guided trajectories, all starting from the same noise
        N_TRAJ = 5
        torch.manual_seed(7)
        z0_8d = torch.randn(N_TRAJ, LATENT_8)
        z = z0_8d.clone(); dt = 1.0 / 100
        ct_tok = torch.full((N_TRAJ,), ct, dtype=torch.long)
        uc_tok = torch.full((N_TRAJ,), NULL_TOKEN, dtype=torch.long)
        w = DEFAULT_W[ct]
        traj = [pca.transform(z.numpy())]
        with torch.no_grad():
            for step in range(100):
                t_s = torch.full((N_TRAJ, 1), step * dt)
                vc  = v_field(z, t_s, ct_tok)
                vu  = v_field(z, t_s, uc_tok)
                z   = z + (vu + w * (vc - vu)) * dt
                traj.append(pca.transform(z.numpy()))
        traj = np.stack(traj)               # (101, N_TRAJ, 2)
        z1   = z * std + mean
        with torch.no_grad():
            toks = ae.decoder.sample(z1, greedy=True)

        clrs = plt.cm.cool(np.linspace(0.1, 0.9, N_TRAJ))
        for i in range(N_TRAJ):
            path = traj[:, i, :]
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.4, alpha=0.85)
            ax.scatter(*path[0],  color="black", marker="x", s=60, zorder=5)
            ax.scatter(*path[-1], color=color, marker="o", s=55, edgecolor="k", lw=0.5, zorder=5)
            nm = decode(toks[i].tolist())
            ax.annotate(nm, path[-1], textcoords="offset points",
                        xytext=(5, 3), fontsize=8.5, color=color, weight="bold")

        ax.set_title(f"Condition: {label}", fontsize=11, color=color, weight="bold")
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
        if col == 0:
            ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")

    patches = [mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS[c]) for c in [1,2,3]]
    fig.legend(handles=patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("8D name-space (PCA projected): real names, learned velocity field, "
                 "and guided ODE trajectories", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("fig6_pca_space.png", dpi=150, bbox_inches="tight")
    print("saved fig6_pca_space.png")


# ── fig7: guidance scale vs length for 8D ───────────────────────────────

def fig7_guidance_8d(ae, v_field, mean, std, names):
    from train_flow_cond import length_to_cond
    conds_np = np.array([length_to_cond(len(n)) for n in names])
    true_means = {ct: np.mean([len(n) for n, c in zip(names, conds_np) if c == ct])
                  for ct in [1, 2, 3]}

    w_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    N = 300
    fig, ax = plt.subplots(figsize=(9, 5))

    for ct, label in COND_LABELS.items():
        color = COND_COLORS[ct]
        avgs  = []
        for w in w_vals:
            torch.manual_seed(0)
            z_norm = sample_ode_8d(v_field, N, ct, guidance_scale=w, n_steps=80)
            z = z_norm * std + mean
            with torch.no_grad():
                toks = ae.decoder.sample(z, greedy=False, temperature=0.9)
            dec  = [decode(t.tolist()) for t in toks if decode(t.tolist())]
            avgs.append(np.mean([len(d) for d in dec]) if dec else 0)
        ax.plot(w_vals, avgs, "o-", color=color, label=label, lw=2)
        ax.axhline(true_means[ct], color=color, ls="--", alpha=0.45, lw=1)

    ax.set_xlabel("Guidance scale  w", fontsize=12)
    ax.set_ylabel("Mean generated name length", fontsize=12)
    ax.set_title("8D model: guidance strength vs generated length\n"
                 "(dashed = training-set mean per bucket)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig7_guidance_8d.png", dpi=150)
    print("saved fig7_guidance_8d.png")


# ── fig8: 2D vs 8D side-by-side quality comparison ──────────────────────

def fig8_compare_2d_vs_8d(ae_2d, v_2d, mean_2d, std_2d,
                            ae_8d, v_8d, mean_8d, std_8d):
    N = 18
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    DEFAULT_W_2D = {1: 2.5, 2: 3.0, 3: 4.0}

    for row, (ct, label) in enumerate(COND_LABELS.items()):
        color = COND_COLORS[ct]

        # 2D
        torch.manual_seed(0)
        z2_norm = sample_ode_cfg(v_2d, N, ct, guidance_scale=DEFAULT_W_2D[ct])
        z2      = z2_norm * std_2d + mean_2d
        with torch.no_grad():
            toks2 = ae_2d.decoder.sample(z2, greedy=False, temperature=0.9)
        dec2  = [decode(t.tolist()) for t in toks2]
        avg2  = np.mean([len(d) for d in dec2 if d])

        # 8D
        torch.manual_seed(0)
        DEFAULT_W_8D = 2.0
        z8_norm = sample_ode_8d(v_8d, N, ct, guidance_scale=DEFAULT_W_8D)
        z8      = z8_norm * std_8d + mean_8d
        with torch.no_grad():
            toks8 = ae_8d.decoder.sample(z8, greedy=False, temperature=0.9)
        dec8  = [decode(t.tolist()) for t in toks8]
        avg8  = np.mean([len(d) for d in dec8 if d])

        for col, (dec, avg, model_label, w) in enumerate([
            (dec2, avg2, "2D", DEFAULT_W_2D[ct]),
            (dec8, avg8, "8D", 2.0),
        ]):
            ax = axes[row][col]
            ax.set_facecolor("#f4f4f4")
            text = "\n".join(d if d else "(empty)" for d in dec[:N])
            ax.text(0.05, 0.97, text, transform=ax.transAxes,
                    fontsize=10, verticalalignment="top",
                    fontfamily="monospace", color="#111111")
            title_col = color if col == 1 else "#aaaaaa"
            ax.set_title(
                f"{model_label} model  |  {label}  |  w={w}  |  avg_len={avg:.1f}",
                color=title_col, fontsize=10, weight="bold", pad=6
            )
            ax.axis("off")

    fig.suptitle("2D vs 8D conditioned generation — same noise seed, same condition",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig("fig8_compare_2d_vs_8d.png", dpi=150)
    print("saved fig8_compare_2d_vs_8d.png")


if __name__ == "__main__":
    # install sklearn if needed
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "scikit-learn", "--break-system-packages", "-q"])
        from sklearn.decomposition import PCA

    ae8, v8, z_all, mean8, std8, names = load_8d()
    ae2, v2, mean2, std2               = load_2d()

    fig6_pca_space(ae8, v8, z_all, mean8, std8, names)
    fig7_guidance_8d(ae8, v8, mean8, std8, names)
    fig8_compare_2d_vs_8d(ae2, v2, mean2, std2, ae8, v8, mean8, std8)
