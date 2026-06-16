"""
Visualizations for the conditioned flow model.

fig3_cond_regions.png
  - Scatter of all real names in normalised 2D latent space, each point
    coloured by its length bucket (short/medium/long).
  - Velocity field at t=0.5 for each condition (3 quiver subplots) so you
    can see how CFG tilts the field differently per condition.

fig4_guidance_strength.png
  - For each condition, show the average generated name length as a
    function of guidance scale w ∈ [0, 1, 2, 3, 4, 5, 6].
  - Horizontal dashed lines mark the true training-set mean lengths.
  -> Makes the "steering the evolution" effect quantitative and visible.

fig5_cond_trajectories.png
  - 5 trajectories per condition (same starting noise), coloured by
    condition, showing how the ODE paths diverge to different regions
    of the space depending on the guidance condition.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model import AutoEncoder, decode
from train_flow_cond import (
    CondVelocityField, sample_ode_cfg, LATENT_DIM, COND_NAMES,
    length_to_cond, NULL_TOKEN,
)

DEVICE = "cpu"
COND_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF5722"}   # blue / green / orange
COND_LABELS = {1: "short (≤4)", 2: "medium (5–6)", 3: "long (≥7)"}
DEFAULT_GUIDANCE = {1: 2.5, 2: 3.0, 3: 4.0}


def load():
    data = torch.load("checkpoints/latents.pt", weights_only=False)
    mean, std = data["mean"], data["std"]
    z_all = data["z"]
    names = data["names"]

    ae = AutoEncoder(latent_dim=LATENT_DIM)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=DEVICE, weights_only=False))
    ae.eval()

    v = CondVelocityField()
    v.load_state_dict(torch.load("checkpoints/flow_cond.pt", map_location=DEVICE, weights_only=False))
    v.eval()

    return ae, v, z_all, mean, std, names


# ── fig3: latent space coloured by condition + per-condition vector fields ──

def fig3_cond_regions(v_field, z_all, mean, std, names):
    z_norm = (z_all - mean) / std
    conds  = np.array([length_to_cond(len(n)) for n in names])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    lo, hi = -3.2, 3.2
    grid = np.linspace(lo, hi, 16)
    gx, gy = np.meshgrid(grid, grid)
    pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)
    t_mid = torch.full((pts.shape[0], 1), 0.5)

    for col, (ct, label) in enumerate(COND_LABELS.items()):
        ax = axes[col]
        color = COND_COLORS[ct]

        # dim background: all other conditions
        for other_ct in [1, 2, 3]:
            mask = conds == other_ct
            alpha = 0.08 if other_ct != ct else 0.30
            sz    = 2    if other_ct != ct else 3
            ax.scatter(z_norm[mask, 0], z_norm[mask, 1],
                       c=COND_COLORS[other_ct], s=sz, alpha=alpha, linewidths=0)

        # vector field for THIS condition (w=1 so it's pure cond field)
        cond_tok = torch.full((pts.shape[0],), ct, dtype=torch.long)
        with torch.no_grad():
            v_vec = v_field(pts, t_mid, cond_tok).numpy()
        ax.quiver(gx.ravel(), gy.ravel(), v_vec[:, 0], v_vec[:, 1],
                  color=color, alpha=0.55, angles="xy", scale_units="xy",
                  scale=3.5, width=0.003)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"Condition: {label}", fontsize=12, color=color, weight="bold")
        ax.set_xlabel("z[0] (normalised)")
        if col == 0:
            ax.set_ylabel("z[1] (normalised)")

    patches = [mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS[c]) for c in [1,2,3]]
    fig.legend(handles=patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Name-space coloured by length bucket + per-condition velocity field at t=0.5",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("fig3_cond_regions.png", dpi=150, bbox_inches="tight")
    print("saved fig3_cond_regions.png")


# ── fig4: guidance scale vs generated length ───────────────────────────────

def fig4_guidance_strength(ae, v_field, mean, std, names):
    # true means from training set
    true_means = {}
    conds_np = np.array([length_to_cond(len(n)) for n in names])
    for ct in [1, 2, 3]:
        true_means[ct] = np.mean([len(n) for n, c in zip(names, conds_np) if c == ct])

    w_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    N_SAMPLES = 300

    fig, ax = plt.subplots(figsize=(9, 5))

    for ct, label in COND_LABELS.items():
        color = COND_COLORS[ct]
        avg_lens = []
        for w in w_vals:
            torch.manual_seed(0)
            z_norm = sample_ode_cfg(v_field, N_SAMPLES, ct, guidance_scale=w, n_steps=80)
            z = z_norm * std + mean
            with torch.no_grad():
                toks = ae.decoder.sample(z, greedy=False, temperature=0.9)
            decoded = [decode(t.tolist()) for t in toks]
            decoded = [d for d in decoded if d]
            avg_lens.append(np.mean([len(d) for d in decoded]) if decoded else 0)

        ax.plot(w_vals, avg_lens, "o-", color=color, label=label, lw=2)
        ax.axhline(true_means[ct], color=color, ls="--", alpha=0.5, lw=1)

    ax.set_xlabel("Guidance scale  w", fontsize=12)
    ax.set_ylabel("Mean generated name length", fontsize=12)
    ax.set_title("Effect of guidance strength on generated name length\n"
                 "(dashed = training-set mean for that bucket)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig4_guidance_strength.png", dpi=150)
    print("saved fig4_guidance_strength.png")


# ── fig5: diverging trajectories per condition ─────────────────────────────

def fig5_cond_trajectories(ae, v_field, mean, std):
    N_TRAJ = 5
    N_STEPS = 100
    lo, hi = -3.2, 3.2

    fig, ax = plt.subplots(figsize=(9, 9))

    # background: faint overall scatter
    data = torch.load("checkpoints/latents.pt", weights_only=False)
    z_all = (data["z"] - mean) / std
    ax.scatter(z_all[:, 0], z_all[:, 1], c="#cccccc", s=2, alpha=0.15, linewidths=0)

    torch.manual_seed(7)
    shared_z0 = torch.randn(N_TRAJ, LATENT_DIM)  # same starting noise for all conditions

    for ct, label in COND_LABELS.items():
        color = COND_COLORS[ct]
        cond_tok = torch.full((N_TRAJ,), ct, dtype=torch.long)
        uncond_tok = torch.full((N_TRAJ,), NULL_TOKEN, dtype=torch.long)
        w = DEFAULT_GUIDANCE[ct]

        z = shared_z0.clone()
        dt = 1.0 / N_STEPS
        traj = [z.clone()]

        for step in range(N_STEPS):
            t_val = step * dt
            t = torch.full((N_TRAJ, 1), t_val)
            with torch.no_grad():
                v_c = v_field(z, t, cond_tok)
                v_u = v_field(z, t, uncond_tok)
            v_guided = v_u + w * (v_c - v_u)
            z = z + v_guided * dt
            traj.append(z.clone())

        traj_np = torch.stack(traj).numpy()  # (steps+1, N_TRAJ, 2)

        z1 = z * std + mean
        with torch.no_grad():
            toks = ae.decoder.sample(z1, greedy=True)

        for i in range(N_TRAJ):
            path = traj_np[:, i, :]
            ax.plot(path[:, 0], path[:, 1], color=color, lw=1.5, alpha=0.85)
            if i == 0:  # mark shared start once per trajectory set
                ax.scatter(path[0, 0], path[0, 1], color="black", marker="x", s=80, zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], color=color, marker="o", s=60,
                       edgecolor="k", lw=0.5, zorder=5)
            name = decode(toks[i].tolist())
            ax.annotate(name, (path[-1, 0], path[-1, 1]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=9, color=color, weight="bold")

    patches = [mpatches.Patch(color=COND_COLORS[c], label=f"{COND_LABELS[c]}  (w={DEFAULT_GUIDANCE[c]})")
               for c in [1, 2, 3]]
    ax.legend(handles=patches, fontsize=10, loc="upper left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Same noise start (×), different condition → diverging ODE trajectories\n"
                 "Condition steers which region of name-space each path settles in",
                 fontsize=12)
    ax.set_xlabel("z[0] (normalised)")
    ax.set_ylabel("z[1] (normalised)")
    fig.tight_layout()
    fig.savefig("fig5_cond_trajectories.png", dpi=150)
    print("saved fig5_cond_trajectories.png")


if __name__ == "__main__":
    ae, v_field, z_all, mean, std, names = load()
    fig3_cond_regions(v_field, z_all, mean, std, names)
    fig4_guidance_strength(ae, v_field, mean, std, names)
    fig5_cond_trajectories(ae, v_field, mean, std)
