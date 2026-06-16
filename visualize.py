"""
Visualizations that tie this back to the original idea:

  fig1_field_and_trajectories.png
    - scatter of every real name's position in the (normalized) 2D latent
      space, colored by name length
    - a quiver plot of the learned velocity field v_theta(z, t=0.5)
    - several full noise->data trajectories produced by the flow, with the
      generated name labeled at the endpoint
    -> this is literally "sampling from a neighborhood / following a path
       that evolves toward a concept"

  fig2_interpolation.png
    - take two real names, encode them to z_A and z_B
    - walk the straight line z(s) = (1-s) z_A + s z_B for s in [0,1]
    - decode each point on the line
    -> the classic "follow a line between two concepts" word2vec-style walk,
       but here applied to the same continuous name-space
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import AutoEncoder, encode, decode
from train_flow import VelocityField, sample_ode, LATENT_DIM

DEVICE = "cpu"


def load_everything():
    data = torch.load("checkpoints/latents.pt")
    mean, std = data["mean"], data["std"]
    z_all = data["z"]
    names = data["names"]

    ae = AutoEncoder(latent_dim=LATENT_DIM)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=DEVICE))
    ae.eval()

    v_field = VelocityField()
    v_field.load_state_dict(torch.load("checkpoints/flow.pt", map_location=DEVICE))
    v_field.eval()

    return ae, v_field, z_all, mean, std, names


def fig1_field_and_trajectories(ae, v_field, z_all, mean, std, names):
    z_norm = (z_all - mean) / std

    fig, ax = plt.subplots(figsize=(9, 9))

    # 1. scatter all real names, colored by length
    lengths = np.array([len(n) for n in names])
    sc = ax.scatter(
        z_norm[:, 0], z_norm[:, 1], c=lengths, cmap="viridis",
        s=4, alpha=0.35, linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("name length")

    # 2. vector field at t=0.5 on a grid
    lo, hi = -3.5, 3.5
    grid = np.linspace(lo, hi, 18)
    gx, gy = np.meshgrid(grid, grid)
    pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)
    t = torch.full((pts.shape[0], 1), 0.5)
    with torch.no_grad():
        v = v_field(pts, t).numpy()
    ax.quiver(
        gx.ravel(), gy.ravel(), v[:, 0], v[:, 1],
        color="gray", alpha=0.5, angles="xy", scale_units="xy", scale=3.0, width=0.0025,
    )

    # 3. sample full trajectories noise -> data, decode endpoints
    n_traj = 8
    torch.manual_seed(42)
    with torch.no_grad():
        z1_norm, traj = sample_ode(v_field, n_samples=n_traj, n_steps=100, return_traj=True)
        z1 = z1_norm * std + mean
        toks = ae.decoder.sample(z1, greedy=True)

    colors = plt.cm.tab10(np.linspace(0, 1, n_traj))
    for i in range(n_traj):
        path = traj[:, i, :].numpy()
        ax.plot(path[:, 0], path[:, 1], color=colors[i], lw=1.5, alpha=0.9)
        ax.scatter(path[0, 0], path[0, 1], color=colors[i], marker="x", s=60)  # noise start
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[i], marker="o", s=60, edgecolor="k")
        name = decode(toks[i].tolist())
        ax.annotate(name, (path[-1, 0], path[-1, 1]), textcoords="offset points",
                     xytext=(6, 6), fontsize=10, weight="bold", color=colors[i])

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(
        "Name-space: real names (dots), learned velocity field at t=0.5 (arrows),\n"
        "and 8 noise->name trajectories (x = noise start, o = decoded name)"
    )
    ax.set_xlabel("z[0] (normalized)")
    ax.set_ylabel("z[1] (normalized)")
    fig.tight_layout()
    fig.savefig("fig1_field_and_trajectories.png", dpi=150)
    print("saved fig1_field_and_trajectories.png")


def fig2_interpolation(ae, name_a="sophia", name_b="william", n_steps=9):
    with torch.no_grad():
        za = ae.encoder(
            torch.tensor([encode(name_a)]), torch.tensor([len(encode(name_a))])
        )[0]
        zb = ae.encoder(
            torch.tensor([encode(name_b)]), torch.tensor([len(encode(name_b))])
        )[0]

    ss = np.linspace(0, 1, n_steps)
    points = []
    decoded = []
    with torch.no_grad():
        for s in ss:
            z = (1 - s) * za + s * zb
            points.append(z.numpy())
            toks = ae.decoder.sample(z.unsqueeze(0), greedy=True)
            decoded.append(decode(toks[0].tolist()))

    points = np.array(points)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(points[:, 0], points[:, 1], "o-", color="C0")
    for (x, y), s, label in zip(points, ss, decoded):
        ax.annotate(f"{label}\n(s={s:.2f})", (x, y), textcoords="offset points",
                     xytext=(8, 4), fontsize=9)
    ax.scatter(*points[0], color="green", s=100, zorder=5, label=name_a)
    ax.scatter(*points[-1], color="red", s=100, zorder=5, label=name_b)
    ax.legend()
    ax.set_title(f'Linear walk in latent space: "{name_a}" -> "{name_b}"')
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    fig.tight_layout()
    fig.savefig("fig2_interpolation.png", dpi=150)
    print("saved fig2_interpolation.png")

    print(f"\nInterpolation {name_a} -> {name_b}:")
    for s, label in zip(ss, decoded):
        print(f"  s={s:.2f}: {label}")


if __name__ == "__main__":
    ae, v_field, z_all, mean, std, names = load_everything()
    fig1_field_and_trajectories(ae, v_field, z_all, mean, std, names)
    fig2_interpolation(ae, "sophia", "william")
