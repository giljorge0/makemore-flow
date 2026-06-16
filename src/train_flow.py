"""
Flow matching on the latent "name space".

We have a dataset of points z_1 = encoder(name) in R^2 (normalized to mean 0,
std 1). We define a simple linear probability path between a noise sample
z_0 ~ N(0, I) and a data sample z_1:

    z_t = (1 - t) * z_0 + t * z_1,   t in [0, 1]

The "ground truth" velocity along this path is constant: dz_t/dt = z_1 - z_0.
We train a small MLP v_theta(z_t, t) to regress onto this velocity
(Lipman et al., "Flow Matching for Generative Modeling", 2022 -- the
linear / optimal-transport conditional path special case).

At sampling time we start from pure noise z_0 ~ N(0, I) and numerically
integrate dz/dt = v_theta(z, t) from t=0 to t=1 with an ODE solver
(simple Euler steps here). The trajectory IS the "evolving path through
concept space" from the original idea -- and because the space is 2D,
we can literally plot it.
"""

import torch
import torch.nn as nn
import time

LATENT_DIM = 2


class VelocityField(nn.Module):
    """v_theta(z, t) -> velocity in R^latent_dim"""

    def __init__(self, latent_dim=LATENT_DIM, hidden=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(latent_dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        # z: (B, latent_dim), t: (B, 1)
        return self.net(torch.cat([z, t], dim=-1))


@torch.no_grad()
def sample_ode(v_field, n_samples, n_steps=100, device="cpu", return_traj=False):
    """Integrate dz/dt = v_theta(z,t) from t=0 (noise) to t=1 (data) via Euler."""
    z = torch.randn(n_samples, LATENT_DIM, device=device)
    dt = 1.0 / n_steps
    traj = [z.clone()] if return_traj else None
    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((n_samples, 1), t_val, device=device)
        v = v_field(z, t)
        z = z + v * dt
        if return_traj:
            traj.append(z.clone())
    if return_traj:
        return z, torch.stack(traj, dim=0)  # (n_steps+1, n_samples, latent_dim)
    return z


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load("checkpoints/latents.pt")
    z1_all = data["z"].to(DEVICE)
    mean, std = data["mean"].to(DEVICE), data["std"].to(DEVICE)

    # normalize the data latents to roughly N(0, I) -- this is the
    # distribution our flow will learn to map noise *to*.
    z1_norm = (z1_all - mean) / std

    v_field = VelocityField().to(DEVICE)
    opt = torch.optim.Adam(v_field.parameters(), lr=1e-3)

    N = z1_norm.shape[0]
    BATCH = 1024
    EPOCHS = 60

    for epoch in range(EPOCHS):
        t0 = time.time()
        perm = torch.randperm(N, device=DEVICE)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, N, BATCH):
            idx = perm[i : i + BATCH]
            z1 = z1_norm[idx]
            B = z1.shape[0]
            z0 = torch.randn_like(z1)
            t = torch.rand(B, 1, device=DEVICE)
            zt = (1 - t) * z0 + t * z1
            target_v = z1 - z0

            pred_v = v_field(zt, t)
            loss = ((pred_v - target_v) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  loss {total_loss/n_batches:.4f}  ({time.time()-t0:.2f}s)")

    torch.save(v_field.state_dict(), "checkpoints/flow.pt")
    print("saved checkpoints/flow.pt")

    # quick sanity check: sample and see where points land vs real data
    samples = sample_ode(v_field, n_samples=2000, n_steps=100, device=DEVICE)
    print("sample mean/std (should be ~0/~1):", samples.mean(0).cpu().numpy(), samples.std(0).cpu().numpy())
    print("data   mean/std (should be ~0/~1):", z1_norm.mean(0).cpu().numpy(), z1_norm.std(0).cpu().numpy())


if __name__ == "__main__":
    main()
