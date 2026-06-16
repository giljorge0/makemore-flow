"""
Conditioned flow matching.

The velocity field v_theta(z, t, c) now takes a condition c alongside
(z, t).  We use two conditions:

  1. length_bucket  -- discretised name length (short/medium/long)
  2. null           -- a learned "unconditional" embedding (used for CFG)

Training: randomly drop the condition (replace with null token) 15% of
the time.  The velocity field learns both a conditioned and an
unconditional direction.

Sampling (Classifier-Free Guidance):
  v_guided = v_uncond + w * (v_cond - v_uncond)
  w > 1 steers more aggressively toward the conditioned distribution.
  w = 0 recovers the unconditional field.

Length buckets (feel free to tune):
  short  : len <= 4
  medium : 5 <= len <= 6
  long   : len >= 7

This is the "steer the evolution" piece: at sample time you say
"give me a 5-letter name" and the ODE integrator tilts every step
toward that region of the name-space.
"""

import torch
import torch.nn as nn
import time

LATENT_DIM = 2

# --- condition vocabulary ---
# 0 = NULL (unconditional), 1 = short, 2 = medium, 3 = long
NUM_CONDITIONS = 4
NULL_TOKEN = 0
COND_NAMES = {1: "short (≤4)", 2: "medium (5-6)", 3: "long (≥7)"}


def length_to_cond(length: int) -> int:
    if length <= 4:
        return 1
    elif length <= 6:
        return 2
    else:
        return 3


class CondVelocityField(nn.Module):
    """v_theta(z, t, c) -> velocity in R^latent_dim"""

    def __init__(self, latent_dim=LATENT_DIM, hidden=256, n_layers=5, cond_embed_dim=16):
        super().__init__()
        self.cond_embed = nn.Embedding(NUM_CONDITIONS, cond_embed_dim)
        in_dim = latent_dim + 1 + cond_embed_dim  # z | t | c
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t, cond):
        # z: (B, latent_dim), t: (B, 1), cond: (B,) long tensor
        c = self.cond_embed(cond)
        return self.net(torch.cat([z, t, c], dim=-1))


@torch.no_grad()
def sample_ode_cfg(
    v_field,
    n_samples,
    cond_token,           # int  – condition label (1/2/3)
    guidance_scale=3.0,   # w in v_guided = v_uncond + w*(v_cond - v_uncond)
    n_steps=100,
    device="cpu",
    return_traj=False,
):
    z = torch.randn(n_samples, LATENT_DIM, device=device)
    dt = 1.0 / n_steps
    cond  = torch.full((n_samples,), cond_token, dtype=torch.long, device=device)
    uncond = torch.full((n_samples,), NULL_TOKEN,  dtype=torch.long, device=device)
    traj = [z.clone()] if return_traj else None

    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((n_samples, 1), t_val, device=device)
        v_cond   = v_field(z, t, cond)
        v_uncond = v_field(z, t, uncond)
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        z = z + v * dt
        if return_traj:
            traj.append(z.clone())

    if return_traj:
        return z, torch.stack(traj, dim=0)
    return z


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DROP_PROB = 0.15   # probability of dropping condition during training

    data = torch.load("checkpoints/latents.pt")
    z1_all  = data["z"].to(DEVICE)
    names   = data["names"]
    mean    = data["mean"].to(DEVICE)
    std     = data["std"].to(DEVICE)

    # build condition tensor for every name
    conds_all = torch.tensor(
        [length_to_cond(len(n)) for n in names], dtype=torch.long, device=DEVICE
    )

    z1_norm = (z1_all - mean) / std

    v_field = CondVelocityField().to(DEVICE)
    opt = torch.optim.Adam(v_field.parameters(), lr=1e-3)

    N      = z1_norm.shape[0]
    BATCH  = 1024
    EPOCHS = 80

    for epoch in range(EPOCHS):
        t0 = time.time()
        perm = torch.randperm(N, device=DEVICE)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, N, BATCH):
            idx  = perm[i : i + BATCH]
            z1   = z1_norm[idx]
            cond = conds_all[idx].clone()
            B    = z1.shape[0]

            # condition dropout -> null token
            drop = torch.rand(B, device=DEVICE) < DROP_PROB
            cond[drop] = NULL_TOKEN

            z0 = torch.randn_like(z1)
            t  = torch.rand(B, 1, device=DEVICE)
            zt = (1 - t) * z0 + t * z1
            target_v = z1 - z0

            pred_v = v_field(zt, t, cond)
            loss   = ((pred_v - target_v) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  loss {total_loss/n_batches:.4f}  ({time.time()-t0:.2f}s)")

    torch.save(v_field.state_dict(), "checkpoints/flow_cond.pt")
    print("saved checkpoints/flow_cond.pt")

    # --- quick sanity: sample each condition at w=3 and check lengths ---
    from model import AutoEncoder, decode
    ae = AutoEncoder(latent_dim=LATENT_DIM)
    ae.load_state_dict(torch.load("checkpoints/autoencoder.pt", map_location=DEVICE))
    ae.eval()

    print("\nSanity check (guidance_scale=3.0):")
    for cond_tok, label in COND_NAMES.items():
        z_norm = sample_ode_cfg(
            v_field, n_samples=200, cond_token=cond_tok,
            guidance_scale=3.0, n_steps=100, device=DEVICE
        )
        z = z_norm * std + mean
        with torch.no_grad():
            toks = ae.decoder.sample(z, greedy=False, temperature=0.9)
        decoded = [decode(t.tolist()) for t in toks]
        avg_len = sum(len(d) for d in decoded) / len(decoded)
        print(f"  {label:20s}  avg_len={avg_len:.1f}  examples: {decoded[:6]}")


if __name__ == "__main__":
    main()
