"""
Train the 8-dimensional char-level autoencoder.

Why 8D instead of 2D?
  The 2D bottleneck forces the model to compress every name into just 2
  numbers. It works for visualisation, but reconstruction is lossy
  (sophia -> emmin) and the length clusters heavily overlap, limiting how
  well conditioned generation can steer. 8D fixes both: reconstruction
  becomes near-perfect and the length signal separates cleanly.
  We use PCA to project down to 2D for visualisation.

Outputs:
  checkpoints/autoencoder_8d.pt   – encoder + decoder weights
  checkpoints/latents_8d.pt       – 8D latent for every name + stats
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import AutoEncoder, encode, decode, VOCAB_SIZE
from train_autoencoder import NameDataset, collate

LATENT_DIM = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(0)
    ds = NameDataset()
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=collate)

    model = AutoEncoder(latent_dim=LATENT_DIM, embed_dim=32, hidden=128).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-3)

    EPOCHS = 30
    for epoch in range(EPOCHS):
        t0 = time.time(); total = 0; nb = 0
        for x, lens in dl:
            x, lens = x.to(DEVICE), lens.to(DEVICE)
            logits, _ = model(x, lens)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), x[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:2d}  loss {total/nb:.4f}  ({time.time()-t0:.1f}s)")

    # qualitative reconstruction check
    model.eval()
    test_names = ["emma", "olivia", "william", "sophia", "liam", "isabella", "charlotte"]
    print("\nReconstruction check:")
    for name in test_names:
        ids  = torch.tensor([encode(name)], device=DEVICE)
        lens = torch.tensor([len(ids[0])], device=DEVICE)
        with torch.no_grad():
            z    = model.encoder(ids, lens)
            toks = model.decoder.sample(z, greedy=True)
        print(f"  {name:12s} -> {decode(toks[0].tolist())}")

    # save model
    torch.save({"state_dict": model.state_dict(), "latent_dim": LATENT_DIM},
               "checkpoints/autoencoder_8d.pt")

    # compute and save all latents
    print("\ncomputing latents for full dataset...")
    all_z = []
    dl2   = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate)
    model.eval()
    with torch.no_grad():
        for x, lens in dl2:
            x, lens = x.to(DEVICE), lens.to(DEVICE)
            all_z.append(model.encoder(x, lens).cpu())
    all_z = torch.cat(all_z)
    mean, std = all_z.mean(0), all_z.std(0)
    torch.save({"z": all_z, "names": ds.names, "mean": mean, "std": std,
                "latent_dim": LATENT_DIM}, "checkpoints/latents_8d.pt")
    print(f"saved checkpoints/autoencoder_8d.pt and checkpoints/latents_8d.pt")
    print(f"latent shape: {all_z.shape}")


if __name__ == "__main__":
    main()
