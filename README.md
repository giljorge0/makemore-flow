# makemore-flow

A tiny, fully-visualizable companion to Karpathy's [`makemore`](https://github.com/karpathy/makemore),
built around one idea: instead of generating names by sampling discrete
characters one at a time, generate them by **evolving a point through a
continuous 2D "name-space"** along a learned vector field, starting from pure
noise.

This is a minimal implementation of **flow matching** (Lipman et al., 2022 —
the same family of techniques behind Stable Diffusion 3 / modern
image & video generators), applied at a scale small enough that the entire
latent space, the learned vector field, and the generation trajectories can
be plotted on a single 2D figure.

## Why this exists

The original prompt for this project asked: what if "making more" wasn't
just resampling discrete tokens, but mapping data into a space where you
could follow paths, sample neighborhoods, and watch a sequence *evolve*
toward a concept? That's almost exactly the flow-matching framing — a model
learns a velocity field over a latent space, and generation = numerically
integrating that field from noise to data. Most demos of this live in opaque,
high-dimensional image latents. Here the latent space is **2 numbers**, so
you can literally see it.

Two things are genuinely "new" relative to plain `makemore`:

1. **The latent space is continuous and 2D**, learned by a char-level
   sequence autoencoder, so every name is a point in the plane.
2. **Generation is an ODE integration** (Euler steps along a learned
   velocity field), not autoregressive sampling — though the *decoder*
   from latent -> characters is still autoregressive.

Everything else (autoencoders, linear interpolation / "word2vec arithmetic"
style walks, flow matching itself) is well-established. The contribution
here is gluing them together at a scale where you can watch it happen.

## Pipeline

```
names.txt --(char-level seq2seq autoencoder)--> 2D latent space z
                                                      |
                                    train a velocity field v(z, t)
                                    via conditional flow matching:
                                    z_t = (1-t) z0 + t z1,  z0 ~ N(0,I)
                                    target velocity = z1 - z0
                                                      |
generation: z0 ~ N(0,I) --(integrate dz/dt = v(z,t), t: 0->1)--> z1 --(decode)--> name
```

## Files

- `model.py` — vocab + char-level GRU encoder/decoder (the autoencoder that
  defines the latent name-space).
- `train_autoencoder.py` — trains the autoencoder on `data/names.txt`,
  saves `checkpoints/autoencoder.pt` and the latent vector for every name in
  `checkpoints/latents.pt`.
- `train_flow.py` — trains the flow-matching velocity field `v_theta(z, t)`
  on the latents; also contains `sample_ode`, the Euler ODE integrator used
  for generation. Saves `checkpoints/flow.pt`.
- `generate.py` — samples noise, integrates the flow, decodes -> prints new
  names.
- `visualize.py` — produces the two figures described below.

## Running it

```bash
pip install torch matplotlib
python train_autoencoder.py   # ~3 min on CPU, 25 epochs
python train_flow.py          # ~15 sec on CPU, 60 epochs
python generate.py 20         # print 20 new names
python visualize.py           # writes fig1_*.png and fig2_*.png
```

## What the figures show

**`fig1_field_and_trajectories.png`** — every real name plotted as a point
in the (normalized) 2D latent space, colored by name length, with the
learned velocity field overlaid as gray arrows (at t=0.5) and 8 full
noise->name trajectories drawn on top. The `x` marks where each trajectory
starts (pure Gaussian noise); the `o` marks where it ends, labeled with the
decoded name. You can see the field pulling noise inward toward the data
cloud, and that trajectories ending in different regions of the cloud
produce systematically different *kinds* of names (e.g. longer names cluster
in one region).

**`fig2_interpolation.png`** — the classic "follow a line between two
concepts" walk: encode two real names ("sophia" and "william") to latent
points, walk the straight line between them, and decode each point along the
way. The decoded names morph smoothly and recognizably from one to the other.

## Honest caveats / what this is *not*

- The 2D bottleneck is extremely lossy — reconstruction of a specific
  training name is often imperfect (e.g. "sophia" decodes back to "emmen").
  That's the deliberate trade-off for being able to plot the entire space.
  If you want better reconstruction, bump `LATENT_DIM` in `model.py` /
  `train_flow.py` to e.g. 8 and use PCA for the 2D plots instead.
- This is not a new architecture or a research contribution — it's flow
  matching, a 2022-era technique, applied at toy scale for a name dataset.
  The value is pedagogical / exploratory, not a paper.
- Sample quality is well below `makemore`'s own Transformer model. This
  trades quality for *visualizability* of the whole generative process.

## Ideas for extending this

- Increase `LATENT_DIM` and use PCA to project the field/trajectories to 2D
  for visualization while keeping a richer space for generation.
- Condition the velocity field on extra attributes (e.g. desired name
  length, or "starts with vowel") to do classifier-free-guidance-style
  steering — directly the "representation engineering" direction from the
  original idea.
- Replace the linear conditional path with a different probability path
  (e.g. variance-preserving diffusion path) and compare trajectory shapes.
- Make `fig1` an animation (matplotlib `FuncAnimation`) showing the point
  cloud morphing from a Gaussian blob into the name-cloud shape as `t`
  goes 0 -> 1.
