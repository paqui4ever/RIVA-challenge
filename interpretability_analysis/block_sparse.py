# Copied and adapted from Goodfire AI's block-sparse-featurizer:
# https://github.com/goodfire-ai/block-sparse-featurizer
# Associated paper: "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds"
# (arXiv:2606.25234, Fel et al.)

"""
Block-Sparse Featurizers (BSFs).

Unlike standard Sparse Autoencoders (SAEs) that treat each concept as an isolated 1D direction,
BSFs group latent directions into low-dimensional subspaces (blocks). This allows the model
to capture continuous concept manifolds (poses, lighting, shapes) natively within subspaces.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ==============================================================================
# Base BSF Functions (Copied from goodfire-ai/block-sparse-featurizer bsf/base.py)
# ==============================================================================

def group_topk(group_norms: torch.Tensor, l0: int) -> torch.Tensor:
    """Per-sample block TopK (the projection Pi_l): in each row keep the `l0`
    blocks of largest norm and zero the rest. Returns a {0, 1} mask (N, G).
    
    Copied from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/base.py
    """
    idx = group_norms.topk(l0, dim=-1).indices
    return torch.zeros_like(group_norms).scatter_(1, idx, 1.0)


def unit_blocks(W: torch.Tensor, n_groups: int, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """Normalise each decoder atom to unit L2 norm.
    
    Copied from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/base.py
    """
    Wb = rearrange(W, '(g s) d -> g s d', g=n_groups, s=group_size)
    norm = Wb.norm(dim=2, keepdim=True).clamp_min(eps)
    Wb_normed = Wb / norm
    return rearrange(Wb_normed, 'g s d -> (g s) d')


class BSF(nn.Module):
    """Block-Sparse Featurizer Base Class.

    Subclasses implement `encode_blocks(x) -> (N, n_groups, group_size)` and
    define their decoder atoms.
    
    Adapted from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/base.py
    """

    def __init__(self, d: int, n_groups: int, group_size: int):
        super().__init__()
        self.d = d
        self.input_shape = d
        self.n_groups = n_groups
        self.nb_concepts = n_groups
        self.group_size = group_size

    def encode_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes activations into block coordinates (N, n_groups, group_size)."""
        raise NotImplementedError

    def decoder_atoms(self) -> torch.Tensor:
        """Returns decoder weights of shape (n_groups * group_size, d)."""
        return self.W_dec

    def get_dictionary(self) -> torch.Tensor:
        """Alias for compatibility with overcomplete metrics."""
        return self.decoder_atoms()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent block coordinates (N, n_groups, group_size) back to (N, d)."""
        return rearrange(z, 'n g s -> n (g s)') @ self.decoder_atoms()

    def atoms(self) -> torch.Tensor:
        """Returns per-concept decoder atoms (n_groups, group_size, d) for visualization."""
        W = self.decoder_atoms().detach()
        return rearrange(W, '(g s) d -> g s d', g=self.n_groups, s=self.group_size)

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-L2 normalizes each decoder atom (no-op for orthonormal decoder)."""
        if hasattr(self, 'W_dec'):
            self.W_dec.data.copy_(
                unit_blocks(self.W_dec.data, self.n_groups, self.group_size)
            )

    def forward(self, x: torch.Tensor):
        """
        Forward method designed to be 100% compatible with overcomplete and train_saes.py.
        Returns:
            z_pre: Pre-activation norms per concept (N, n_groups)
            z: Gated norms per concept (N, n_groups)
            x_hat: Reconstructed activation tensor (N, d)
        """
        z_blocks, z_pre_norms = self.encode_with_pre(x)
        x_hat = self.decode(z_blocks)
        z_norms = z_blocks.norm(dim=-1)
        return z_pre_norms, z_norms, x_hat


# ==============================================================================
# Vanilla BSF (Copied and adapted from bsf/vanilla.py)
# ==============================================================================

class VanillaBSF(BSF):
    """Vanilla Block-Sparse Featurizer.

    A free linear encoder produces signed per-block codes; the block projection
    Pi_l keeps the `l0` blocks of largest norm and zeroes the rest. Encoder and
    decoder are untied; sparsity is enforced by construction (Block TopK).

    Copied and adapted from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/vanilla.py
    """

    def __init__(self, d: int, n_groups: int, group_size: int = 4, l0: int = 16):
        super().__init__(d, n_groups, group_size)
        self.l0 = l0
        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        # tied init
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))

    def decoder_atoms(self) -> torch.Tensor:
        return unit_blocks(self.W_dec, self.n_groups, self.group_size)

    def preact(self, x: torch.Tensor) -> torch.Tensor:
        a = x @ self.W_enc + self.b_enc
        return rearrange(a, 'n (g s) -> n g s', g=self.n_groups, s=self.group_size)

    def encode_blocks(self, x: torch.Tensor) -> torch.Tensor:
        z_blocks, _ = self.encode_with_pre(x)
        return z_blocks

    def encode_with_pre(self, x: torch.Tensor):
        a = self.preact(x)
        pre_norms = a.norm(dim=-1)
        mask = group_topk(pre_norms, self.l0)
        z = a * mask.unsqueeze(-1)
        return z, pre_norms

    def loss(self, x: torch.Tensor, target: torch.Tensor = None):
        target = x if target is None else target
        _, _, x_hat = self(x)
        recon = (target - x_hat).pow(2).mean()
        return recon, {'recon': recon.item()}


# ==============================================================================
# Grassmannian BSF (Copied and adapted from bsf/grassmannian.py)
# ==============================================================================

class GrassmannianBSF(BSF):
    """Grassmannian Block-Sparse Featurizer.

    Each concept is an orthonormal `group_size`-frame D_g. Encoder and decoder are
    tied to the same frame: z_g = gamma_g * x D_g^T, and the block projection Pi_l
    keeps the `l0` frames of largest energy.

    Copied and adapted from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/grassmannian.py
    """

    def __init__(self, d: int, n_groups: int, group_size: int = 4, l0: int = 16):
        super().__init__(d, n_groups, group_size)
        self.l0 = l0
        B = torch.randn(n_groups, d, group_size)
        B, _ = torch.linalg.qr(B)
        self.B_raw = nn.Parameter(B)
        self.gamma = nn.Parameter(torch.ones(n_groups))

    def decoder_atoms(self) -> torch.Tensor:
        B, _ = torch.linalg.qr(self.B_raw)
        return rearrange(B, 'g d s -> (g s) d')

    def encode_with_pre(self, x: torch.Tensor):
        atoms = self.decoder_atoms()
        z = rearrange(x @ atoms.t(), 'n (g s) -> n g s', g=self.n_groups, s=self.group_size)
        z = z * self.gamma[None, :, None]
        pre_norms = z.norm(dim=-1)
        mask = group_topk(pre_norms, self.l0)
        return z * mask.unsqueeze(-1), pre_norms

    def encode_blocks(self, x: torch.Tensor) -> torch.Tensor:
        z_blocks, _ = self.encode_with_pre(x)
        return z_blocks

    def loss(self, x: torch.Tensor, target: torch.Tensor = None):
        target = x if target is None else target
        _, _, x_hat = self(x)
        recon = (target - x_hat).pow(2).mean()
        return recon, {'recon': recon.item()}


# ==============================================================================
# Visualization Utilities for Manifolds (Copied & adapted from bsf/viz.py)
# ==============================================================================

def pca_fit(c: np.ndarray, k: int = 3):
    """Mean and top-`k` principal axes of contribution cloud `c` (n, d).
    
    Copied from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/viz.py
    """
    mean = c.mean(0)
    comps = np.linalg.svd(c - mean, full_matrices=False)[2][:k]
    return mean, comps


def radial_clip(proj: np.ndarray, pct: float = 98.0) -> np.ndarray:
    """Radial outlier clipping.
    
    Copied from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/viz.py
    """
    norm = np.linalg.norm(proj, axis=1)
    cap = float(np.percentile(norm, pct))
    s = np.minimum(1.0, cap / np.maximum(norm, 1e-12))
    return proj * s[:, None]


def make_colorize(proj: np.ndarray, saturation: float = 1.0):
    """Computes color hue from the radial direction of each point on the manifold.
    
    Copied from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/viz.py
    """
    def f(p):
        n = np.linalg.norm(p, axis=-1, keepdims=True)
        unit = p / np.maximum(n, 1e-8)
        return (0.5 + 0.5 * unit * saturation).clip(0, 1)
    return f


def plot_block_concepts_manifold(
    z_blocks: np.ndarray,
    atoms: np.ndarray,
    images: np.ndarray,
    concepts: list,
    grid_size: int,
    n_img: int = 5,
    save_path: str = None
):
    """
    Plots 3D manifold geometry and image patch overlays for selected concept blocks.
    
    Adapted from: https://github.com/goodfire-ai/block-sparse-featurizer/blob/main/bsf/viz.py
    """
    P = grid_size * grid_size
    heat = np.linalg.norm(z_blocks, axis=-1) # (N, G)
    zr = rearrange(z_blocks, '(n p) g k -> n p g k', p=P)
    heat_img = rearrange(heat, '(n p) g -> n p g', p=P)

    n_con = len(concepts)
    fig, axes = plt.subplots(n_con, n_img + 1, figsize=(3 * (n_img + 1), 3 * n_con),
                             subplot_kw={'projection': None})

    if n_con == 1:
        axes = np.expand_dims(axes, 0)

    for r, g in enumerate(concepts):
        idx = np.where(heat[:, g] > 1e-6)[0]
        if idx.size < 8:
            axes[r, 0].text(0.5, 0.5, f"Concept {g}\nInactive", ha='center', va='center')
            axes[r, 0].axis('off')
            continue

        sub = idx if idx.size <= 2000 else np.random.choice(idx, 2000, replace=False)
        c = z_blocks[sub, g, :] @ atoms[g]
        mean, comps = pca_fit(c, 3)
        proj = (c - mean) @ comps.T
        proj = radial_clip(proj)
        colorize = make_colorize(proj)
        rgb = colorize(proj)

        # 3D scatter in the first column
        ax3d = fig.add_subplot(n_con, n_img + 1, r * (n_img + 1) + 1, projection='3d')
        axes[r, 0].remove() # replace dummy 2d with 3d
        norm = np.linalg.norm(proj, axis=1)
        inten = (norm / max(norm.max(), 1e-8)).clip(0, 1)
        rgba = np.concatenate([np.asarray(rgb), inten[:, None]], axis=1)
        ax3d.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=rgba, s=3.0, edgecolors='none')
        ax3d.set_title(f"Concept {g} Manifold")
        ax3d.set_axis_off()

        # Top activating images
        top_imgs = np.argsort(-(heat_img[:, :, g] ** 2).sum(1))[:n_img]
        for j, img_idx in enumerate(top_imgs):
            ax = axes[r, j + 1]
            img = images[img_idx]
            z_patch = zr[img_idx, :, g, :]
            c_p = z_patch @ atoms[g]
            proj_p = (c_p - mean) @ comps.T
            rgb_p = colorize(proj_p)
            norm_p = np.linalg.norm(c_p, axis=1)
            alpha_p = (norm_p / max(norm_p.max(), 1e-8)).clip(0, 1)
            rgba_p = np.concatenate([rgb_p, alpha_p[:, None]], 1).reshape(grid_size, grid_size, 4)

            h, w = img.shape[:2]
            ax.imshow(img)
            ax.imshow(rgba_p, extent=(0, w, h, 0), interpolation='bicubic')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Rank {j+1}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved manifold visualization to {save_path}")
    plt.close()
    return fig
