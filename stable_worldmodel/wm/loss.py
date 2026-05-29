import torch
import torch.nn.functional as F
from einops import einsum


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (SIGReg).

    Device-agnostic, multi-GPU-capable adapter over the canonical LeJEPA
    implementation shipped in ``stable_pretraining`` (already a dependency of
    this package's ``train`` extra):
    :class:`stable_pretraining.methods.lejepa.SlicedEppsPulley`.

    Compared to the previous in-house implementation this version:

    * runs on any device (CPU / CUDA / MPS) -- the random projection matrix is
      drawn on the input's device rather than a hard-coded ``"cuda"``;
    * supports distributed (DDP) training -- ``SlicedEppsPulley`` all-reduces the
      per-direction Epps-Pulley statistic across ranks (``ReduceOp.AVG``), scales
      by the global batch size, and seeds the random projections identically on
      every rank, so the multi-GPU loss matches the single global-batch
      statistic. It transparently falls back to single-process behaviour when
      ``torch.distributed`` is not initialised.

    Note:
        Embeddings are pooled across all leading axes into ``(N, D)`` before the
        sliced Epps-Pulley test, matching the reference LeJEPA application
        (``all_projected.reshape(-1, D)``). The previous version instead computed
        a separate statistic per time-step; pooling uses a larger effective
        sample count, so the loss magnitude -- and therefore a good SIGReg
        weight -- differs from the old single-GPU version.

    Args:
        knots: number of Epps-Pulley quadrature nodes (maps to ``n_points``);
            must be odd.
        num_proj: number of random 1-D projections (maps to ``num_slices``).

    Reference: https://arxiv.org/abs/2511.08544
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        try:
            from stable_pretraining.methods.lejepa import SlicedEppsPulley
        except ImportError as exc:  # pragma: no cover - requires train extra
            raise ImportError(
                'SIGReg delegates to '
                'stable_pretraining.methods.lejepa.SlicedEppsPulley, which '
                'requires the optional training dependency. Install it with '
                '`pip install "stable-worldmodel[train]"` '
                '(or `pip install stable-pretraining`).'
            ) from exc

        self.num_proj = num_proj
        # knots -> Epps-Pulley quadrature nodes; num_proj -> random slices.
        self.sliced_ep = SlicedEppsPulley(num_slices=num_proj, n_points=knots)

    def forward(self, proj):
        """Compute the SIGReg loss.

        Args:
            proj: embeddings of shape ``(..., D)`` (e.g. ``(T, B, D)``). All
                leading axes are flattened into the sample axis; the statistic
                is invariant to their order.

        Returns:
            Scalar tensor: the mean sliced Epps-Pulley statistic.
        """
        proj = proj.reshape(-1, proj.size(-1))
        return self.sliced_ep(proj)


class VCReg(torch.nn.Module):
    """Variance-Covariance Regularizer"""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def _std_loss(self, z):
        z = z.transpose(0, 1)  # (T, B, D)
        std = (z.var(dim=1) + self.eps).sqrt()  # (T, D)
        std_loss = torch.mean(F.relu(1 - std), dim=-1)  # (T,)
        return std_loss

    def _cov_loss(self, z):
        B, T, D = z.shape
        z = z.transpose(0, 1)  # (T, B, D)
        cov = einsum(z, z, 't b i, t b j -> t i j') / (B - 1)  # (T, D, D)
        diag = einsum(cov, 't i i -> t i').pow(2).sum(dim=-1)  # (T,)
        cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diag).div(D**2 - D)  # (T,)
        return cov_loss

    def forward(self, z):
        """
        z: (..., D)
        """

        if z.dim() == 2:
            D = z.size(-1)
            z = z.view(-1, D)

        z = z - z.mean(
            dim=0, keepdim=True
        )  # mean for each dim across batch samples

        return {
            'std_loss': self._std_loss(z).mean(),
            'std_t_loss': self._std_loss(z.transpose(0, 1)).mean(),
            'cov_loss': self._cov_loss(z).mean(),
            'cov_t_loss': self._cov_loss(z.transpose(0, 1)).mean(),
        }


class PLDMLoss(torch.nn.Module):
    """VCReg anti-collapse + Temporal Alignment + Inverse Dynamics Modeling losses
    reference: https://arxiv.org/abs/2502.14819
    """

    def __init__(self):
        super().__init__()
        self.vc_reg = VCReg()

    def forward(self, z, a_pred=None, a_target=None):
        """
        z: (B, T, D)
        a_pred: (B, T-1, A)
        a_target: (B, T-1, A)
        """

        output = {}
        if a_pred is not None and a_target is not None:
            output['idm_loss'] = F.mse_loss(a_pred, a_target)

        output['temp_align_loss'] = F.mse_loss(z[:, :-1], z[:, 1:])  # detach?
        output.update(self.vc_reg(z))

        return output


class TemporalStraighteningLoss(torch.nn.Module):
    """Temporal Straightening Loss Module (Mean Pairwise Negative Cosine Similarity)
    reference: https://arxiv.org/abs/2603.12231
    """

    def __init__(self):
        super().__init__()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        v = x[:, 1:] - x[:, :-1]  # velocities
        sim = self.cos_sim(v[:, :-1], v[:, 1:])
        return -sim.mean()


__all__ = [
    'PLDMLoss',
    'SIGReg',
    'TemporalStraighteningLoss',
    'VCReg',
]
