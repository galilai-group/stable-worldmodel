import torch
import torch.nn.functional as F
from einops import einsum


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Warning: This version only support single-gpu.
    Reference: https://arxiv.org/abs/2511.08544
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device='cuda')
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
            -3
        ).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


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
