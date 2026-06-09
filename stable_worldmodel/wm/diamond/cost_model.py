"""Cost model wrapping Diamond's diffusion + reward model for solver integration.

Implements the ``Costable`` protocol so that ``CategoricalCEMSolver`` can
plan by evaluating candidate action sequences in imagination.
"""

import torch
from .edm_sampling import sample_euler


class DiamondCostModel(torch.nn.Module):
    """Wraps diffusion + reward models into a ``Costable`` for planning.

    At each ``get_cost`` call, the model:
    1. Warms up its internal LSTM(s) from the observation history (L steps),
    2. Rolls out each candidate action sequence for H_plan steps,
    3. Returns **negative** cumulative predicted reward as the cost.

    Expected ``info_dict`` keys:
        pixels: (B, L, C, H, W) observation history (preprocessed 64×64 RGB).
        hx_rw:  (B, lstm_dim) reward LSTM hidden state (warmed up).
        cx_rw:  (B, lstm_dim) reward LSTM cell state.
    """

    def __init__(
        self,
        diffusion,
        reward_term,
        action_encoder,
        history_size=4,
        n_denoise=3,
        device='cpu',
    ):
        super().__init__()
        self.diffusion = diffusion
        self.reward_term = reward_term
        self.action_encoder = action_encoder
        self.history_size = history_size
        self.n_denoise = n_denoise
        self.device = device

    @torch.no_grad()
    def warmup(self, obs_buf, act_buf):
        """Warm up the reward LSTM from scratch using real experience history.

        Processes all ``history_size`` steps starting from a zero state,
        returning the final ``(hx, cx)`` to use as the starting point for
        imagination rollouts.  This follows the DIAMOND paper protocol:
        the LSTM must be seeded with L real transitions before planning.

        Args:
            obs_buf: (B, L, C, H, W) observation history.
            act_buf: (B, L) discrete action indices.

        Returns:
            hx: (B, lstm_dim) LSTM hidden state after warmup.
            cx: (B, lstm_dim) LSTM cell state after warmup.
        """
        L = self.history_size
        B = obs_buf.shape[0]
        hx = torch.zeros(
            B, self.reward_term.lstm.hidden_size, device=self.device
        )
        cx = torch.zeros(
            B, self.reward_term.lstm.hidden_size, device=self.device
        )
        for t in range(L):
            frame = obs_buf[:, t]
            act = act_buf[:, t]
            act_emb = self.reward_term.action_embed(act.long())
            vis_emb = self.reward_term.encoder(frame, act_emb)
            inp = torch.cat([vis_emb, act_emb], dim=-1)
            hx, cx = self.reward_term.lstm(inp, (hx, cx))
        return hx, cx

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        """Compute negative cumulative reward for candidate action sequences.

        Args:
            info_dict: dict with keys 'pixels', 'hx_rw', 'cx_rw'.
                The solver has already broadcast these along the candidate
                dimension: shapes are (B, N, L, C, H, W) for pixels and
                (B, N, lstm_dim) for the states.
            action_candidates: (B, N, H_plan, K) one-hot categorical actions
                where K = number of discrete actions.

        Returns:
            cost: (B, N) tensor — lower is better.
        """
        pixels = info_dict['pixels']
        hx_rw = info_dict['hx_rw']
        cx_rw = info_dict['cx_rw']
        B, N, L, C, H_im, W_im = pixels.shape
        H_plan = action_candidates.shape[2]
        K = action_candidates.shape[-1]

        BN = B * N

        obs = pixels.reshape(BN, L, C, H_im, W_im)
        hx = hx_rw.reshape(BN, -1)
        cx = cx_rw.reshape(BN, -1)

        act = torch.zeros(BN, L, dtype=torch.long, device=self.device)

        total_rewards = torch.zeros(BN, device=self.device)

        for t in range(H_plan):
            a_onehot = action_candidates[:, :, t]  # (B, N, K)
            a_idx = a_onehot.argmax(dim=-1)  # (B, N) discrete indices
            a_t = a_idx.reshape(BN)  # (BN,)

            new_act = torch.cat([act[:, 1:], a_t.unsqueeze(-1)], dim=1)
            act_emb = self.action_encoder(new_act)
            cond_vec = act_emb.mean(dim=1)

            history_flat = obs.reshape(BN, C * L, H_im, W_im)
            cond = {'history': history_flat, 'cond_vec': cond_vec}
            frame = sample_euler(
                self.diffusion,
                cond,
                (BN, C, H_im, W_im),
                self.device,
                n_steps=self.n_denoise,
            )

            a_emb = self.reward_term.action_embed(a_t)
            vis_emb = self.reward_term.encoder(frame, a_emb)
            inp = torch.cat([vis_emb, a_emb], dim=-1)
            hx, cx = self.reward_term.lstm(inp, (hx, cx))
            reward = self.reward_term.reward_head(hx).squeeze(-1)
            total_rewards += reward

            obs = torch.cat([obs[:, 1:], frame.unsqueeze(1)], dim=1)
            act = new_act

        cost = -total_rewards.view(B, N)
        return cost

    def criterion(self, info_dict, action_candidates):
        """Alias for ``get_cost`` — satisfies ``Costable`` protocol."""
        return self.get_cost(info_dict, action_candidates)


__all__ = ['DiamondCostModel']
