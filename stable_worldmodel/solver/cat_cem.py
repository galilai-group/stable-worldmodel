import time

import numpy as np
import torch
from gymnasium.spaces import Discrete, MultiDiscrete

from .solver import Costable


class CategoricalCEMSolver:
    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        independence: bool = True,
        warmstart_eps: float = 0.05,
        n_steps: int = 30,
        topk: int = 30,
        device: str = "cpu",
        seed: int = 1234,
    ):
        """Categorical CEM planner for Discrete/MultiDiscrete action spaces using elite-sample updates.
        Args:
            model (Costable): World model used to score candidate action sequences.
            batch_size (int): Number of envs processed per batch.
            num_samples (int): Number of candidates sampled per CEM iteration.
            independence (bool): If True, factorize over MultiDiscrete components; else use joint categorical.
            warmstart_eps (float): Smoothing factor for warm-start distributions.
            n_steps (int): Number of CEM refinement iterations.
            topk (int): Number of elites used to update logits each iteration.
            device (str): Torch device string (e.g., "cpu", "cuda").
            seed (int): Random seed for sampling.

        """
        self.model = model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.independence = bool(independence)
        self.warmstart_eps = float(warmstart_eps)
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)

        self._configured = False

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config

        # Normalize to MultiDiscrete-style nvec
        if isinstance(action_space, Discrete):
            nvec = np.array([int(action_space.n)], dtype=np.int64)
        elif isinstance(action_space, MultiDiscrete):
            nvec = np.array(action_space.nvec, dtype=np.int64)
            # VecEnv can sometimes wrap nvec with leading env dim
            if nvec.ndim == 2:
                nvec = nvec[0]
        else:
            raise ValueError(
                f"Action space is not Discrete/MultiDiscrete, got {type(action_space)}. "
                "CategoricalCEMSolver expects discrete actions."
            )

        assert nvec.ndim == 1 and nvec.size >= 1, f"Expected 1D nvec, got shape {nvec.shape}"
        assert np.all(nvec > 0), f"All nvec entries must be positive, got {nvec}"

        self._nvec = nvec.tolist()
        self._k = len(self._nvec)
        self._action_dim = int(np.sum(self._nvec))
        self._K_joint = int(np.prod(self._nvec))

        # slices into packed _action_dim
        self._comp_slices: list[tuple[int, int]] = []
        off = 0
        for ni in self._nvec:
            self._comp_slices.append((off, off + int(ni)))
            off += int(ni)
        assert off == self._action_dim

        # offsets for fast scatter into _action_dim
        self._offsets = torch.tensor([s for (s, _) in self._comp_slices], dtype=torch.long, device=self.device)

        # strides for joint ravel/unravel
        strides = []
        prod_tail = 1
        for i in range(self._k - 1, -1, -1):
            strides.append(prod_tail)
            prod_tail *= int(self._nvec[i])
        strides = list(reversed(strides))
        self._strides = torch.tensor(strides, dtype=torch.long, device=self.device)
        self._nvec_t = torch.tensor(self._nvec, dtype=torch.long, device=self.device)

        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def horizon(self) -> int:
        return int(self._config.horizon)

    @property
    def action_block(self) -> int:
        return int(self._config.action_block)

    @property
    def action_dim(self) -> int:
        # world model sees blocked concatenation per time step
        return self.action_block * self._action_dim

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def _ravel_joint(self, comp_idx: torch.Tensor) -> torch.Tensor:
        """Convert per-component MultiDiscrete indices into a single joint index.

        Args:
            comp_idx (torch.Tensor): Tensor of shape (..., k) with per-component indices.

        Returns:
            torch.Tensor: Tensor of shape (...) with joint indices in [0, prod(nvec)).
        """
        # joint_index = sum_i idx[i] * stride[i]
        return (comp_idx.long() * self._strides).sum(dim=-1)

    def _unravel_joint(self, joint_idx: torch.Tensor) -> torch.Tensor:
        """Convert joint indices into per-component MultiDiscrete indices.

        Args:
            joint_idx (torch.Tensor): Tensor of shape (...) with joint indices in [0, prod(nvec)).

        Returns:
            torch.Tensor: Tensor of shape (..., k) with per-component indices.
        """
        # idx[i] = (joint_idx // stride[i]) % nvec[i]
        j = joint_idx.long().unsqueeze(-1)  # (..., 1)
        idx = torch.div(j, self._strides, rounding_mode="floor") % self._nvec_t  # (..., k)
        return idx

    def _indices_to_onehot_concat(self, idx: torch.Tensor) -> torch.Tensor:
        """Convert categorical indices into concatenated one-hot vectors over components.

        Args:
            idx (torch.Tensor): Categorical action indices in canonical form with shape
                (..., B, k), where B is action_block and k is the number of (Multi)Discrete
                components (k=1 for Discrete). Values must satisfy idx[..., i] in [0, nvec[i]).

        Returns:
            torch.Tensor: Concatenated one-hot encoding with shape (..., B, sum(nvec)),
                matching the input prefix dimensions and replacing the last dim k with
                self._action_dim = sum(nvec).
        """
        B = idx.shape[-2]
        k = idx.shape[-1]

        if k != self._k:
            raise ValueError(f"Expected last dim k={self._k}, got {k}")

        # flatten all leading dims into one
        prefix = idx.shape[:-2]
        idx2 = idx.reshape(-1, B, k)  # (P, B, k)

        # offsets: (k,) -> (1,1,k)
        global_idx = idx2 + self._offsets.view(1, 1, k)  # (P,B,k) in [0, self._action_dim)

        out = torch.zeros((idx2.shape[0], B, self._action_dim), device=self.device, dtype=torch.float32)
        out.scatter_add_(-1, global_idx, torch.ones_like(global_idx, dtype=torch.float32))
        return out.reshape(*prefix, B, self._action_dim)

    def init_action_distrib(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Initialize logits for the action distribution, optionally warm-started from indices using smoothing.

        Args:
            actions (torch.Tensor, optional): Warm-start indices. Discrete expects (E,T,B).
                MultiDiscrete expects (E,T,B*k)

        Returns:
            torch.Tensor: Initial logits tensor. Shape is (E,H,B,sum(nvec)) if independence=True,
                else (E,H,B,prod(nvec)).
        """
        assert self._configured, "Call configure() first."
        E, H = self.n_envs, self.horizon
        B = int(self._config.action_block)
        eps = float(self.warmstart_eps)

        # Uniform prior everywhere by default: logits = 0
        if actions is None:
            if self.independence:
                return torch.zeros((E, H, B, self._action_dim), device=self.device)
            else:
                return torch.zeros((E, H, B, self._K_joint), device=self.device)

        T = int(actions.shape[1])
        # Normalize action shape:
        if self._k == 1:
            # Discrete (E,T,B) -> (E,T,B,1)
            a = actions.unsqueeze(-1)
        else:
            # MultiDiscrete (E,T,B*k) -> (E,T,B,k)
            a = actions.reshape(E, T, B, self._k)

        a = a.long()

        if self.independence:
            logits = torch.zeros((E, H, B, self._action_dim), device=self.device)

            # Fill first T steps with smoothed per-component distributions
            for comp_i, ((s, e), ni) in enumerate(zip(self._comp_slices, self._nvec)):
                idx = a[:, :T, :, comp_i]  # (E,T,B) categorical indices

                # p = eps/ni everywhere, then scatter chosen index to (1-eps + eps/ni)
                p = torch.full((E, T, B, int(ni)), eps / float(ni), device=self.device)
                p.scatter_(-1, idx.unsqueeze(-1), 1.0 - eps + eps / float(ni))

                logits[:, :T, :, s:e] = torch.log(p.clamp_min(1e-12))

            return logits

        # Joint
        logits_joint = torch.zeros((E, H, B, self._K_joint), device=self.device)

        joint_idx = self._ravel_joint(a[:, :T, :, :])  # (E,T,B)
        p = torch.full((E, T, B, self._K_joint), eps / float(self._K_joint), device=self.device)
        p.scatter_(-1, joint_idx.unsqueeze(-1), 1.0 - eps + eps / float(self._K_joint))

        logits_joint[:, :T] = torch.log(p.clamp_min(1e-12))
        return logits_joint

    def _sample_factorized_indices(self, logits_packed: torch.Tensor, bs: int) -> torch.Tensor:
        """Sample candidate indices from factorized logits.

        Args:
            logits_packed (torch.Tensor): Logits of shape (bs,H,B,sum(nvec)).
            bs (int): Batch size (number of envs).

        Returns:
            torch.Tensor: Sampled indices of shape (bs,N,H,B,k).
        """
        H, B, N = self.horizon, self.action_block, self.num_samples
        idx = torch.empty((bs, N, H, B, self._k), device=self.device, dtype=torch.long)

        for comp_i, ((s, e), ni) in enumerate(zip(self._comp_slices, self._nvec)):
            ni = int(ni)
            probs = torch.softmax(logits_packed[:, :, :, s:e], dim=-1)  # (bs,H,B,ni)

            # Flatten (bs*H*B, ni) and sample N per row -> (bs*H*B, N)
            probs_flat = probs.reshape(-1, ni)
            samp = torch.multinomial(
                probs_flat, num_samples=N, replacement=True, generator=self.torch_gen
            )  # (bs*H*B, N)

            # Reshape to (bs, H, B, N) then permute to (bs, N, H, B)
            idx[..., comp_i] = samp.reshape(bs, H, B, N).permute(0, 3, 1, 2)

        return idx

    def _sample_joint_indices(self, logits_joint: torch.Tensor, bs: int) -> torch.Tensor:
        """Sample candidate indices from joint logits and also return unraveled component indices.

        Args:
            logits_joint (torch.Tensor): Logits of shape (bs,H,B,prod(nvec)).
            bs (int): Batch size (number of envs).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - joint_idx: (bs,N,H,B) joint indices
                - comp_idx: (bs,N,H,B,k) per-component indices
        """
        H, B, N = self.horizon, self.action_block, self.num_samples
        K = self._K_joint

        probs = torch.softmax(logits_joint, dim=-1)  # (bs,H,B,K)
        probs_flat = probs.reshape(-1, K)  # (bs*H*B, K)

        joint_flat = torch.multinomial(
            probs_flat, num_samples=N, replacement=True, generator=self.torch_gen
        )  # (bs*H*B, N)

        # reshape to (bs,N,H,B)
        joint = joint_flat.reshape(bs, H, B, N).permute(0, 3, 1, 2).contiguous()

        # unravel: flatten (bs*N*H*B,) then reshape back
        comp = self._unravel_joint(joint.reshape(-1)).reshape(bs, N, H, B, self._k)

        return joint, comp

    def _update_factorized_from_elites_idx(self, elites_idx: torch.Tensor) -> torch.Tensor:
        """Update packed logits using elite samples (factorized case).

        Args:
            elites_idx (torch.Tensor): Elite indices of shape (bs,K,H,B,k).

        Returns:
            torch.Tensor: Updated logits of shape (bs,H,B,sum(nvec)).
        """
        bs, K, H, B, k = elites_idx.shape
        logits_new = torch.zeros((bs, H, B, self._action_dim), device=self.device, dtype=torch.float32)

        for comp_i, ((s, e), ni) in enumerate(zip(self._comp_slices, self._nvec)):
            ni = int(ni)
            idx = elites_idx[..., comp_i]  # (bs,K,H,B)

            # Flatten (bs*H*B, K) and build histogram (bs*H*B, ni)
            flat_idx = idx.permute(0, 2, 3, 1).reshape(-1, K)  # (R, K), R=bs*H*B
            counts = torch.zeros((flat_idx.shape[0], ni), device=self.device, dtype=torch.float32)
            counts.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))

            p = (counts / float(K)).clamp_min(1e-12)
            p = p / p.sum(dim=-1, keepdim=True)
            logits_new[:, :, :, s:e] = torch.log(p).reshape(bs, H, B, ni)

        return logits_new

    def _update_joint_from_elites_jointidx(self, elites_joint: torch.Tensor) -> torch.Tensor:
        """Update joint logits using elite samples (joint case).

        Args:
            elites_joint (torch.Tensor): Elite joint indices of shape (bs,K,H,B).

        Returns:
            torch.Tensor: Updated logits of shape (bs,H,B,prod(nvec)).
        """
        # elites_joint: (bs,K,H,B)
        bs, K, H, B = elites_joint.shape
        flat = elites_joint.permute(0, 2, 3, 1).reshape(-1, K)  # (R,K)
        counts = torch.zeros((flat.shape[0], self._K_joint), device=self.device, dtype=torch.float32)
        counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))

        p = (counts / float(K)).clamp_min(1e-12)
        p = p / p.sum(dim=-1, keepdim=True)
        return torch.log(p).reshape(bs, H, B, self._K_joint)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        """Solve the planning optimization problem using categorical CEM with batch processing.

        Args:
            info_dict (dict): Per-env info inputs for the world model.
                init_action (torch.Tensor, optional): Warm-start indices passed to init_action_distrib().

        Returns:
            dict: A dictionary containing:
                - "actions" (torch.Tensor): Mode action indices of shape (E,H,B*k).
                - "logits" (list[torch.Tensor]): List with final logits tensor.
                - "costs" (list[float]): Final mean elite cost per environment.
        """
        start_time = time.time()
        outputs = {"costs": [], "logits": []}

        logits = self.init_action_distrib(init_action).to(self.device)

        total_envs = self.n_envs
        H, B, N = self.horizon, self.action_block, self.num_samples

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            bs = end_idx - start_idx

            batch_logits = logits[start_idx:end_idx]

            # Expand info_dict to (bs,N,...)
            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    v_batch = v_batch.unsqueeze(1)
                    v_batch = v_batch.expand(bs, N, *v_batch.shape[2:])
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], N, axis=1)
                expanded_infos[k] = v_batch

            final_batch_cost = None

            for _ in range(self.n_steps):
                # Sample indices
                if self.independence:
                    idx = self._sample_factorized_indices(batch_logits, bs)  # (bs,N,H,B,k)

                    # Force sample 0 to be mode indices
                    mode = torch.empty((bs, H, B, self._k), device=self.device, dtype=torch.long)
                    for comp_i, ((s, e), _) in enumerate(zip(self._comp_slices, self._nvec)):
                        mode[..., comp_i] = batch_logits[:, :, :, s:e].argmax(dim=-1)
                    idx[:, 0] = mode

                    # Convert to worldmodel candidates one-hot concat
                    atomic = self._indices_to_onehot_concat(idx)  # (bs,N,H,B,_action_dim)
                    candidates = atomic.reshape(bs, N, H, B * self._action_dim)

                    costs = self.model.get_cost(expanded_infos.copy(), candidates)

                    topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                    batch_ids = torch.arange(bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                    elites_idx = idx[batch_ids, topk_inds]  # (bs,K,H,B,k)

                    batch_logits = self._update_factorized_from_elites_idx(elites_idx)

                else:
                    joint_idx, comp_idx = self._sample_joint_indices(batch_logits, bs)  # (bs,N,H,B), (bs,N,H,B,k)

                    # Force sample 0 to be joint mode
                    joint_mode = batch_logits.argmax(dim=-1)  # (bs,H,B)
                    joint_idx[:, 0] = joint_mode
                    comp_idx[:, 0] = self._unravel_joint(joint_mode.reshape(-1)).reshape(bs, H, B, self._k)

                    atomic = self._indices_to_onehot_concat(comp_idx)  # (bs,N,H,B,_action_dim)
                    candidates = atomic.reshape(bs, N, H, B * self._action_dim)

                    costs = self.model.get_cost(expanded_infos.copy(), candidates)

                    topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                    batch_ids = torch.arange(bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                    elites_joint = joint_idx[batch_ids, topk_inds]  # (bs,K,H,B)

                    batch_logits = self._update_joint_from_elites_jointidx(elites_joint)

                final_batch_cost = topk_vals.mean(dim=1).detach().cpu().tolist()

            logits[start_idx:end_idx] = batch_logits
            outputs["costs"].extend(final_batch_cost)

        # Final mode actions as indices
        E = total_envs
        if self.independence:
            actions_idx = torch.empty((E, H, B, self._k), device=self.device, dtype=torch.long)
            for comp_i, ((s, e), _) in enumerate(zip(self._comp_slices, self._nvec)):
                actions_idx[..., comp_i] = logits[:, :, :, s:e].argmax(dim=-1)
        else:
            joint_mode = logits.argmax(dim=-1)  # (E,H,B)
            actions_idx = self._unravel_joint(joint_mode.reshape(-1)).reshape(E, H, B, self._k)

        actions_idx_out = actions_idx.reshape(E, H, B * self._k)

        outputs["actions"] = actions_idx_out.detach().cpu()
        outputs["logits"] = [logits.detach().cpu()]

        print(f"Categorical CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs
