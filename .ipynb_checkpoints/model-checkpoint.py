import numpy as np
import torch

from torch.distributions import Categorical
from torch import nn
from torch.nn import functional as F
import tntorch as tn
from transformer import Transformer

class ActorCriticModel(nn.Module):
    def __init__(self, config, observation_space, action_space_shape, max_episode_length, device):
        super().__init__()

        # ─── experiment flags ──────────────────────────────────────────
        self.svd_rank_frac = config.get("svd_rank_frac", None)
        self.tt_rank_frac  = config.get("tt_rank_frac",  None)
        self.gauss_filter  = config.get("gauss_filter", False)
        self.laplace_filter = config.get("laplace_filter", False)

        if self.svd_rank_frac:   print(f"SVD frac  = {self.svd_rank_frac}")
        if self.tt_rank_frac:    print(f"TT  frac  = {self.tt_rank_frac}")
        if self.gauss_filter:    print("Gaussian filter will be applied")
        if self.laplace_filter:  print("Laplacian filter will be applied")

        # ─── basic attrs ───────────────────────────────────────────────
        self.device = device                       # итоговое устройство
        self.hidden_size        = config["hidden_layer_size"]
        self.memory_layer_size  = config["transformer"]["embed_dim"]
        self.max_episode_length = max_episode_length
        self.observation_space_shape = observation_space.shape

        # ─── 1. encoder (CPU) ──────────────────────────────────────────
        if len(self.observation_space_shape) > 1:           # visual obs
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4, device='cpu')
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0, device='cpu')
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0, device='cpu')
            for c in (self.conv1, self.conv2, self.conv3):
                nn.init.orthogonal_(c.weight, np.sqrt(2))
            self.conv_out_size = self._get_conv_output(observation_space.shape)
            in_features = self.conv_out_size
        else:                                              # vector obs
            in_features = observation_space.shape[0]

        # ─── 2. MLP & Transformer (CPU) ────────────────────────────────
        self.lin_hidden = nn.Linear(in_features, self.memory_layer_size, device='cpu')
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        self.transformer = Transformer(
            config["transformer"],
            self.memory_layer_size,
            self.max_episode_length
        ).cpu()                                            # ensure CPU

        self.lin_policy = nn.Linear(self.memory_layer_size, self.hidden_size, device='cpu')
        self.lin_value  = nn.Linear(self.memory_layer_size, self.hidden_size, device='cpu')
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))
        nn.init.orthogonal_(self.lin_value.weight,  np.sqrt(2))

        self.policy_branches = nn.ModuleList()
        for n_act in action_space_shape:
            head = nn.Linear(self.hidden_size, n_act, device='cpu')
            nn.init.orthogonal_(head.weight, np.sqrt(0.01))
            self.policy_branches.append(head)

        self.value = nn.Linear(self.hidden_size, 1, device='cpu')
        nn.init.orthogonal_(self.value.weight, 1)

    def svd_low_rank_safe(self, x: torch.Tensor,
                          energy_frac: float = 1.0, max_try: int = 3) -> torch.Tensor:
        orig_shape = x.shape                    # (..., D)
        x_2d = x.reshape(-1, x.shape[-1])       # (M, D)   без батча
    
        for i in range(max_try):
            try:
                U, S, Vh = torch.linalg.svd(x_2d, full_matrices=False)
                break
            except RuntimeError:
                if i == max_try - 1:
                    return x
                x_2d = x_2d.cpu().double()
    
        if energy_frac < 1.0:
            energy = torch.cumsum(S ** 2, dim=-1) / (S ** 2).sum()
            k = int((energy < energy_frac).sum() + 1)
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    
        x_hat = (U * S.unsqueeze(-2)) @ Vh       # (M, D)
        return x_hat.reshape(orig_shape)

    def tt_low_rank_safe(self, memory: torch.Tensor,
                         energy_frac: float = 1.0,
                         max_rank: int = 200) -> torch.Tensor:
        """
        TT-аппроксимация с удержанием заданной доли энергии.
        Возвращает dense-тензор той же формы.
        """
        if energy_frac >= 1.0:
            return memory                               # ничего не сжимаем
    
        shape = memory.shape
        flat  = memory.reshape(-1, shape[-2], shape[-1])
        full_norm = torch.linalg.norm(flat)
    
        for r in range(1, max_rank + 1):
            mem_tt_r = tn.Tensor(flat, ranks_tt=r)      # построить TT ранга r
            approx   = mem_tt_r.torch()
            if torch.linalg.norm(approx) / full_norm >= energy_frac:
                return approx.reshape_as(memory)

        return approx.reshape_as(memory)

    def apply_gaussian_filter(self, memory: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        Gaussian 1D-фильтр по последней размерности (D) → shape: (B, blocks, D)
        """
        B, blocks, D = memory.shape
        kernel_size = int(torch.ceil(torch.tensor(6 * sigma))) | 1
        half = (kernel_size - 1) // 2
        x = torch.arange(-half, half + 1, dtype=memory.dtype, device=memory.device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
    
        mem = memory.view(B * blocks, 1, D)                      # (B*blocks, 1, D)
        smoothed = F.conv1d(mem, kernel, padding=half)
        return smoothed.view(B, blocks, D)

    def apply_laplacian_filter(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Лаплас-фильтр по последней оси (фичи D).
        Работает для памяти (B, blocks, D) и (B, L, blocks, D).
        """
        shift_left  = torch.roll(memory,  1, dims=-1)
        shift_right = torch.roll(memory, -1, dims=-1)
        laplacian = shift_left + shift_right - 2 * memory
        return memory + laplacian


    def forward(self, obs:torch.tensor, memory:torch.tensor, memory_mask:torch.tensor, memory_indices:torch.tensor):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            memory {torch.tensor} -- Episodic memory window
            memory_mask {torch.tensor} -- Mask to prevent the model from attending to the padding
            memory_indices {torch.tensor} -- Indices to select the positional encoding that matches the memory window

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value function: Value
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))
        
        # Forward transformer blocks
        h, memory = self.transformer(h, memory, memory_mask, memory_indices)

        if self.svd_rank_frac is not None:
            approx = self.svd_low_rank_safe(memory, self.svd_rank_frac)
            memory = approx + (memory - approx).detach()

        elif self.tt_rank_frac is not None:
            approx = self.tt_low_rank_safe(memory, self.tt_rank_frac)
            memory = approx + (memory - approx).detach()

        elif self.gauss_filter:
            memory = self.apply_gaussian_filter(memory)

        elif self.laplace_filter:
            memory = self.apply_laplacian_filter(memory)


        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]
        
        return pi, value, memory

    def _get_conv_output(self, shape: tuple) -> int:
        dev = self.conv1.weight.device            # ← cpu
        with torch.no_grad():
            x = torch.zeros((1, *shape), device=dev)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return int(np.prod(x.size()))


    
    def get_grad_norm(self):
        """Returns the norm of the gradients of the model.
        
        Returns:
            {dict} -- Dictionary of gradient norms grouped by layer name
        """
        grads = {}
        if len(self.observation_space_shape) > 1:
            grads["encoder"] = self._calc_grad_norm(self.conv1, self.conv2, self.conv3)  
            
        grads["linear_layer"] = self._calc_grad_norm(self.lin_hidden)
        
        transfomer_blocks = self.transformer.transformer_blocks
        for i, block in enumerate(transfomer_blocks):
            grads["transformer_block_" + str(i)] = self._calc_grad_norm(block)
        
        for i, head in enumerate(self.policy_branches):
            grads["policy_head_" + str(i)] = self._calc_grad_norm(head)
        
        grads["lin_policy"] = self._calc_grad_norm(self.lin_policy)
        grads["value"] = self._calc_grad_norm(self.lin_value, self.value)
        grads["model"] = self._calc_grad_norm(self, self.value)
          
        return grads
    
    def _calc_grad_norm(self, *modules):
        """Computes the norm of the gradients of the given modules.

        Arguments:
            modules {list} -- List of modules to compute the norm of the gradients of.

        Returns:
            {float} -- Norm of the gradients of the given modules. 
        """
        grads = []
        for module in modules:
            for name, parameter in module.named_parameters():
                grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        num_blocks = self.transformer.num_blocks
        memory_len = 1  # начальная длина памяти = 1 (будет расти по ходу эпизода)
        dim = self.memory_layer_size
        return torch.zeros((batch_size, memory_len, num_blocks, dim), device=device)