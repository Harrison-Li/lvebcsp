import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or output_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn(),
            norm_fn,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.mlp(x)


class GatedMLP(nn.Module):
    """
    Using a gating mechanism to control the flow of information.
    """
    def __init__(
        self, 
        input_dim,
        hidden_dim, 
        output_dim,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn(),
            norm_fn,
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        return self.mlp(x) * self.gate(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer-norm shift and scale."""

    return x * (1 + scale) + shift


class AdaLNMLP(nn.Module):
    """Conditional AdaLN-zero MLP predictor without attention."""

    takes_condition = True

    def __init__(
        self,
        input_dim,
        cond_dim,
        hidden_dim=None,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        norm_eps=1.0e-6,
        act_fn=nn.GELU,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or output_dim
        self.input_dim = int(input_dim)
        self.condition_dim = int(cond_dim)
        self.output_dim = int(output_dim)
        self.norm = nn.LayerNorm(self.input_dim, elementwise_affine=False, eps=norm_eps)
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            act_fn(),
            norm_fn,
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.residual_proj = (
            nn.Linear(self.input_dim, self.output_dim)
            if self.input_dim != self.output_dim
            else nn.Identity()
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.condition_dim, 2 * self.input_dim + self.output_dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        squeeze_t = x.ndim == 2
        if squeeze_t:
            x = x.unsqueeze(1)
            c = c.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"AdaLNMLP expects x shape [B, D] or [B, T, D], got {tuple(x.shape)}")
        if c.ndim != 3:
            raise ValueError(f"AdaLNMLP expects c shape [B, D] or [B, T, D], got {tuple(c.shape)}")

        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        out = self.residual_proj(x) + gate * self.mlp(modulate(self.norm(x), shift, scale))
        return out.squeeze(1) if squeeze_t else out
