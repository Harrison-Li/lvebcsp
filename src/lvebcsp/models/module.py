import torch
from torch import nn
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from lvebcsp.models.layers import MLP

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        condition_dim=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        condition_dim = condition_dim or input_dim

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(condition_dim, hidden_dim)
            if condition_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Predictor(nn.Module):
    """Predictor for embedding difference prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        condition_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.takes_condition = True
        condition_dim = condition_dim or input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            condition_dim=condition_dim,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        squeeze_time = x.ndim == 2
        if squeeze_time:
            x = x.unsqueeze(1)
            c = c.unsqueeze(1)
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        if squeeze_time:
            x = x.squeeze(1)
        return x



class AtomEmbedding(nn.Module):
    """Encode an atom by its atomic number using a learnable embedding layer."""

    def __init__(self, fea_dim: int, max_num_elements: int = 119) -> None:
        """Initialize the Atom featurizer.

        Args:
            atom_feature_dim (int): dimension of atomic embedding.
            max_num_elements (int): maximum number of elements in the dataset.
                Default = 94
        """
        super().__init__()
        self.embedding = nn.Embedding(max_num_elements, fea_dim)

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        """Convert the structure to a atom embedding tensor.

        Args:
            atomic_numbers (Tensor): [n_atom, 1].

        Returns:
            atom_fea (Tensor): atom embeddings [n_atom, atom_feature_dim].
        """
        return self.embedding(atomic_numbers)


class Fourier(nn.Module):
    """Fourier Expansion for angle features."""

    def __init__(self, *, order: int = 5, learnable: bool = False) -> None:
        """Initialize the Fourier expansion.

        Args:
            order (int): the maximum order
                Default = 5
            learnable (bool): whether to set the frequencies as learnable parameters
                Default = False
        """
        super().__init__()
        self.order = order
        # Initialize frequencies at canonical
        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.arange(1, order + 1, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "frequencies", torch.arange(1, order + 1, dtype=torch.float32)
            )

    def forward(self, x: Tensor) -> Tensor:
        """Apply Fourier expansion to a feature Tensor."""
        result = x.new_zeros(x.shape[0], 1 + 2 * self.order)
        result[:, 0] = 1 / torch.sqrt(torch.tensor([2]))
        tmp = x.unsqueeze(-1) * self.frequencies # [B, 3, 1] * [N]
        const = x.new_full((*x.shape, 1), 1.0 / np.sqrt(2.0))
        out = torch.cat([const, torch.sin(tmp), torch.cos(tmp)], dim=-1)
        return out



class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis expansion for scalar inputs.
    """

    def __init__(
        self,
        r_max: float,
        num_basis: int = 128,
        trainable: bool = False,
    ):
        super().__init__()

        if num_basis < 2:
            raise ValueError("num_basis must be at least 2.")
        if r_max <= 0:
            raise ValueError("r_max must be positive.")

        centers = torch.linspace(
            0.0,
            r_max,
            steps=num_basis,
            dtype=torch.get_default_dtype(),
        )

        if trainable:
            self.centers = torch.nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

        spacing = r_max / (num_basis - 1)
        self.coeff = -0.5 / spacing**2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with arbitrary shape [...]

        Returns:
            Gaussian basis tensor with shape [..., num_basis]
        """
        diff = x.unsqueeze(-1) - self.centers
        return torch.exp(self.coeff * diff.square())
