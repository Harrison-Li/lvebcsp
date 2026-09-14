from __future__ import annotations

from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from lvebcsp.models.condition_encoder import ConditionEncoder
from lvebcsp.models.encoder import UniversalEncoder, EncoderConfig
from lvebcsp.models.sigreg import SIGReg


class Lvebm(nn.Module):
    """Crystal-context JEPA with a shared encoder for partial and complete structures.

    ``context`` batches M representative building-block graphs, each encoded
    independently with intrablock edges and no target lattice. ``block_batch``
    [M] assigns blocks to B crystals; ``multiplicity`` [M] gives their absolute
    copy counts in the target cell. ``target`` batches B complete crystal graphs
    in the same crystal order. All graphs follow UniversalEncoder's input schema.
    """

    def __init__(
        self,
        predictor: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        d_jepa: int = 512,
        stop_gradient: bool = False,
        lambda_sig: float = 0.1,
        sigreg_num_projections: int = 1024,
        crystal_encoder: EncoderConfig | dict[str, Any] | None = None,
        condition_encoder: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.d_jepa = d_jepa
        cfg = crystal_encoder
        if not isinstance(cfg, EncoderConfig):
            cfg = EncoderConfig(**{"output_dim": d_jepa, **(cfg or {})})
        self.context_encoder = UniversalEncoder(cfg)
        self.condition_hidden_dim = (condition_encoder or {}).get("hidden_dim", 256)
        self.condition_encoder = ConditionEncoder(d_jepa, self.condition_hidden_dim)
        self.projector = projector or nn.Identity()
        self.predictor = predictor
        self.pred_proj = pred_proj or nn.Identity()
        self.sigreg = SIGReg(num_proj=sigreg_num_projections)
        self.lambda_sig = lambda_sig
        self.stop_gradient = stop_gradient

    def encode(self, graph: Data | Batch) -> Tensor:
        """Encode partial or complete crystals into projected latents [B, d_jepa]."""
        return self.projector(self.context_encoder(graph))

    def encode_cond(
        self, block_embeddings: Tensor, multiplicity: Tensor, block_batch: Tensor,
    ) -> Tensor:
        """Pool block identities and their copy counts into a condition [B, H]."""
        return self.condition_encoder(block_embeddings, multiplicity, block_batch)

    def predict(self, emb: Tensor, z_emb: Tensor) -> Tensor:
        """Predict complete-crystal latents from representative blocks and multiplicities."""
        if getattr(self.predictor, "takes_condition", False):
            preds = self.predictor(emb, z_emb)
        else:
            preds = self.predictor(torch.cat([emb, z_emb], dim=-1))
        return self.pred_proj(preds)

    def encode_tgt(self, graph: Data | Batch) -> Tensor:
        """Share encoder weights and enable target gradients by default, as in LeWM."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.stop_gradient):
            return self.encode(graph)

    def encode_ctx(
        self,
        graph: Data | Batch,
        multiplicity: Tensor,
        block_batch: Tensor,
    ) -> Tensor:
        """Predict crystal latents from representative blocks and their counts."""
        blocks = self.encode(graph)
        context = global_mean_pool(blocks, block_batch)
        condition = self.encode_cond(blocks, multiplicity, block_batch)
        return self.predict(context, condition)

    def compute_energy(self, query_latent: Tensor, candidate: Data | Batch) -> Tensor:
        """Return one latent energy per candidate crystal."""
        return (query_latent - self.encode_tgt(candidate)).square().mean(dim=-1)

    def train_jepa(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Compute JEPA losses and convert metrics for logging."""
        out = self(batch)
        out["metrics"] = {name: float(value.detach().cpu()) for name, value in out["metrics"].items()}
        return out

    def criterion(
        self,
        pred_emb: Tensor,
        tgt_emb: Tensor,
        *,
        ctx_emb: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """MSE alignment with SIGReg on trainable context embeddings."""
        target = tgt_emb.detach() if self.stop_gradient else tgt_emb
        loss_pred = F.mse_loss(pred_emb, target)
        loss_sig = self.sigreg(ctx_emb)
        loss = loss_pred + self.lambda_sig * loss_sig

        pred_norm = F.normalize(pred_emb.detach(), dim=-1)
        target_norm = F.normalize(target.detach(), dim=-1)
        paired_similarity = (pred_norm * target_norm).sum(dim=-1)
        total_similarity = (pred_norm.sum(dim=0) * target_norm.sum(dim=0)).sum()
        batch_size = pred_emb.shape[0]
        offdiag_similarity = (total_similarity - paired_similarity.sum()) / max(batch_size * (batch_size - 1), 1)
        return loss, {
            "loss_pred": loss_pred,
            "loss_sigreg": loss_sig,
            "sim_diag": paired_similarity.mean(),
            "sim_offdiag": offdiag_similarity,
            "context_std": ctx_emb.detach().std(dim=0, unbiased=False).mean(),
            "target_std": target.detach().std(dim=0, unbiased=False).mean(),
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Train on representative blocks, multiplicities, and complete crystals."""
        blocks = self.encode(batch["context"])
        ctx_emb = global_mean_pool(blocks, batch["block_batch"])
        z_emb = self.encode_cond(blocks, batch["multiplicity"], batch["block_batch"])
        pred_emb = self.predict(ctx_emb, z_emb)
        tgt_emb = self.encode_tgt(batch["target"])
        loss, metrics = self.criterion(pred_emb, tgt_emb, ctx_emb=ctx_emb)
        return {
            "pred_emb": pred_emb,
            "tgt_emb": tgt_emb,
            "loss": loss,
            "metrics": metrics,
        }
