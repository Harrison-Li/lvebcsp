from __future__ import annotations

from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from lvebcsp.models.condition_encoder import ConditionEncoder
from lvebcsp.models.encoder import UniversalEncoder, EncoderConfig, build_peak_encoder
from lvebcsp.models.sigreg import SIGReg


class Lvebm(nn.Module):
    """Predict clean crystal tokens from a perturbed crystal and observations.

    ``crystal_context`` is Cc: all atoms of Ct, with a rigid transform per block.
    ``target`` is the untouched Ct. ``context`` holds isolated representative
    blocks; those and ``multiplicity`` supply the predictor's condition only.
    ``block_batch`` assigns each representative to its crystal.
    Optional target-PXRD peaks condition the predictor alone, replacing or adding
    to representatives. Inference must provide these observed peaks explicitly.

    The shared encoder produces [B, T, D] tokens for Cc and Ct. Both branches
    receive MSE/SIGReg gradients (LeWM); there is no EMA or stop-gradient target.
    """

    prediction_target = "crystal_tokens"

    def __init__(
        self,
        predictor: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        d_jepa: int = 512,
        lambda_sig: float = 0.1,
        sigreg_num_projections: int = 1024,
        crystal_encoder: EncoderConfig | dict[str, Any] | None = None,
        condition_encoder: dict[str, Any] | None = None,
        mu_diff: float = 0.0,
        condition_source: str = "conformer",
        peak_encoder: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.d_jepa = d_jepa
        cfg = crystal_encoder
        if not isinstance(cfg, EncoderConfig):
            cfg = EncoderConfig(**{"output_dim": d_jepa, **(cfg or {})})
        self.context_encoder = UniversalEncoder(cfg)
        self.condition_hidden_dim = (condition_encoder or {}).get("hidden_dim", 256)
        if condition_source not in {"conformer", "pxrd", "combined"}:
            raise ValueError("condition_source must be conformer, pxrd, or combined")
        self.condition_source = condition_source
        self.condition_encoder = (ConditionEncoder(d_jepa, self.condition_hidden_dim)
                                  if condition_source != "pxrd" else None)
        self.peak_encoder = (build_peak_encoder(peak_encoder, d_jepa=self.condition_hidden_dim)
                             if condition_source != "conformer" else None)
        self.projector = projector or nn.Identity()
        self.predictor = predictor
        self.pred_proj = pred_proj or nn.Identity()
        self.sigreg = SIGReg(num_proj=sigreg_num_projections)
        if not 0 <= lambda_sig < float("inf"):
            raise ValueError("lambda_sig must be finite and >= 0")
        self.lambda_sig = lambda_sig
        if not 0 <= mu_diff < float("inf"):
            raise ValueError("mu_diff must be finite and >= 0")
        self.mu_diff = mu_diff

    def encode(self, graph: Data | Batch, *, return_atoms: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Encode crystal tokens, optionally returning all local atom features."""
        encoded = self.context_encoder(graph, return_atoms=return_atoms)
        if return_atoms:
            tokens, atoms = encoded
            return self.projector(tokens), atoms
        return self.projector(encoded)

    def encode_cond(
        self, block_embeddings: Tensor, multiplicity: Tensor, block_batch: Tensor,
        size: int | None = None,
    ) -> Tensor:
        """Aggregate representatives and counts into condition tokens [B, T, H]."""
        if self.condition_encoder is None:
            raise ValueError("This PXRD-only model has no conformer conditioner")
        return self.condition_encoder(block_embeddings, multiplicity, block_batch, size=size)  # [B, T, H]

    def predict(self, emb: Tensor, z_emb: Tensor) -> Tensor:
        """Map context [B, T, D] and condition [B, T, H] to clean [B, T, D]."""
        # emb: [B, T, D]; z_emb: [B, T, H].
        if getattr(self.predictor, "takes_condition", False):
            preds = self.predictor(emb, z_emb)  # [B, T, D]
        else:
            preds = self.predictor(torch.cat([emb, z_emb], dim=-1))  # [B, T, D + H] -> [B, T, D]
        return self.pred_proj(preds)

    def encode_tgt(self, graph: Data | Batch, *, return_atoms: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Share encoder weights and preserve target gradients, as in LeWM."""
        return self.encode(graph, return_atoms=return_atoms)

    def encode_ctx(
        self,
        graph: Data | Batch,
        multiplicity: Tensor,
        block_batch: Tensor,
        *,
        crystal_context: Data | Batch | None = None,
        peak_d: Tensor | None = None,
        peak_i: Tensor | None = None,
        peak_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict using Cc and supplied observations, without reading Ct's graph."""
        context, condition = self.encode_context(graph, multiplicity, block_batch, crystal_context,
                                                  peak_d=peak_d, peak_i=peak_i, peak_mask=peak_mask)
        return self.predict(context, condition)

    def encode_context(
        self, graph: Data | Batch, multiplicity: Tensor, block_batch: Tensor,
        crystal_context: Data | Batch | None,
        *, peak_d: Tensor | None = None, peak_i: Tensor | None = None, peak_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode the crystal view and the configured observations separately."""
        if crystal_context is None:
            raise ValueError("Crystal JEPA requires crystal_context, including when perturbations are zero.")
        blocks = self.encode(graph) if self.condition_encoder is not None else None  # [M, T, D]
        context = self.encode(crystal_context)  # [B, T, D]
        condition = (self.encode_cond(blocks, multiplicity, block_batch, size=context.size(0))
                     if blocks is not None else None)  # [B, T, H]
        if self.peak_encoder is not None:
            if peak_d is None or peak_i is None or peak_mask is None:
                raise ValueError(f"condition_source={self.condition_source} requires target peak_d, peak_i, peak_mask")
            pxrd = self.peak_encoder(peak_d, peak_i, peak_mask)
            if pxrd.size(0) != context.size(0):
                raise ValueError("PXRD batch size differs from crystal context batch size")
            pxrd = pxrd[:, None].expand(-1, context.size(1), -1)  # [B, T, H]
            # Additive fusion keeps the predictor width and objective unchanged.
            condition = pxrd if condition is None else condition + pxrd
        return context, condition

    def compute_energy(self, query_latent: Tensor, candidate: Data | Batch) -> Tensor:
        """Return one latent energy per candidate crystal."""
        return (query_latent - self.encode_tgt(candidate)).square().flatten(1).mean(dim=-1)  # [B, T, D] -> [B, T * D] -> [B]

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
        """Prediction MSE and per-slot SIGReg; differences are a legacy opt-in."""
        if pred_emb.shape != tgt_emb.shape:
            raise ValueError(f"Prediction and target shapes differ: {pred_emb.shape} vs {tgt_emb.shape}.")
        loss_pred = F.mse_loss(pred_emb, tgt_emb)

        # Each slot defines a distribution; crystals are the statistical samples.
        loss_sig_ctx = self.sigreg(ctx_emb.transpose(0, 1))  # [T, B, D] -> scalar
        loss_sig_tgt = self.sigreg(tgt_emb.transpose(0, 1))  # [T, B, D] -> scalar
        loss_sig = 0.5 * loss_sig_ctx + 0.5 * loss_sig_tgt  # scalar
        loss_diff = loss_pred.new_zeros(())
        if self.mu_diff > 0 and ctx_emb.size(1) > 1:
            # One random cycle pairs distinct slots; crystals remain the sample axis.
            order = torch.randperm(ctx_emb.size(1), device=ctx_emb.device)
            for emb in (ctx_emb, tgt_emb):
                paired = emb[:, order]  # [B, T, D]; same pairs on both branches
                delta = (paired - paired.roll(1, dims=1)) / (2 ** 0.5)
                loss_diff = loss_diff + 0.5 * self.sigreg(delta.transpose(0, 1))
        loss = loss_pred + self.lambda_sig * loss_sig + self.mu_diff * loss_diff

        # Compare flattened crystal slots in FP32; detach diagnostics only.
        context_norm = F.normalize(ctx_emb.detach().float().flatten(1), dim=-1)  # [B, T * D]
        target_norm = F.normalize(tgt_emb.detach().float().flatten(1), dim=-1)
        paired_similarity = (context_norm * target_norm).sum(dim=-1)  # Same crystal: context_i vs target_i.
        total_similarity = (context_norm.sum(dim=0) * target_norm.sum(dim=0)).sum()
        batch_size = ctx_emb.shape[0]
        offdiag_similarity = (total_similarity - paired_similarity.sum()) / max(batch_size * (batch_size - 1), 1)
        # Prediction agreement is separate from raw context/target slot agreement.
        prediction_similarity = F.cosine_similarity(
            pred_emb.detach().float().flatten(1), tgt_emb.detach().float().flatten(1), dim=-1,
        )
        return loss, {
            "loss_pred": loss_pred,
            "loss_sigreg": loss_sig,
            "loss_sigreg_diff": loss_diff,
            "sim_diag": paired_similarity.mean(),
            "sim_prediction": prediction_similarity.sum() / max(prediction_similarity.numel(), 1),
            "sim_offdiag": offdiag_similarity,
            "context_std": ctx_emb.detach().std(dim=0, unbiased=False).mean(),
            "target_std": tgt_emb.detach().std(dim=0, unbiased=False).mean(),
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Cc -> context tokens -> prediction; Ct -> clean target tokens -> loss."""
        ctx_emb, condition = self.encode_context(
            batch["context"], batch["multiplicity"], batch["block_batch"], batch.get("crystal_context"),
            peak_d=batch.get("peak_d"), peak_i=batch.get("peak_i"), peak_mask=batch.get("peak_mask"),
        )
        pred_emb = self.predict(ctx_emb, condition)
        tgt_emb = self.encode_tgt(batch["target"])
        loss, metrics = self.criterion(pred_emb, tgt_emb, ctx_emb=ctx_emb)
        return {
            "ctx_emb": ctx_emb,
            "pred_emb": pred_emb,
            "tgt_emb": tgt_emb,
            "loss": loss,
            "metrics": metrics,
        }
