"""Crystal graph integration for Lvebm."""

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from lvebcsp.models.encoder import UniversalEncoder, EncoderConfig
from lvebcsp.models.jepa import Lvebm


def test_crystal_jepa_forward_backward_and_candidate_energy():
    torch.manual_seed(12)
    model = Lvebm(
        predictor=nn.Linear(24, 16), projector=nn.Linear(12, 16),
        d_jepa=16, condition_encoder={"hidden_dim": 8},
        crystal_encoder=dict(atom_dim=16, dim=16, output_dim=12, heads=4,
                             num_latents=4, local_layers=1, latent_layers=1),
        sigreg_num_projections=8,
    ).eval()
    context = Batch.from_data_list([
        Data(z=torch.tensor([z]), pos=torch.zeros(1, 3),
             edge_index=torch.empty(2, 0, dtype=torch.long),
             geometry_known=torch.tensor([False]))
        for z in [6, 8, 14]
    ])
    target = Batch.from_data_list([
        Data(z=torch.tensor(z), pos=torch.randn(3, 3),
             edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
             cell=(torch.eye(3) * 5)[None])
        for z in [[6, 6, 8], [14, 14, 14]]
    ])
    batch = dict(context=context, target=target,
                 block_batch=torch.tensor([0, 0, 1]), multiplicity=torch.tensor([2, 1, 3]))
    out = model(batch)
    assert isinstance(model.context_encoder, UniversalEncoder)
    assert out["pred_emb"].shape == out["tgt_emb"].shape == (2, 16)
    assert torch.isfinite(out["loss"])
    assert out["tgt_emb"].requires_grad
    target_grad = torch.autograd.grad(
        out["metrics"]["loss_pred"], out["tgt_emb"], retain_graph=True,
    )[0]
    assert torch.isfinite(target_grad).all()
    assert target_grad.abs().sum() > 0
    out["loss"].backward()
    assert torch.isfinite(model.context_encoder.output.weight.grad).all()
    assert torch.isfinite(model.predictor.weight.grad).all()
    torch.testing.assert_close(model.encode_ctx(context, batch["multiplicity"], batch["block_batch"]), out["pred_emb"])
    torch.testing.assert_close(model.encode_tgt(target), out["tgt_emb"])
    torch.testing.assert_close(
        model.compute_energy(out["pred_emb"], target),
        (out["pred_emb"] - out["tgt_emb"]).square().mean(dim=-1),
    )
    assert all(isinstance(value, float) for value in model.train_jepa(batch)["metrics"].values())


@pytest.mark.parametrize("stop_gradient", [False, True])
def test_target_alignment_gradient_and_sigreg(stop_gradient):
    model = Lvebm(
        predictor=nn.Identity(), d_jepa=16, stop_gradient=stop_gradient,
        crystal_encoder=EncoderConfig(atom_dim=16, dim=16, output_dim=16,
                                      heads=4, local_layers=1, latent_layers=1),
        sigreg_num_projections=8,
    )
    pred = torch.randn(3, 16, requires_grad=True)
    target = torch.randn(3, 16, requires_grad=True)
    context = torch.randn(3, 16, requires_grad=True)
    loss, metrics = model.criterion(pred, target, ctx_emb=context)
    torch.testing.assert_close(metrics["loss_pred"], (pred - target).square().mean())
    grad = torch.autograd.grad(metrics["loss_pred"], target, allow_unused=True, retain_graph=True)[0]
    if stop_gradient:
        assert grad is None
    else:
        assert grad.abs().sum() > 0
    target_grad, context_grad = torch.autograd.grad(loss, (target, context), allow_unused=True)
    assert (target_grad is None) == stop_gradient
    assert torch.isfinite(context_grad).all()
    assert context_grad.abs().sum() > 0


def test_block_order_invariance_and_multiplicity_pairing():
    torch.manual_seed(21)
    model = Lvebm(
        predictor=nn.Linear(24, 16), d_jepa=16,
        condition_encoder={"hidden_dim": 8},
        crystal_encoder=dict(atom_dim=16, dim=16, heads=4, num_latents=4,
                             local_layers=1, latent_layers=1),
        sigreg_num_projections=8,
    ).eval()
    graphs = [Data(z=torch.tensor([z]), pos=torch.zeros(1, 3),
                   edge_index=torch.empty(2, 0, dtype=torch.long)) for z in [6, 8, 14]]
    counts = torch.tensor([2, 5, 3])
    owners = torch.tensor([0, 0, 1])
    expected = model.encode_ctx(Batch.from_data_list(graphs), counts, owners)
    permutation = torch.tensor([2, 1, 0])
    reordered = model.encode_ctx(
        Batch.from_data_list([graphs[i] for i in permutation]),
        counts[permutation], owners[permutation],
    )
    torch.testing.assert_close(reordered, expected)
    swapped = model.encode_ctx(Batch.from_data_list(graphs), counts[[1, 0, 2]], owners)
    assert not torch.allclose(swapped[0], expected[0])
    torch.testing.assert_close(swapped[1], expected[1])
    scaled = model.encode_ctx(Batch.from_data_list(graphs), 2 * counts, owners)
    assert not torch.allclose(scaled, expected)


def test_stopped_target_tracks_shared_weight_updates():
    model = Lvebm(predictor=nn.Linear(272, 16), d_jepa=16, stop_gradient=True,
                  crystal_encoder=dict(atom_dim=16, dim=16, heads=4, local_layers=1, latent_layers=1),
                  sigreg_num_projections=8)
    graph = Data(z=torch.tensor([6, 8]), pos=torch.randn(2, 3),
                 edge_index=torch.tensor([[0, 1], [1, 0]]))
    before = model.encode_tgt(graph)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.encode(graph).square().mean().backward()
    optimizer.step()
    after = model.encode_tgt(graph)
    assert not before.requires_grad and not after.requires_grad
    assert not torch.allclose(before, after)
