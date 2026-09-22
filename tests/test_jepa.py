"""Crystal graph integration for Lvebm."""

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from lvebcsp.models.encoder import UniversalEncoder, EncoderConfig
from lvebcsp.models.jepa import Lvebm
from lvebcsp.train.train_jepa import build_jepa_from_config, load_jepa_weights
from lvebcsp.data.crystal_dataset import collate_crystal_graphs
from test_crystal_graphs import crystal_dataset


def test_crystal_jepa_forward_backward_and_candidate_energy():
    torch.manual_seed(12)
    model = Lvebm(
        predictor=nn.Linear(24, 16), projector=nn.Linear(12, 16),
        d_jepa=16, condition_encoder={"hidden_dim": 8},
        crystal_encoder=dict(atom_dim=16, dim=16, output_dim=12, heads=4,
                             num_latents=4, local_layers=1, latent_layers=1),
        sigreg_num_projections=8,
    ).eval()
    representatives = Batch.from_data_list([
        Data(z=torch.tensor([z]), pos=torch.zeros(1, 3),
             edge_index=torch.empty(2, 0, dtype=torch.long)) for z in [6, 8, 14]
    ])
    target = Batch.from_data_list([
        Data(z=torch.tensor(z), pos=torch.randn(3, 3),
             edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
             cell=(torch.eye(3) * 5)[None]) for z in [[6, 6, 8], [14, 14, 14]]
    ])
    context = target.clone()
    context.pos += torch.randn_like(context.pos) * .2
    context.pos.requires_grad_()
    batch = dict(context=representatives, target=target, crystal_context=context,
                 block_batch=torch.tensor([0, 0, 1]), multiplicity=torch.tensor([2, 1, 3]))
    with pytest.raises(ValueError, match="requires crystal_context"):
        model.encode_ctx(representatives, batch["multiplicity"], batch["block_batch"])
    with pytest.raises(ValueError, match="requires crystal_context"):
        model({key: value for key, value in batch.items() if key != "crystal_context"})
    inputs = []
    hook = model.context_encoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    out = model(batch)
    hook.remove()
    assert inputs[0] is representatives and inputs[1] is context and inputs[2] is target
    assert isinstance(model.context_encoder, UniversalEncoder)
    assert out["ctx_emb"].shape == out["pred_emb"].shape == out["tgt_emb"].shape == (2, 4, 16)
    assert torch.isfinite(out["loss"]) and out["tgt_emb"].requires_grad
    target_grad = torch.autograd.grad(out["metrics"]["loss_pred"], out["tgt_emb"], retain_graph=True)[0]
    assert torch.isfinite(target_grad).all() and target_grad.abs().sum() > 0
    out["loss"].backward()
    for module in (model.context_encoder, model.condition_encoder, model.predictor):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert torch.isfinite(context.pos.grad).all() and context.pos.grad.norm() > 0
    assert (model.context_encoder.local_encoder.lattice_presence_embedding.weight.grad.norm(dim=1) > 0).all()
    torch.testing.assert_close(model.encode_ctx(representatives, batch["multiplicity"], batch["block_batch"],
                                               crystal_context=context), out["pred_emb"])
    torch.testing.assert_close(model.encode_tgt(target), out["tgt_emb"])
    torch.testing.assert_close(model.compute_energy(out["pred_emb"], target),
                               (out["pred_emb"] - out["tgt_emb"]).square().mean(dim=(-2, -1)))
    # Changing Ct alone affects supervision, never the prediction input.
    changed = target.clone()
    changed.pos[0] += .3
    other = model({**batch, "target": changed})
    torch.testing.assert_close(other["pred_emb"], out["pred_emb"])
    assert not torch.allclose(other["tgt_emb"][0], out["tgt_emb"][0])
    # Changing Cc affects only that crystal's prediction.
    changed = context.clone()
    changed.pos = changed.pos.detach().clone()
    changed.pos[0] += .3
    other = model({**batch, "crystal_context": changed})
    assert not torch.allclose(other["pred_emb"][0], out["pred_emb"][0])
    torch.testing.assert_close(other["pred_emb"][1], out["pred_emb"][1])
    torch.testing.assert_close(other["tgt_emb"], out["tgt_emb"])
    assert all(isinstance(value, float) for value in model.train_jepa(batch)["metrics"].values())


@pytest.mark.parametrize("loss_overrides", [{}, {"mu_diff": .01}], ids=["default", "legacy_differences"])
def test_alignment_and_sigreg_backpropagate_to_both_branches(loss_overrides):
    model = Lvebm(
        predictor=nn.Identity(), d_jepa=16,
        crystal_encoder=EncoderConfig(atom_dim=16, dim=16, output_dim=16,
                                      heads=4, local_layers=1, latent_layers=1),
        sigreg_num_projections=8,
        **loss_overrides,
    )
    pred = torch.randn(3, 4, 16, requires_grad=True)
    target = torch.randn(3, 4, 16, requires_grad=True)
    context = torch.randn(3, 4, 16, requires_grad=True)
    loss, metrics = model.criterion(pred, target, ctx_emb=context)
    torch.testing.assert_close(metrics["loss_pred"], (pred - target).square().mean())
    expected_mu = loss_overrides.get("mu_diff", 0.)
    assert model.mu_diff == expected_mu
    torch.testing.assert_close(loss, metrics["loss_pred"] + 0.1 * metrics["loss_sigreg"]
                               + expected_mu * metrics["loss_sigreg_diff"])
    components = [
        (metrics["loss_pred"], (pred, target)),
        (metrics["loss_sigreg"], (context, target)),
        (loss, (pred, context, target)),
    ]
    if expected_mu > 0:
        components.append((metrics["loss_sigreg_diff"], (context, target)))
    else:
        assert metrics["loss_sigreg_diff"] == 0
        assert not metrics["loss_sigreg_diff"].requires_grad
    for component, inputs in components:
        for grad in torch.autograd.grad(component, inputs, retain_graph=True):
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0
    for name in ("sim_prediction", "sim_diag", "sim_offdiag", "context_std", "target_std"):
        assert not metrics[name].requires_grad


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
    crystal = Batch.from_data_list(graphs[:2])
    expected = model.encode_ctx(Batch.from_data_list(graphs), counts, owners, crystal_context=crystal)
    permutation = torch.tensor([2, 1, 0])
    reordered = model.encode_ctx(
        Batch.from_data_list([graphs[i] for i in permutation]),
        counts[permutation], owners[permutation], crystal_context=crystal,
    )
    torch.testing.assert_close(reordered, expected)
    swapped = model.encode_ctx(Batch.from_data_list(graphs), counts[[1, 0, 2]], owners, crystal_context=crystal)
    assert not torch.allclose(swapped[0], expected[0])
    torch.testing.assert_close(swapped[1], expected[1])
    scaled = model.encode_ctx(Batch.from_data_list(graphs), 2 * counts, owners, crystal_context=crystal)
    assert not torch.allclose(scaled, expected)


def test_target_encoder_preserves_autograd_context():
    model = Lvebm(predictor=nn.Linear(272, 16), d_jepa=16,
                  crystal_encoder=dict(atom_dim=16, dim=16, heads=4, local_layers=1, latent_layers=1),
                  sigreg_num_projections=8)
    graph = Data(z=torch.tensor([6, 8]), pos=torch.randn(2, 3),
                 edge_index=torch.tensor([[0, 1], [1, 0]]))
    for training in (True, False):
        model.train(training)
        encoded = model.encode_tgt(graph)
        assert encoded.requires_grad
        grad = torch.autograd.grad(encoded.square().mean(), model.context_encoder.output.weight)[0]
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
        with torch.no_grad():
            assert not model.encode_tgt(graph).requires_grad


@pytest.mark.parametrize("legacy_key", ["stop_gradient", "stop_gradient_target"])
def test_legacy_config_cannot_disable_target_gradients(legacy_key):
    config = dict(
        d_jepa=16,
        crystal_encoder=dict(atom_dim=16, dim=16, num_latents=4, heads=4,
                             local_layers=1, latent_layers=1),
        condition_encoder=dict(hidden_dim=8), loss=dict(sigreg_num_projections=8),
    )
    config[legacy_key] = True
    model = build_jepa_from_config(config)
    graph = Data(z=torch.tensor([6, 8]), pos=torch.randn(2, 3),
                 edge_index=torch.tensor([[0, 1], [1, 0]]))
    target = model.encode_tgt(graph)
    pred = torch.zeros_like(target, requires_grad=True)
    context = torch.randn_like(target, requires_grad=True)
    _, metrics = model.criterion(pred, target, ctx_emb=context)
    grad = torch.autograd.grad(metrics["loss_pred"], model.context_encoder.output.weight)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_both_sigreg_terms_keep_crystals_as_samples():
    model = Lvebm(predictor=nn.Identity(), d_jepa=4,
                  crystal_encoder=dict(atom_dim=8, dim=8, num_latents=2, heads=2,
                                       local_layers=1, latent_layers=1),
                  sigreg_num_projections=8, mu_diff=.01)
    # Token identity varies, but every crystal has exactly the same context.
    context = torch.randn(1, 2, 4).expand(3, -1, -1)
    target = torch.randn(3, 2, 4)
    pred = -target  # Prediction agreement must stay separate from context/target agreement.
    samples = []
    hook = model.sigreg.register_forward_pre_hook(lambda module, args: samples.append(args[0]))
    _, metrics = model.criterion(pred, target, ctx_emb=context)
    hook.remove()
    assert [tuple(x.shape) for x in samples] == [(2, 3, 4)] * 4
    torch.testing.assert_close(samples[0], context.transpose(0, 1))
    torch.testing.assert_close(samples[1], target.transpose(0, 1))
    assert all(torch.unique(slot, dim=0).shape[0] == 1 for slot in samples[0])
    assert all(torch.unique(slot, dim=0).shape[0] == 1 for slot in samples[2])
    assert metrics["context_std"] == 0
    similarities = torch.stack([
        torch.stack([nn.functional.cosine_similarity(p.reshape(-1), t.reshape(-1), dim=0) for t in target])
        for p in context
    ])
    torch.testing.assert_close(metrics["sim_diag"], similarities.diag().mean())
    torch.testing.assert_close(metrics["sim_offdiag"], similarities[~torch.eye(3, dtype=torch.bool)].mean())
    assert metrics["sim_prediction"] == pytest.approx(-1.)


def test_difference_sigreg_penalizes_copied_tokens_on_either_branch():
    torch.manual_seed(12)
    model = Lvebm(predictor=nn.Identity(), d_jepa=16,
                  crystal_encoder=dict(atom_dim=8, dim=8, num_latents=4, heads=2,
                                       local_layers=1, latent_layers=1),
                  sigreg_num_projections=64, mu_diff=.01)
    independent = torch.randn(128, 4, 16)
    copied = independent[:, :1].expand_as(independent)
    torch.manual_seed(42)
    _, baseline = model.criterion(independent, independent, ctx_emb=independent)
    for context, target in [(copied, independent), (independent, copied), (copied, copied)]:
        torch.manual_seed(42)
        _, metrics = model.criterion(target, target, ctx_emb=context)
        assert metrics["loss_sigreg_diff"] > 10 * baseline["loss_sigreg_diff"]


@pytest.mark.parametrize("loss_overrides, slots", [({}, 4), ({"mu_diff": 0.}, 4), ({"mu_diff": .01}, 1)])
def test_difference_sigreg_skips_default_disabled_or_single_slot(loss_overrides, slots):
    model = build_jepa_from_config(dict(
        d_jepa=4, crystal_encoder=dict(atom_dim=8, dim=8, num_latents=slots, heads=2,
                                      local_layers=1, latent_layers=1),
        loss=dict(lambda_sig=.2, sigreg_num_projections=8, **loss_overrides),
    ))
    assert model.mu_diff == loss_overrides.get("mu_diff", 0.)
    embeddings = torch.randn(3, slots, 4, requires_grad=True)
    calls = []
    hook = model.sigreg.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    loss, metrics = model.criterion(embeddings, embeddings, ctx_emb=embeddings)
    hook.remove()
    assert len(calls) == 2
    assert metrics["loss_sigreg_diff"] == 0
    torch.testing.assert_close(loss, .2 * metrics["loss_sigreg"])
    loss.backward()
    assert torch.isfinite(embeddings.grad).all() and embeddings.grad.abs().sum() > 0


@pytest.mark.parametrize("head", ["mlp", "gated_mlp", "adaln_mlp", "predictor"])
def test_predictor_heads_preserve_tokens(head):
    config = dict(d_jepa=16, crystal_encoder=dict(atom_dim=16, dim=16, heads=4,
                  num_latents=4, local_layers=1, latent_layers=1),
                  condition_encoder=dict(hidden_dim=8),
                  predictor=dict(type=head, hidden_dim=32), loss=dict(sigreg_num_projections=8))
    model = build_jepa_from_config(config)
    context = torch.randn(2, 4, 16, requires_grad=True)
    condition = torch.randn(2, 4, 8, requires_grad=True)
    prediction = model.predict(context, condition)
    assert prediction.shape == (2, 4, 16)
    prediction.square().mean().backward()
    assert torch.isfinite(context.grad).all()
    assert torch.isfinite(condition.grad).all()


def test_loading_pooled_checkpoint_only_discards_removed_readout():
    config = dict(d_jepa=16, crystal_encoder=dict(atom_dim=16, dim=16, heads=4,
                  num_latents=4, local_layers=1, latent_layers=1),
                  condition_encoder=dict(hidden_dim=8), predictor=dict(type="mlp"))
    source, restored = build_jepa_from_config(config), build_jepa_from_config({**config, "context_translation_std": .25})
    weights = {**source.state_dict(), "context_encoder.readout_query": torch.randn(1, 1, 16),
               "context_encoder.readout.q_norm.weight": torch.ones(16)}
    load_jepa_weights(restored, weights)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[name])
    del weights["context_encoder.latent_queries"]
    with pytest.raises(RuntimeError, match="latent_queries"):
        load_jepa_weights(restored, weights)


def test_legacy_lattice_initialization_requires_training_and_stays_strict():
    config = dict(d_jepa=16, crystal_encoder=dict(atom_dim=16, dim=16, heads=4,
                  num_latents=4, local_layers=1, latent_layers=1),
                  condition_encoder=dict(hidden_dim=8), predictor=dict(type="mlp"))
    source, restored = build_jepa_from_config(config), build_jepa_from_config(config)
    prefixes = ("context_encoder.local_encoder.lattice_embedding.", "context_encoder.local_encoder.lattice_presence_embedding.")
    legacy = {key: value for key, value in source.state_dict().items() if not key.startswith(prefixes)}
    initial_lattice = {key: value.clone() for key, value in restored.state_dict().items() if key.startswith(prefixes)}
    with pytest.raises(RuntimeError, match="JEPA --init-from"):
        load_jepa_weights(restored, legacy)
    with pytest.warns(UserWarning, match="Initializing new lattice embeddings"):
        load_jepa_weights(restored, legacy, initialize_lattice=True)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, initial_lattice[key] if key in initial_lattice else legacy[key])
    assert not any(key.startswith(prefixes) for key in legacy)

    partial = {**legacy, "context_encoder.local_encoder.lattice_presence_embedding.weight": initial_lattice[
        "context_encoder.local_encoder.lattice_presence_embedding.weight"]}
    with pytest.raises(RuntimeError, match="lattice_embedding"):
        load_jepa_weights(restored, partial, initialize_lattice=True)
    del legacy["context_encoder.latent_queries"]
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="latent_queries"):
        load_jepa_weights(restored, legacy, initialize_lattice=True)


def test_old_atom_heads_are_discarded_only_for_explicit_warm_start():
    config = dict(d_jepa=8, crystal_encoder=dict(atom_dim=8, dim=8, heads=2))
    model = build_jepa_from_config(config)
    weights = {**model.state_dict(), "atom_queries.weight": torch.randn(119, 8),
               "atom_decoder.legacy.weight": torch.randn(8, 8), "atom_output.weight": torch.randn(8, 8)}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_jepa_weights(model, weights)
    with pytest.warns(UserWarning, match="Discarding"):
        load_jepa_weights(model, weights, initialize_predictor=True)
    assert not any(name.startswith("atom_") for name in model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name])
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="unrelated"):
        load_jepa_weights(model, {**weights, "unrelated": torch.zeros(1)}, initialize_predictor=True)
