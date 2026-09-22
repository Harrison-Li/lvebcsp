"""Tests for UniversalEncoder and PeakEncoder."""

import sys
from pathlib import Path

# Ensure src/ is in sys.path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pytest
import torch
from torch_geometric.data import Batch, Data

from lvebcsp.common.utils import lattice_params_to_matrix_torch
from lvebcsp.models.encoder import (
    EncoderConfig,
    LocalAtomTransformer,
    UniversalEncoder,
)


def test_crystal_encoder_pyg_style_batch():
    cfg = EncoderConfig(
        atom_dim=64,
        dim=128,
        output_dim=256,
        num_latents=32,
        local_layers=2,
        latent_layers=2,
        heads=8,
        radial_basis=16,
        cutoff=6.0,
    )
    encoder = UniversalEncoder(cfg)

    # 2 graphs in PyG format
    # Graph 0 has 3 atoms, Graph 1 has 2 atoms
    z = torch.tensor([1, 6, 8, 14, 26], dtype=torch.long)
    pos = torch.randn(5, 3)
    cells = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 5.0
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 4],  # src
        [1, 0, 2, 1, 4, 3],  # dst
    ], dtype=torch.long)
    edge_shifts = torch.zeros(6, 3, dtype=torch.long)

    pyg_data = Batch.from_data_list([
        Data(z=z[:3], pos=pos[:3], cell=cells[:1],
             edge_index=edge_index[:, :4], edge_shifts=edge_shifts[:4]),
        Data(z=z[3:], pos=pos[3:], cell=cells[1:],
             edge_index=edge_index[:, 4:] - 3, edge_shifts=edge_shifts[4:]),
    ])

    embedding = encoder(pyg_data)
    assert embedding.shape == (2, 32, 256)
    assert torch.isfinite(embedding).all()



def test_crystal_encoder_isolated_atoms():
    cfg = EncoderConfig(
        atom_dim=32,
        dim=64,
        output_dim=128,
        num_latents=32,
        local_layers=1,
        latent_layers=1,
        heads=4,
        radial_basis=8,
        cutoff=5.0,
    )
    encoder = UniversalEncoder(cfg)

    # 3 atoms with 0 edges
    batch = Data(
        z=torch.tensor([6, 7, 8], dtype=torch.long),
        pos=torch.randn(3, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )

    out = encoder(batch)
    assert out.shape == (1, 32, 128)
    assert not torch.isnan(out).any()


def test_peak_encoder_compatibility():
    from lvebcsp.models.encoder import PeakEncoder

    encoder = PeakEncoder(d_jepa=128)
    peak_d = torch.rand(4, 20)
    peak_i = torch.rand(4, 20)
    mask = torch.ones(4, 20, dtype=torch.bool)

    out = encoder(peak_d, peak_i, mask)
    assert out.shape == (4, 128)
    assert not torch.isnan(out).any()


def test_pyg_batch_matches_individual_graphs_and_backpropagates():
    torch.manual_seed(7)
    encoder = UniversalEncoder(EncoderConfig(
        atom_dim=16, dim=32, output_dim=16, num_latents=4,
        local_layers=1, latent_layers=1, heads=4, radial_basis=8,
        atom_feature_dim=2,
    )).eval()
    graphs = [
        Data(
            z=torch.tensor([6, 8, 1]), pos=torch.randn(3, 3),
            cell=(torch.eye(3) * 4)[None], periodic=torch.tensor([True]),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_shifts=torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
            geometry_known=torch.tensor([True, False, True]),
            atom_features=torch.randn(3, 2), cutoff=torch.tensor([6.0]),
        ),
        Data(
            z=torch.tensor([14]), pos=torch.randn(1, 3),
            cell=torch.zeros(1, 3, 3), periodic=torch.tensor([False]),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_shifts=torch.empty(0, 3, dtype=torch.long),
            geometry_known=torch.tensor([True]),
            atom_features=torch.randn(1, 2), cutoff=torch.tensor([6.0]),
        ),
    ]
    batch = Batch.from_data_list(graphs).to("cpu")
    batch.pos.requires_grad_()
    result = encoder(batch)
    assert result.shape == (2, 4, 16)
    individual = torch.cat([encoder(graph) for graph in graphs])
    torch.testing.assert_close(result, individual, atol=1e-6, rtol=1e-5)
    result.square().sum().backward()
    assert torch.isfinite(batch.pos.grad).all()
    assert torch.isfinite(encoder.local_encoder.layers[0].q_proj.weight.grad).all()
    assert (encoder.latent_queries.grad.norm(dim=-1) > 0).all()
    position_grad = batch.pos.grad.clone()
    weight_grad = encoder.local_encoder.layers[0].q_proj.weight.grad.clone()

    # CPU-computed padding metadata must preserve outputs and training gradients.
    batch.max_num_nodes = 3
    encoder.zero_grad(set_to_none=True)
    batch.pos.grad = None
    hinted = encoder(batch)
    torch.testing.assert_close(hinted, result)
    hinted.square().sum().backward()
    torch.testing.assert_close(batch.pos.grad, position_grad)
    torch.testing.assert_close(encoder.local_encoder.layers[0].q_proj.weight.grad, weight_grad)


@pytest.fixture
def lattice_encoder():
    torch.manual_seed(7)
    return UniversalEncoder(EncoderConfig(
        atom_dim=16, dim=32, output_dim=16, num_latents=4,
        local_layers=1, latent_layers=1, heads=4, radial_basis=8, cutoff=6.0,
    )).eval()


@pytest.mark.parametrize("bf16", [False, True])
def test_local_features_support_independent_token_compression(lattice_encoder, bf16):
    local = LocalAtomTransformer(lattice_encoder.config).eval()
    local.load_state_dict(lattice_encoder.local_encoder.state_dict())
    graphs = [
        Data(z=torch.tensor([6, 8]), pos=torch.tensor([[0., 0., 0.], [1.2, 0., 0.]]),
             edge_index=torch.tensor([[0, 1], [1, 0]])),
        Data(z=torch.tensor([14]), pos=torch.zeros(1, 3),
             edge_index=torch.empty(2, 0, dtype=torch.long)),
    ]
    batch = Batch.from_data_list(graphs)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        expected, expected_atoms = lattice_encoder(batch, return_atoms=True)
        atoms = local(batch)
        individual = torch.cat([local(graph) for graph in graphs])
    torch.testing.assert_close(atoms, expected_atoms)
    torch.testing.assert_close(atoms, individual, atol=1e-5, rtol=1e-5)

    # Compression needs only prepared atom features and graph membership.
    features = atoms.detach().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        tokens = lattice_encoder.compress(features, batch.batch, num_graphs=2, max_num_nodes=2)
        permuted = lattice_encoder.compress(features[[1, 0, 2]], batch.batch)
    torch.testing.assert_close(tokens, expected)
    torch.testing.assert_close(permuted, tokens, atol=1e-5, rtol=1e-5)
    tokens.float().square().mean().backward()
    assert torch.isfinite(features.grad).all() and features.grad.norm() > 0
    assert lattice_encoder.latent_queries.grad.norm() > 0
    assert all(parameter.grad is None for parameter in lattice_encoder.local_encoder.parameters())


def single_atom_graph(cell=None):
    return Data(z=torch.tensor([6]), pos=torch.zeros(1, 3), cell=cell,
                edge_index=torch.empty(2, 0, dtype=torch.long))


@pytest.mark.parametrize("bf16", [False, True])
def test_lattice_scale_and_shape_are_visible_without_neighbors(lattice_encoder, bf16):
    # Every cell's shortest translation exceeds the 6 A cutoff: no local geometry.
    cells = lattice_params_to_matrix_torch(
        torch.tensor([[10., 10., 10.], [12., 12., 12.], [10., 10., 10.]]),
        torch.tensor([[90., 90., 90.], [90., 90., 90.], [90., 90., 105.]]),
    )
    batch = Batch.from_data_list([single_atom_graph(cell[None]) for cell in cells])
    batch.cell.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        result, atoms = lattice_encoder(batch, return_atoms=True)
        # Supervise atoms directly: lattice information must arrive before pooling.
        loss = atoms.float().square().mean()
    assert result.shape == (3, 4, 16)
    assert not torch.allclose(atoms[0], atoms[1])
    assert not torch.allclose(atoms[0], atoms[2])
    assert torch.isfinite(result).all()
    assert not torch.allclose(result[0], result[1])  # Absolute scale.
    assert not torch.allclose(result[0], result[2])  # Angles at fixed lengths.
    loss.backward()
    assert torch.isfinite(batch.cell.grad).all()
    assert (batch.cell.grad.flatten(1).norm(dim=1) > 0).all()
    for parameter in lattice_encoder.local_encoder.lattice_embedding.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


def test_lattice_availability_and_zero_cell_placeholders(lattice_encoder):
    # Ablate the geometric embedding to isolate the learned availability state.
    with torch.no_grad():
        for parameter in lattice_encoder.local_encoder.lattice_embedding.parameters():
            parameter.zero_()
    absent = single_atom_graph()
    placeholder = single_atom_graph(torch.zeros(1, 3, 3))
    present = single_atom_graph((torch.eye(3) * 10)[None])
    batch = Batch.from_data_list([present, placeholder])
    batch.cell.requires_grad_()
    result = lattice_encoder(batch)
    torch.testing.assert_close(result[:1], lattice_encoder(present), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result[1:], lattice_encoder(absent), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(lattice_encoder(placeholder), lattice_encoder(absent))
    assert not torch.allclose(result[0], result[1])
    result.square().mean().backward()
    assert torch.isfinite(batch.cell.grad).all()
    assert batch.cell.grad[1].count_nonzero() == 0
    state_grad = lattice_encoder.local_encoder.lattice_presence_embedding.weight.grad
    assert torch.isfinite(state_grad).all()
    assert (state_grad.norm(dim=1) > 0).all()


def test_lattice_conditioning_preserves_rigid_transform_invariance(lattice_encoder):
    cell = lattice_params_to_matrix_torch(
        torch.tensor([[4., 5., 6.]]), torch.tensor([[80., 95., 110.]]),
    )
    graph = Data(
        z=torch.tensor([6, 8, 1]),
        pos=torch.tensor([[.05, .05, .05], [.95, .05, .05], [.2, .2, .25]]) @ cell[0],
        cell=cell, edge_index=torch.tensor([[1, 0, 2, 0], [0, 1, 0, 2]]),
        edge_shifts=torch.tensor([[-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    )
    rotation = torch.tensor([[.36, -.48, .8], [.8, .6, 0.], [-.48, .64, .6]])
    rotated = graph.clone()
    rotated.cell = graph.cell @ rotation
    rotated.pos = graph.pos @ rotation + torch.tensor([1., 2., 3.])
    torch.testing.assert_close(
        lattice_encoder(rotated, return_atoms=True),
        lattice_encoder(graph, return_atoms=True), atol=1e-6, rtol=1e-5,
    )


def test_nonperiodic_boxes_do_not_change_molecule_embeddings(lattice_encoder):
    molecule = Data(z=torch.tensor([6, 8]), pos=torch.tensor([[0., 0., 0.], [1.2, 0., 0.]]),
                    edge_index=torch.tensor([[0, 1], [1, 0]]), periodic=torch.tensor([False]))
    expected = lattice_encoder(molecule)
    boxed = molecule.clone()
    boxed.cell = (torch.eye(3) * 8)[None].requires_grad_()
    actual = lattice_encoder(boxed)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert boxed.cell.grad.count_nonzero() == 0
    boxed.cell = (torch.eye(3) * 16)[None]
    crystal = boxed.clone()
    crystal.periodic = torch.tensor([True])
    result = lattice_encoder(Batch.from_data_list([boxed, crystal]))
    torch.testing.assert_close(result[:1], expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(result[1:], lattice_encoder(crystal), atol=1e-6, rtol=1e-5)
    assert not torch.allclose(result[0], result[1])
    # Explicitly nonperiodic inputs also suppress image translations.
    boxed.edge_shifts = torch.tensor([[1, 0, 0], [-1, 0, 0]])
    torch.testing.assert_close(lattice_encoder(boxed), expected)


@pytest.mark.parametrize("distance", [5.999, 6.0, 6.1])
@pytest.mark.parametrize("bf16", [False, True])
def test_cutoff_edges_vanish_continuously_with_finite_gradients(lattice_encoder, distance, bf16):
    graph = Data(z=torch.tensor([6, 8]),
                 pos=torch.tensor([[0., 0., 0.], [distance, 0., 0.]], requires_grad=True),
                 edge_index=torch.tensor([[0, 1], [1, 0]]))
    removed = graph.clone()
    removed.edge_index = torch.empty(2, 0, dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        actual, expected = lattice_encoder(graph), lattice_encoder(removed)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.float().square().mean().backward()
    assert torch.isfinite(graph.pos.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in lattice_encoder.local_encoder.layers.parameters())


def test_edge_conditioning_changes_atoms_with_fixed_distances(lattice_encoder):
    # Zero only the initial atom conditioning to isolate the edge message path.
    graph = Data(
        z=torch.tensor([6, 8]), pos=torch.tensor([[0., 0., 0.], [1.2, 0., 0.]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]), cell=(torch.eye(3) * 8)[None],
    )
    larger = graph.clone()
    larger.cell = graph.cell * 1.5
    captured = []

    def fixed_atoms(module, args):
        atoms, edges, edge_features, weight = args
        captured.append(edge_features.detach().clone())
        return torch.zeros_like(atoms), edges, edge_features, weight

    handle = lattice_encoder.local_encoder.layers[0].register_forward_pre_hook(fixed_atoms)
    try:
        _, first = lattice_encoder(graph, return_atoms=True)
        _, second = lattice_encoder(larger, return_atoms=True)
    finally:
        handle.remove()
    assert not torch.allclose(captured[0], captured[1])
    assert not torch.allclose(first, second)
