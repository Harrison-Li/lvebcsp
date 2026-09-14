"""Tests for UniversalEncoder and PeakEncoder."""

import sys
from pathlib import Path

# Ensure src/ is in sys.path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import torch
from torch_geometric.data import Batch, Data

from lvebcsp.models.encoder import (
    EncoderConfig,
    UniversalEncoder,
    PeakEncoder,
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
    assert embedding.shape == (2, 256)
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
    assert out.shape == (1, 128)
    assert not torch.isnan(out).any()


def test_peak_encoder_compatibility():
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
    individual = torch.cat([encoder(graph) for graph in graphs])
    torch.testing.assert_close(result, individual, atol=1e-6, rtol=1e-5)
    result.square().sum().backward()
    assert torch.isfinite(batch.pos.grad).all()
    assert torch.isfinite(encoder.local_layers[0].q_proj.weight.grad).all()

