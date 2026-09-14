"""Direct crystal-structure tensor representation.

Lattices are represented by six unconstrained coordinates of a canonical
Cholesky factor of their Gram matrix.  The first three coordinates are log
lengths.  The remaining coordinates parameterize two correlations and one
partial correlation.  This is a bijective coordinate chart for the
non-degenerate cells used by this project and, unlike three independently
predicted angles, always decodes to a positive-definite Gram matrix.
"""

from __future__ import annotations

import math

import torch
from pymatgen.core import Lattice, Structure


# Keep decoded samples in the numerically useful region accepted by pymatgen.
# Clean cells in the supported datasets lie well inside these limits.
MIN_LATTICE_LENGTH = 0.25
MAX_LATTICE_LENGTH = 1_000.0

# Correlations strictly inside (-1, 1) keep every Cholesky diagonal positive.
# A margin of 0.05 also prevents a finite model output from becoming an
# effectively singular cell after float32 rounding.
PARTIAL_CORRELATION_BOUND = 0.95
_ENCODE_EPS = 1.0e-6
_COSINE_EPS = 1.0e-7


def _require_last_dimension(tensor: torch.Tensor, size: int, name: str) -> None:
    if tensor.ndim == 0 or tensor.shape[-1] != size:
        raise ValueError(f"{name} must have final dimension {size}")


def _bounded_partial_correlations(values: torch.Tensor) -> torch.Tensor:
    """Map unconstrained shape coordinates into a stable open correlation box."""

    return PARTIAL_CORRELATION_BOUND * torch.tanh(values)


def encode_lattice_params(lattice_params: torch.Tensor) -> torch.Tensor:
    """Encode ``[a,b,c,alpha,beta,gamma]`` as Gram-Cholesky coordinates.

    For the angle order ``[alpha, beta, gamma]``, the shape coordinates are
    ``[atanh(rho_alpha|beta,gamma), atanh(cos(beta)), atanh(cos(gamma))]``
    after scaling by :data:`PARTIAL_CORRELATION_BOUND`.  The conditional
    correlation is

    ``(cos(alpha) - cos(beta) cos(gamma)) / (sin(beta) sin(gamma))``.

    It is precisely the remaining degree of freedom in a lower-triangular
    Cholesky factor.  Valid input cells round-trip exactly when they lie inside
    the generous numerical bounds above; near-degenerate cells are projected
    to the stable boundary.
    """

    params = lattice_params.to(torch.float32)
    _require_last_dimension(params, 6, "lattice_params")
    lengths = params[..., :3].clamp(
        min=MIN_LATTICE_LENGTH,
        max=MAX_LATTICE_LENGTH,
    )
    alpha, beta, gamma = torch.deg2rad(params[..., 3:6]).unbind(dim=-1)
    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_beta = torch.sin(beta).abs().clamp_min(_ENCODE_EPS)
    sin_gamma = torch.sin(gamma).abs().clamp_min(_ENCODE_EPS)
    partial_alpha = (
        cos_alpha - cos_beta * cos_gamma
    ) / (sin_beta * sin_gamma)
    correlations = torch.stack(
        (partial_alpha, cos_beta, cos_gamma),
        dim=-1,
    )
    scaled = (correlations / PARTIAL_CORRELATION_BOUND).clamp(
        min=-1.0 + _ENCODE_EPS,
        max=1.0 - _ENCODE_EPS,
    )
    shape_coordinates = torch.atanh(scaled)
    return torch.cat((torch.log(lengths), shape_coordinates), dim=-1)


def canonicalize_normalized_lattice_angles(angles: torch.Tensor) -> torch.Tensor:
    """Fold legacy angle coordinates onto ``[0, 1]``.

    This remains available for loading or inspecting the former representation;
    current lattice encoding uses Gram-Cholesky coordinates instead.
    """

    wrapped = torch.remainder(angles, 2.0)
    return torch.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def encoded_lattice_to_matrix(encoded: torch.Tensor) -> torch.Tensor:
    """Decode unconstrained coordinates to a canonical lattice matrix.

    Rows are the lattice vectors ``a``, ``b``, and ``c``.  The matrix is lower
    triangular with a strictly positive diagonal, so it is non-singular for
    every finite encoded input.  Consequently ``L @ L.T`` is always a valid
    positive-definite lattice Gram matrix.
    """

    values = encoded.to(torch.float32)
    _require_last_dimension(values, 6, "encoded")
    log_lengths = values[..., :3].clamp(
        min=math.log(MIN_LATTICE_LENGTH),
        max=math.log(MAX_LATTICE_LENGTH),
    )
    lengths = torch.exp(log_lengths)
    partial_alpha, cos_beta, cos_gamma = _bounded_partial_correlations(
        values[..., 3:6]
    ).unbind(dim=-1)
    sin_beta = torch.sqrt((1.0 - cos_beta.square()).clamp_min(_COSINE_EPS))
    sin_gamma = torch.sqrt((1.0 - cos_gamma.square()).clamp_min(_COSINE_EPS))
    sin_partial = torch.sqrt(
        (1.0 - partial_alpha.square()).clamp_min(_COSINE_EPS)
    )

    a, b, c = lengths.unbind(dim=-1)
    zeros = torch.zeros_like(a)
    row_a = torch.stack((a, zeros, zeros), dim=-1)
    row_b = torch.stack((b * cos_gamma, b * sin_gamma, zeros), dim=-1)
    row_c = torch.stack(
        (
            c * cos_beta,
            c * sin_beta * partial_alpha,
            c * sin_beta * sin_partial,
        ),
        dim=-1,
    )
    return torch.stack((row_a, row_b, row_c), dim=-2)


def encoded_lattice_to_gram(encoded: torch.Tensor) -> torch.Tensor:
    """Return the rotation-invariant positive-definite Gram matrix ``L L^T``."""

    lattice = encoded_lattice_to_matrix(encoded)
    return lattice @ lattice.transpose(-1, -2)


def scale_normalized_lattice_gram(encoded: torch.Tensor) -> torch.Tensor:
    """Return a bounded-scale Gram feature while retaining cell shape.

    Absolute scale is supplied separately by the three log lengths in the
    models.  Dividing by the mean squared length lets the full nine Gram entries
    enter an MLP without large-cell magnitudes dominating its activations.
    """

    gram = encoded_lattice_to_gram(encoded)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
    return gram / scale.clamp_min(_COSINE_EPS).unsqueeze(-1)


def lattice_matrix_to_params(lattice_matrix: torch.Tensor) -> torch.Tensor:
    """Convert row-vector lattice matrices to lengths and degree angles."""

    matrix = lattice_matrix.to(torch.float32)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError("lattice_matrix must have final dimensions [3, 3]")
    lengths = torch.linalg.vector_norm(matrix, dim=-1).clamp_min(_COSINE_EPS)
    a_vec, b_vec, c_vec = matrix.unbind(dim=-2)
    a, b, c = lengths.unbind(dim=-1)
    cos_alpha = (b_vec * c_vec).sum(dim=-1) / (b * c)
    cos_beta = (a_vec * c_vec).sum(dim=-1) / (a * c)
    cos_gamma = (a_vec * b_vec).sum(dim=-1) / (a * b)
    cosines = torch.stack((cos_alpha, cos_beta, cos_gamma), dim=-1).clamp(
        min=-1.0 + _COSINE_EPS,
        max=1.0 - _COSINE_EPS,
    )
    angles = torch.rad2deg(torch.acos(cosines))
    return torch.cat((lengths, angles), dim=-1)


def decode_lattice_params(encoded: torch.Tensor) -> torch.Tensor:
    """Decode Gram-Cholesky coordinates to valid lattice parameters."""

    return lattice_matrix_to_params(encoded_lattice_to_matrix(encoded))


def build_structure_tensor(
    atom_types: torch.Tensor,
    frac_coords: torch.Tensor,
    lattice_params: torch.Tensor,
    atom_mask: torch.Tensor,
    d_y: int = 8,
) -> torch.Tensor:
    """Build Y_0 with one lattice token followed by padded atom tokens."""

    if d_y < 6:
        raise ValueError("d_y must be at least 6")
    if frac_coords.ndim == 2:
        n_max = frac_coords.shape[0]
        y = torch.zeros(1 + n_max, d_y, dtype=torch.float32, device=frac_coords.device)
        y[0, :6] = encode_lattice_params(lattice_params)
        y[1:, :3] = torch.remainder(frac_coords.to(torch.float32), 1.0) * atom_mask.to(
            frac_coords.device
        ).float().unsqueeze(-1)
        return y
    if frac_coords.ndim == 3:
        batch, n_max, _ = frac_coords.shape
        y = torch.zeros(batch, 1 + n_max, d_y, dtype=torch.float32, device=frac_coords.device)
        y[:, 0, :6] = encode_lattice_params(lattice_params)
        y[:, 1:, :3] = torch.remainder(frac_coords.to(torch.float32), 1.0) * atom_mask.to(
            frac_coords.device
        ).float().unsqueeze(-1)
        return y
    raise ValueError(f"Expected frac_coords rank 2 or 3, got shape {tuple(frac_coords.shape)}")


def tensor_to_structure(
    y: torch.Tensor,
    atom_types: torch.Tensor,
    atom_mask: torch.Tensor,
) -> Structure:
    """Construct a pymatgen Structure from one direct structure tensor."""

    if y.ndim != 2:
        raise ValueError("tensor_to_structure expects a single [1 + N, D_y] tensor")
    lattice_params = decode_lattice_params(y[0, :6]).detach().cpu().numpy()
    frac_coords = torch.remainder(y[1:, :3].to(torch.float32), 1.0).detach().cpu()
    atom_types_cpu = atom_types.detach().cpu().long()
    mask_cpu = atom_mask.detach().cpu().bool()
    species = [int(z) for z, keep in zip(atom_types_cpu.tolist(), mask_cpu.tolist()) if keep and int(z) > 0]
    coords = frac_coords[mask_cpu].numpy()
    if not species:
        raise ValueError("Cannot construct a structure with zero valid atoms")
    lattice = Lattice.from_parameters(*[float(value) for value in lattice_params])
    return Structure(lattice, species, coords, coords_are_cartesian=False)
