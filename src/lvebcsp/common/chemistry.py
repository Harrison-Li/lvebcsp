"""Formula parsing and element-count utilities."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from pymatgen.core import Composition, Element


def formula_to_element_counts(formula: str) -> torch.Tensor:
    """Return counts indexed by atomic number in a length-119 tensor."""

    counts = torch.zeros(119, dtype=torch.float32)
    composition = Composition(formula)
    for element, amount in composition.get_el_amt_dict().items():
        z = Element(element).Z
        counts[z] = float(amount)
    return counts


def formula_to_reduced_element_counts(formula: str) -> torch.Tensor:
    """Return reduced, possibly non-integer formula coefficients by element.

    ``Fe6O8`` becomes Fe3O4, while an explicitly noisy formula such as
    ``Fe1.9O3`` retains the coefficients 1.9 and 3.0.
    """

    counts = torch.zeros(119, dtype=torch.float32)
    composition = Composition(formula).reduced_composition
    for element, amount in composition.get_el_amt_dict().items():
        counts[Element(element).Z] = float(amount)
    return counts


def formula_to_atom_types(formula: str, n_max: int | None = None) -> torch.Tensor:
    """Expand a formula into atomic numbers sorted by atomic number."""

    atom_types: list[int] = []
    composition = Composition(formula)
    for element, amount in composition.get_el_amt_dict().items():
        z = Element(element).Z
        rounded = int(round(float(amount)))
        atom_types.extend([z] * rounded)
    atom_types.sort()
    if n_max is not None:
        if len(atom_types) > n_max:
            raise ValueError(f"Formula {formula} has {len(atom_types)} atoms, above n_max={n_max}")
        atom_types.extend([0] * (n_max - len(atom_types)))
    return torch.tensor(atom_types, dtype=torch.long)


def atom_types_to_element_counts(atom_types: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Convert atomic-number tokens into a length-119 count vector."""

    tensor = torch.as_tensor(atom_types, dtype=torch.long)
    counts = torch.zeros(119, dtype=torch.float32, device=tensor.device)
    valid = tensor[(tensor > 0) & (tensor < 119)]
    if valid.numel() > 0:
        counts.scatter_add_(0, valid, torch.ones_like(valid, dtype=torch.float32))
    return counts.cpu()


def element_counts_to_ratios(element_counts: torch.Tensor) -> torch.Tensor:
    """Normalize element counts into elemental ratios along the atomic-number axis."""

    counts = torch.as_tensor(element_counts, dtype=torch.float32)
    totals = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    ratios = counts / totals
    ratios[..., 0] = 0.0
    return ratios


def element_counts_to_reduced_counts(element_counts: torch.Tensor) -> torch.Tensor:
    """Remove a shared integer multiplicity from element-count vectors.

    The returned counts retain stoichiometric coefficients but cannot reveal a
    unit-cell repetition factor. For example, both ``NaCl`` and ``Na2Cl2`` map
    to one Na and one Cl. Inputs may be a single ``[119]`` vector or a batch
    with shape ``[..., 119]``.
    """

    counts = torch.as_tensor(element_counts, dtype=torch.float32)
    if counts.ndim < 1 or counts.shape[-1] != 119:
        raise ValueError(
            "element_counts must have shape [119] or [..., 119], got "
            f"{tuple(counts.shape)}"
        )
    if not torch.isfinite(counts).all() or torch.any(counts < 0):
        raise ValueError("element_counts must be finite and non-negative")
    rounded = counts.round()
    if not torch.allclose(counts, rounded, atol=1.0e-5, rtol=0.0):
        raise ValueError("element_counts must contain integer stoichiometric counts")
    rounded = rounded.to(torch.long).clone()
    rounded[..., 0] = 0
    if torch.any(rounded.sum(dim=-1) < 1):
        raise ValueError("each element-count vector must contain an element")

    flat = rounded.reshape(-1, rounded.shape[-1])
    divisors = torch.zeros(flat.shape[0], dtype=torch.long, device=flat.device)
    for element_counts_column in flat.unbind(dim=-1):
        divisors = torch.gcd(divisors, element_counts_column)
    reduced = flat // divisors.clamp_min(1).unsqueeze(-1)
    return reduced.reshape_as(rounded).to(dtype=counts.dtype)


def element_counts_to_formula_multiplicity(
    target_element_counts: torch.Tensor,
    condition_element_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an integer formula-unit multiplier and an availability mask.

    A multiplier is available only when the target composition is exactly one
    positive integer multiple of the supplied condition composition. Inputs may
    have shape ``[119]`` or ``[..., 119]``. Unavailable rows return multiplier
    zero and ``False`` without weakening validation of malformed count tensors.
    """

    target = torch.as_tensor(target_element_counts, dtype=torch.float32)
    condition = torch.as_tensor(
        condition_element_counts,
        dtype=torch.float32,
        device=target.device,
    )
    if target.shape != condition.shape or target.ndim < 1 or target.shape[-1] != 119:
        raise ValueError(
            "target and condition element counts must share shape [119] or "
            f"[..., 119], got {tuple(target.shape)} and {tuple(condition.shape)}"
        )
    if (
        not torch.isfinite(target).all()
        or not torch.isfinite(condition).all()
        or torch.any(target < 0)
        or torch.any(condition < 0)
    ):
        raise ValueError("element counts must be finite and non-negative")
    target_rounded = target.round()
    condition_rounded = condition.round()
    if not torch.allclose(target, target_rounded, atol=1.0e-5, rtol=0.0):
        raise ValueError("target element counts must be integers")
    if not torch.allclose(condition, condition_rounded, atol=1.0e-5, rtol=0.0):
        raise ValueError("condition element counts must be integers")

    target_long = target_rounded.to(torch.long).clone()
    condition_long = condition_rounded.to(torch.long).clone()
    target_long[..., 0] = 0
    condition_long[..., 0] = 0
    flat_target = target_long.reshape(-1, 119)
    flat_condition = condition_long.reshape(-1, 119)
    target_active = flat_target > 0
    condition_active = flat_condition > 0
    support_matches = torch.equal(target_active, condition_active)
    if support_matches:
        support_mask = torch.ones(
            flat_target.shape[0],
            dtype=torch.bool,
            device=target.device,
        )
    else:
        support_mask = (target_active == condition_active).all(dim=-1)

    safe_condition = torch.where(
        condition_active,
        flat_condition,
        torch.ones_like(flat_condition),
    )
    ratios = flat_target // safe_condition
    divisible = torch.where(
        condition_active,
        flat_target.remainder(safe_condition) == 0,
        torch.ones_like(condition_active),
    ).all(dim=-1)
    large = torch.iinfo(torch.long).max
    minimum = ratios.masked_fill(~condition_active, large).min(dim=-1).values
    maximum = ratios.masked_fill(~condition_active, 0).max(dim=-1).values
    nonempty = condition_active.any(dim=-1) & target_active.any(dim=-1)
    available = (
        support_mask
        & divisible
        & nonempty
        & (minimum == maximum)
        & (maximum >= 1)
    )
    multiplicity = torch.where(available, maximum, torch.zeros_like(maximum))
    output_shape = target.shape[:-1]
    return multiplicity.reshape(output_shape), available.reshape(output_shape)


def atom_types_to_element_ratios(atom_types: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Convert atomic-number tokens into normalized elemental ratios."""

    tensor = torch.as_tensor(atom_types, dtype=torch.long)
    squeeze = tensor.ndim == 1
    if squeeze:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"atom_types must have shape [B, N] or [N], got {tuple(tensor.shape)}")
    counts = torch.zeros(tensor.shape[0], 119, dtype=torch.float32, device=tensor.device)
    valid = (tensor > 0) & (tensor < 119)
    safe_atom_types = torch.where(valid, tensor, torch.zeros_like(tensor))
    counts.scatter_add_(1, safe_atom_types, valid.to(torch.float32))
    ratios = element_counts_to_ratios(counts)
    return ratios.squeeze(0) if squeeze else ratios
