"""Sample structure candidates with a JEPA-conditioned LatentDiT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lvebcsp.data.peak_preprocess import preprocess_raw_pxrd, read_csv_peak_file, read_xy_file
from lvebcsp.data.representation import tensor_to_structure
from lvebcsp.inference.export_cif import export_structure_to_cif
from lvebcsp.inference.rank_pxrd import rank_structures, save_ranked_csv
from lvebcsp.models.dit import LatentDiT
from lvebcsp.train.train_direct_cspdit import build_direct_from_config, load_direct_model_state_dict, load_jepa_checkpoint
from lvebcsp.common.chemistry import atom_types_to_element_ratios, formula_to_atom_types
from lvebcsp.common.config import load_config, select_device
from lvebcsp.common.seed import seed_everything


def inherit_checkpoint_sampling_defaults(config: dict[str, Any], checkpoint_config: dict[str, Any]) -> None:
    """Fill sampling defaults that must match the LatentDiT training path."""

    checkpoint_flow = checkpoint_config.get("flow", {}) if isinstance(checkpoint_config, dict) else {}
    config_flow = config.get("flow", {})
    has_diffuse_lattice = "diffuse_lattice" in config or (
        isinstance(config_flow, dict) and "diffuse_lattice" in config_flow
    )
    if not has_diffuse_lattice and isinstance(checkpoint_flow, dict) and "diffuse_lattice" in checkpoint_flow:
        config["diffuse_lattice"] = bool(checkpoint_flow["diffuse_lattice"])


def load_models(config: dict[str, Any], device: torch.device):
    """Load JEPA and LatentDiT checkpoints."""

    jepa = load_jepa_checkpoint(config["jepa_ckpt"], device)
    jepa.eval()

    latent_dit_ckpt_path = config.get("latent_dit_ckpt")
    if latent_dit_ckpt_path is None:
        raise ValueError("sample config must set latent_dit_ckpt")
    direct_ckpt = torch.load(latent_dit_ckpt_path, map_location=device)
    direct_config = direct_ckpt.get("config", config)
    inherit_checkpoint_sampling_defaults(config, direct_config)
    direct = build_direct_from_config(direct_config).to(device)
    use_direct_ema = bool(config.get("use_direct_ema", config.get("use_ema", True)))
    direct_state = direct_ckpt["ema_model"] if use_direct_ema and "ema_model" in direct_ckpt else direct_ckpt["model"]
    load_direct_model_state_dict(direct, direct_state)
    direct.eval()
    return jepa, direct


def read_experimental_pxrd(
    path: str | Path,
    p_max: int,
    wavelength: str | float,
    preprocess: dict[str, Any] | None = None,
):
    """Read .xy or .csv experimental PXRD and preprocess to d-I peaks."""

    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        two_theta, intensity = read_csv_peak_file(input_path)
    else:
        two_theta, intensity = read_xy_file(input_path)
    return preprocess_raw_pxrd(
        two_theta,
        intensity,
        p_max=p_max,
        wavelength=wavelength,
        **dict(preprocess or {}),
    )


def normalize_clean_fractional_coords(y: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Wrap predicted clean fractional coordinates and zero padded atom slots."""

    out = y.clone()
    atom_mask_f = atom_mask.to(device=out.device, dtype=out.dtype).unsqueeze(-1)
    out[:, 1:, :3] = torch.remainder(out[:, 1:, :3], 1.0) * atom_mask_f
    out[:, 1:, 3:] = out[:, 1:, 3:] * atom_mask_f
    return out


@torch.no_grad()
def _flow_velocity(
    model: LatentDiT,
    lattice_encoded: torch.Tensor | None,
    frac_coords: torch.Tensor,
    t: torch.Tensor,
    z: torch.Tensor,
    atom_types: torch.Tensor,
    atom_mask: torch.Tensor,
    guidance_scale: float,
    eps_time: float = 1e-4,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Compute the direct flow velocity at one time."""

    if guidance_scale == 1.0:
        out = model(
            lattice_encoded,
            frac_coords,
            t,
            z,
            atom_types,
            atom_mask,
        )
    else:
        out = model.guided_forward(
            lattice_encoded,
            frac_coords,
            t,
            z,
            atom_types,
            atom_mask,
            guidance_scale=guidance_scale,
        )
    denom = (1.0 - t).view(-1, 1).clamp_min(eps_time)
    lattice_velocity = None
    if lattice_encoded is not None:
        lattice_velocity = (out["Y0_pred"][:, 0, :6] - lattice_encoded) / denom
    if "coord_velocity" in out:
        coord_velocity = out["coord_velocity"]
    else:
        delta = torch.remainder(out["coord_pred"] - frac_coords + 0.5, 1.0) - 0.5
        coord_velocity = delta / denom.unsqueeze(-1)
    return lattice_velocity, coord_velocity


def _wrap_fractional_state(frac_coords: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Keep fractional coordinates in the unit interval and padded atom slots zero."""

    return torch.remainder(frac_coords, 1.0) * atom_mask.to(dtype=frac_coords.dtype).unsqueeze(-1)


@torch.no_grad()
def integrate_euler(
    model: LatentDiT,
    lattice_encoded: torch.Tensor | None,
    frac_coords: torch.Tensor,
    z: torch.Tensor,
    atom_types: torch.Tensor,
    atom_mask: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    t_end: float = 1.0,
) -> torch.Tensor:
    """Euler integration of the direct flow."""

    eps_time = 1e-4
    t_end = min(max(float(t_end), eps_time), 1.0 - eps_time)
    delta_t = t_end / max(num_steps, 1)
    for step in range(num_steps):
        t_value = min(step * delta_t, t_end)
        t = torch.full((frac_coords.shape[0],), t_value, device=frac_coords.device, dtype=torch.float32)
        lattice_velocity, coord_velocity = _flow_velocity(
            model,
            lattice_encoded,
            frac_coords,
            t,
            z,
            atom_types,
            atom_mask,
            guidance_scale,
            eps_time=eps_time,
        )
        if lattice_encoded is not None and lattice_velocity is not None:
            lattice_encoded = lattice_encoded + delta_t * lattice_velocity
        frac_coords = frac_coords + delta_t * coord_velocity
        frac_coords = _wrap_fractional_state(frac_coords, atom_mask)
    return lattice_encoded, frac_coords


@torch.no_grad()
def integrate_heun(
    model: LatentDiT,
    lattice_encoded: torch.Tensor | None,
    frac_coords: torch.Tensor,
    z: torch.Tensor,
    atom_types: torch.Tensor,
    atom_mask: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    t_end: float = 1.0,
) -> torch.Tensor:
    """Second-order Heun integration of the direct flow."""

    eps_time = 1e-4
    t_end = min(max(float(t_end), eps_time), 1.0 - eps_time)
    delta_t = t_end / max(num_steps, 1)
    for step in range(num_steps):
        t_value = min(step * delta_t, t_end)
        next_t_value = min((step + 1) * delta_t, t_end)
        t = torch.full((frac_coords.shape[0],), t_value, device=frac_coords.device, dtype=torch.float32)
        next_t = torch.full((frac_coords.shape[0],), next_t_value, device=frac_coords.device, dtype=torch.float32)
        k1_lattice, k1_coord = _flow_velocity(
            model,
            lattice_encoded,
            frac_coords,
            t,
            z,
            atom_types,
            atom_mask,
            guidance_scale,
            eps_time=eps_time,
        )
        lattice_euler = None
        if lattice_encoded is not None and k1_lattice is not None:
            lattice_euler = lattice_encoded + delta_t * k1_lattice
        coords_euler = _wrap_fractional_state(frac_coords + delta_t * k1_coord, atom_mask)
        k2_lattice, k2_coord = _flow_velocity(
            model,
            lattice_euler,
            coords_euler,
            next_t,
            z,
            atom_types,
            atom_mask,
            guidance_scale,
            eps_time=eps_time,
        )
        if lattice_encoded is not None and k1_lattice is not None and k2_lattice is not None:
            lattice_encoded = lattice_encoded + 0.5 * delta_t * (k1_lattice + k2_lattice)
        frac_coords = _wrap_fractional_state(frac_coords + 0.5 * delta_t * (k1_coord + k2_coord), atom_mask)
    return lattice_encoded, frac_coords


def integrate_flow(
    model: LatentDiT,
    frac_coords: torch.Tensor,
    z: torch.Tensor,
    atom_types: torch.Tensor,
    atom_mask: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    sampler: str = "euler",
    t_end: float = 1.0,
    final_denoise: bool = True,
    diffuse_lattice: bool = False,
    lattice_encoded: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the direct flow with a configured ODE sampler."""

    if diffuse_lattice and lattice_encoded is None:
        lattice_encoded = torch.randn(frac_coords.shape[0], 6, device=frac_coords.device, dtype=frac_coords.dtype)
    sampler = sampler.lower()
    if sampler == "euler":
        lattice_encoded, frac_coords = integrate_euler(
            model,
            lattice_encoded,
            frac_coords,
            z,
            atom_types,
            atom_mask,
            num_steps,
            guidance_scale,
            t_end=t_end,
        )
    elif sampler == "heun":
        lattice_encoded, frac_coords = integrate_heun(
            model,
            lattice_encoded,
            frac_coords,
            z,
            atom_types,
            atom_mask,
            num_steps,
            guidance_scale,
            t_end=t_end,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler!r}; expected 'euler' or 'heun'")
    t = torch.full(
        (frac_coords.shape[0],),
        min(max(float(t_end), 1.0e-4), 1.0 - 1.0e-4),
        device=frac_coords.device,
        dtype=torch.float32,
    )
    if guidance_scale == 1.0:
        out = model(
            lattice_encoded,
            frac_coords,
            t,
            z,
            atom_types,
            atom_mask,
        )
    else:
        out = model.guided_forward(
            lattice_encoded,
            frac_coords,
            t,
            z,
            atom_types,
            atom_mask,
            guidance_scale=guidance_scale,
        )
    return normalize_clean_fractional_coords(out["Y0_pred"], atom_mask) if final_denoise else out["Y0_pred"]


def sample_candidates(
    config: dict[str, Any],
    pxrd_path: str | Path,
    formula: str,
    out_dir: str | Path,
    num_samples: int | None = None,
) -> list[dict[str, Any]]:
    """End-to-end sampling and ranking routine."""

    seed_everything(int(config.get("seed", 42)))
    device = select_device(str(config.get("device", "auto")))
    jepa, direct = load_models(config, device)
    p_max = int(config.get("p_max", 128))
    n_max = int(config.get("n_max", 64))
    d_y = int(config.get("d_y", 8))
    wavelength = config.get("wavelength", "CuKa")
    count = int(num_samples or config.get("num_samples", 20))
    output_dir = Path(out_dir)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    exp_peaks = read_experimental_pxrd(
        pxrd_path,
        p_max=p_max,
        wavelength=wavelength,
        preprocess=config.get("pxrd_preprocess"),
    )
    atom_types_one = formula_to_atom_types(formula, n_max=n_max)
    atom_mask_one = atom_types_one > 0
    ratio_one = atom_types_to_element_ratios(atom_types_one)

    atom_types = atom_types_one.unsqueeze(0).repeat(count, 1).to(device)
    atom_mask = atom_mask_one.unsqueeze(0).repeat(count, 1).to(device)

    z = jepa.encode_ctx(
        exp_peaks.peak_d.unsqueeze(0).to(device),
        exp_peaks.peak_i.unsqueeze(0).to(device),
        exp_peaks.peak_mask.unsqueeze(0).to(device),
        atom_types=atom_types_one.unsqueeze(0).to(device),
        ratio=ratio_one.unsqueeze(0).to(device),
    )
    z_samples = z.repeat(count, 1)
    frac_coords = torch.rand(count, n_max, 3, device=device)
    frac_coords = frac_coords * atom_mask.float().unsqueeze(-1)
    y_final = integrate_flow(
        direct,
        frac_coords,
        z_samples,
        atom_types,
        atom_mask,
        num_steps=int(config.get("num_steps", 200)),
        guidance_scale=float(config.get("guidance_scale", 1.5)),
        sampler=str(config.get("sampler", "euler")),
        t_end=float(config.get("sample_t_max", config.get("t_end", 0.95))),
        final_denoise=bool(config.get("final_denoise", True)),
        diffuse_lattice=bool(config.get("diffuse_lattice", config.get("flow", {}).get("diffuse_lattice", False))),
    )
    structures = []
    for idx in range(count):
        structure = tensor_to_structure(y_final[idx].cpu(), atom_types[idx].cpu(), atom_mask[idx].cpu())
        path = export_structure_to_cif(structure, candidates_dir / f"candidate_{idx:04d}.cif")
        structures.append((path, structure))

    rows = rank_structures(
        structures,
        exp_peaks,
        z.squeeze(0),
        jepa,
        p_max=p_max,
        wavelength=wavelength,
        ranking_weights=config.get("ranking", {}),
        validity=config.get("validity", {}),
    )
    save_ranked_csv(rows, output_dir / "ranked_candidates.csv")
    summary = {"num_candidates": len(rows), "best": rows[0] if rows else None}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample PXRD-conditioned crystal structures")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pxrd", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    sample_candidates(config, args.pxrd, args.formula, args.out_dir, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
