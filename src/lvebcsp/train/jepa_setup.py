"""Model/data construction and checkpoint migration for crystal JEPA.

Keep configuration and historical checkpoint support outside the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn

from lvebcsp.data.cif_dataset import CIFPXRDDataset
from lvebcsp.data.mp20_lmdb import MP20LMDBDataset
from lvebcsp.data.organic_lmdb import OrganicLMDBDataset
from lvebcsp.data.crystal_dataset import CrystalDataset
from lvebcsp.data.crystal_pxrd import pxrd_settings
from lvebcsp.models.encoder import PeakEncoder, TransformerPeakEncoder
from lvebcsp.models.jepa import Lvebm
from lvebcsp.models.layers import AdaLNMLP, GatedMLP, MLP
from lvebcsp.models.module import Predictor


def build_jepa_head(
    config: Any,
    input_dim: int,
    output_dim: int,
    *,
    condition_dim: int | None = None,
) -> nn.Module | None:
    """Build an optional JEPA projection/prediction head from a small local config."""

    if config is None:
        return None
    if isinstance(config, nn.Module):
        return config
    if isinstance(config, str):
        head_type = config
        cfg: dict[str, Any] = {}
    elif isinstance(config, dict):
        cfg = dict(config)
        head_type = str(cfg.pop("type", "mlp"))
    else:
        raise TypeError(f"JEPA head config must be a string, dict, nn.Module, or None; got {type(config).__name__}")

    head_type = head_type.lower().replace("-", "_")
    if head_type in {"identity", "none"}:
        return nn.Identity()

    if head_type == "predictor":
        if condition_dim is None:
            raise ValueError("JEPA head type 'predictor' is only valid for the conditional predictor head")
        if "num_layers" in cfg and "depth" not in cfg:
            cfg["depth"] = cfg.pop("num_layers")
        if "num_heads" in cfg and "heads" not in cfg:
            cfg["heads"] = cfg.pop("num_heads")
        context_dim = int(cfg.pop("context_dim", cfg.get("input_dim", output_dim)))
        cfg.setdefault("input_dim", context_dim)
        cfg.setdefault("condition_dim", condition_dim)
        cfg.setdefault("output_dim", output_dim)
        cfg.setdefault("hidden_dim", output_dim)
        cfg.setdefault("num_frames", 1)
        cfg.setdefault("depth", 2)
        cfg.setdefault("heads", 4)
        cfg["input_dim"] = int(cfg["input_dim"])
        cfg["condition_dim"] = int(cfg["condition_dim"])
        cfg["output_dim"] = int(cfg["output_dim"])
        cfg["hidden_dim"] = int(cfg["hidden_dim"])
        cfg["num_frames"] = int(cfg["num_frames"])
        cfg["depth"] = int(cfg["depth"])
        cfg["heads"] = int(cfg["heads"])
        heads = int(cfg["heads"])
        if heads < 1:
            raise ValueError("predictor.heads must be >= 1")
        cfg.setdefault("dim_head", max(int(cfg["hidden_dim"]) // heads, 1))
        cfg.setdefault("mlp_dim", int(cfg["hidden_dim"]) * 4)
        cfg["dim_head"] = int(cfg["dim_head"])
        cfg["mlp_dim"] = int(cfg["mlp_dim"])
        return Predictor(**cfg)

    if head_type == "mlp":
        cfg.setdefault("input_dim", input_dim)
        cfg.setdefault("output_dim", output_dim)
        return MLP(**cfg)
    if head_type in {"adaln_mlp", "adalnmlp", "ada_ln_mlp"}:
        if condition_dim is None:
            raise ValueError("JEPA head type 'adaln_mlp' is only valid for the conditional predictor head")
        context_dim = int(cfg.pop("context_dim", cfg.pop("input_dim", output_dim)))
        cfg["input_dim"] = context_dim
        cfg.setdefault("cond_dim", condition_dim)
        cfg.setdefault("output_dim", output_dim)
        cfg.setdefault("hidden_dim", output_dim)
        return AdaLNMLP(**cfg)
    if head_type == "gated_mlp":
        cfg.setdefault("input_dim", input_dim)
        cfg.setdefault("output_dim", output_dim)
        cfg.setdefault("hidden_dim", output_dim)
        return GatedMLP(**cfg)
    raise ValueError(f"Unknown JEPA head type {head_type!r}")


def build_jepa_from_config(config: dict[str, Any]) -> Lvebm:
    """Instantiate Lvebm; legacy target stop-gradient config keys have no effect."""

    if "context_mask_ratio" in config:
        raise ValueError("Atom masking was removed from JEPA. Remove context_mask_ratio from the config.")
    loss_cfg = config.get("loss", {})
    d_jepa = int(config.get("d_jepa", 512))
    condition_cfg = config.get("condition_encoder", {})
    condition_hidden_dim = int(condition_cfg.get("hidden_dim", 256))
    predictor_input_dim = d_jepa + condition_hidden_dim
    crystal_cfg = config.get("crystal_encoder") or {}
    encoder_output_dim = crystal_cfg.get("output_dim", d_jepa)
    predictor_cfg = config.get("predictor") or {"type": "gated_mlp"}
    if isinstance(predictor_cfg, dict) and predictor_cfg.get("type") == "predictor":
        predictor_cfg = {"num_frames": crystal_cfg.get("num_latents", 32), **predictor_cfg}
    if config.get("prediction_target", "crystal_tokens") != "crystal_tokens":
        raise ValueError("JEPA now predicts crystal_tokens only. Warm-start old weights with --init-from and a new config.")
    if config.get("context_source", "crystal") != "crystal":
        raise ValueError("JEPA context must be a crystal; representatives supply the condition only.")
    peak_cfg = _peak_encoder_config(config)
    if config.get("condition_source", "conformer") != "conformer" and peak_cfg.get("version") == TransformerPeakEncoder.version:
        encoder_capacity = int(peak_cfg.get("p_max", TransformerPeakEncoder.defaults["p_max"]))
        data_capacity = (pxrd_settings(graph_pxrd_config(config))["p_max"]
                         if "graph_manifest_dir" in config or "pxrd" in config else resolve_p_max(config))
        if encoder_capacity < data_capacity:
            raise ValueError(f"peak_encoder.p_max={encoder_capacity} is smaller than the data peak limit {data_capacity}")
    return Lvebm(
        d_jepa=d_jepa,
        crystal_encoder=crystal_cfg,
        condition_encoder=condition_cfg,
        condition_source=config.get("condition_source", "conformer"),
        peak_encoder=peak_cfg,
        projector=build_jepa_head(config.get("projector"), input_dim=encoder_output_dim, output_dim=d_jepa),
        predictor=build_jepa_head(
            predictor_cfg,
            input_dim=predictor_input_dim,
            output_dim=d_jepa,
            condition_dim=condition_hidden_dim,
        ),
        pred_proj=build_jepa_head(config.get("pred_proj"), input_dim=d_jepa, output_dim=d_jepa),
        sigreg_num_projections=int(loss_cfg.get("sigreg_num_projections", 1024)),
        lambda_sig=float(loss_cfg.get("lambda_sig", 0.1)),
        mu_diff=float(loss_cfg.get("mu_diff", 0.0)),
    )


def load_jepa_weights(
    model: Lvebm, state_dict: dict[str, torch.Tensor], *, initialize_lattice: bool = False,
    initialize_predictor: bool = False,
    initialize_conditioning: bool = False,
    initialize_peak_encoder: bool = False,
) -> None:
    """Strict loading, with explicit initialization of changed warm-start branches."""
    weights = {
        key: value for key, value in state_dict.items()
        if not key.startswith("context_encoder.readout")
    }
    if initialize_predictor:
        obsolete = [key for key in weights if key.startswith(("atom_queries.", "atom_decoder.", "atom_output."))]
        if obsolete:
            warnings.warn("Discarding masked-atom output layers for crystal-token prediction; shared weights are retained.",
                          UserWarning, stacklevel=2)
            for key in obsolete:
                del weights[key]
    if initialize_peak_encoder:
        if model.peak_encoder is None:
            raise ValueError("Cannot initialize a PXRD encoder on a model without that branch")
        warnings.warn("Initializing the new PXRD encoder; crystal encoder and predictor weights are retained. "
                      "Optimizer state is not retained.", UserWarning, stacklevel=2)
        weights = {key: value for key, value in weights.items() if not key.startswith("peak_encoder.")}
        weights.update({key: value for key, value in model.state_dict().items() if key.startswith("peak_encoder.")})
    if initialize_conditioning:
        # Whole absent branches may be added/removed explicitly on warm start.
        # A partial branch remains an error; never hide a damaged checkpoint.
        current = model.state_dict()
        for prefix in ("condition_encoder.", "peak_encoder."):
            old_keys = {key for key in weights if key.startswith(prefix)}
            new_keys = {key for key in current if key.startswith(prefix)}
            if bool(old_keys) != bool(new_keys):
                warnings.warn(f"Warm start changes {prefix[:-1]} branch; initializing added weights "
                              "or discarding removed weights. Optimizer state is not retained.",
                              UserWarning, stacklevel=2)
                for key in old_keys:
                    del weights[key]
                weights.update({key: current[key] for key in new_keys})
    lattice_state = {
        key: value for key, value in model.state_dict().items()
        if key.startswith(("context_encoder.local_encoder.lattice_embedding.",
                           "context_encoder.local_encoder.lattice_presence_embedding."))
    }
    if not lattice_state.keys() & weights.keys():
        if not initialize_lattice:
            raise RuntimeError(
                "Checkpoint has no explicit lattice embeddings. First train the new lattice path "
                "with the JEPA --init-from option before evaluating or freezing this encoder."
            )
        warnings.warn(
            "Initializing new lattice embeddings; existing JEPA weights are retained. "
            "The lattice path needs training before evaluation or frozen-encoder use.",
            UserWarning, stacklevel=2,
        )
        weights.update(lattice_state)
    # Partial new-layer states and all unrelated missing/unexpected weights remain errors.
    model.load_state_dict(weights)


def _peak_encoder_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("peak_encoder", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"peak_encoder must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def resolve_p_max(config: dict[str, Any]) -> int:
    """Return the dataset peak count and validate the encoder can handle it."""

    peak_cfg = _peak_encoder_config(config)
    default_capacity = (TransformerPeakEncoder.defaults["p_max"]
                        if peak_cfg.get("version") == TransformerPeakEncoder.version else 128)
    data_p_max = (
        int(config["p_max"])
        if config.get("p_max") is not None
        else int(peak_cfg.get("p_max", default_capacity))
    )
    encoder_default = default_capacity if peak_cfg.get("version") == TransformerPeakEncoder.version else data_p_max
    encoder_p_max = int(peak_cfg.get("p_max", encoder_default))
    if data_p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {data_p_max}")
    if encoder_p_max < data_p_max:
        raise ValueError(
            f"peak_encoder.p_max={encoder_p_max} is smaller than p_max={data_p_max}; "
            "increase peak_encoder.p_max or lower p_max so JEPA batches fit the encoder"
        )
    return data_p_max


def graph_pxrd_config(config):
    """Graph training's explicit target-PXRD contract (separate from legacy data)."""
    settings = dict(config.get("pxrd", {}))
    pxrd_settings(settings)
    return settings


def conditioning_contract(config):
    source = config.get("condition_source", "conformer")
    contract = dict(source=source)
    if source != "conformer":
        peak = config.get("peak_encoder") or {}
        contract.update(pxrd=pxrd_settings(graph_pxrd_config(config)),
                        encoder_version=peak.get("version", PeakEncoder.version),
                        encoder_hidden_dim=int(peak.get("hidden_dim", 256)))
        if contract["encoder_version"] == TransformerPeakEncoder.version:
            settings = {key: peak.get(key, value) for key, value in TransformerPeakEncoder.defaults.items()}
            settings = {key: float(value) if key == "dropout" else int(value) for key, value in settings.items()}
            contract["encoder_hidden_dim"] = settings["hidden_dim"]
            contract["transformer_settings"] = settings
            contract["encoder_output_dim"] = int((config.get("condition_encoder") or {}).get("hidden_dim", 256))
    return contract


def build_dataset_from_config(config: dict[str, Any], split: str = "train"):
    """Build a manifest-backed CIF dataset or a supported LMDB dataset."""

    split = "valid" if split == "val" else split
    if "graph_manifest_dir" in config:
        return CrystalDataset(
            Path(config["graph_manifest_dir"]) / f"{split}.npz",
            cutoff=config.get("crystal_encoder", {}).get("cutoff", 6.0),
            random_block_geometry=split == "train" and config.get("random_block_geometry", True),
            max_samples=config.get(f"max_{split}_samples"),
            random_context=split == "train",
            context_translation_std=config.get("context_translation_std", 0.0),
            context_rotation_degrees=config.get("context_rotation_degrees", 0.0),
            pxrd_config=(graph_pxrd_config(config) if config.get("condition_source", "conformer") != "conformer" else None),
        )
    mp20_path = config.get("mp20_lmdb_path") if split == "train" else config.get(f"mp20_{split}_lmdb_path")
    organic_path = config.get("organic_lmdb_path") if split == "train" else config.get(f"organic_{split}_lmdb_path")
    manifest_path = config.get("manifest_path") if split == "train" else config.get(f"{split}_manifest_path")
    p_max = resolve_p_max(config)

    if mp20_path and organic_path:
        raise ValueError(
            f"Configure only one LMDB backend for split={split!r}: "
            "mp20 or organic"
        )
    lmdb_path = organic_path or mp20_path
    if lmdb_path:
        two_theta_range = config.get("two_theta_range", [5.0, 80.0])
        dataset_cls = OrganicLMDBDataset if organic_path else MP20LMDBDataset
        return dataset_cls(
            lmdb_path,
            n_max=int(config.get("n_max", 64)),
            p_max=p_max,
            wavelength=config.get("wavelength", "CuKa"),
            two_theta_range=(float(two_theta_range[0]), float(two_theta_range[1])),
            augmentation=config.get("augmentation", {}) if split == "train" else config.get("validation_augmentation", {"enabled": False}),
            use_conventional=bool(config.get("use_conventional", False)),
            include_pxrd=bool(config.get("include_pxrd", True)),
        )
    if not manifest_path:
        raise KeyError(f"No dataset path configured for split={split!r}")
    return CIFPXRDDataset(
        manifest_path,
        split=split,
        n_max=int(config.get("n_max", 64)),
        p_max=p_max,
        augmentation=config.get("augmentation", {}) if split == "train" else config.get("validation_augmentation", {"enabled": False}),
        include_pxrd=bool(config.get("include_pxrd", True)),
    )


def has_dataset_split(config: dict[str, Any], split: str) -> bool:
    """Return whether the config names a dataset path for a split."""

    split = "valid" if split == "val" else split
    if "graph_manifest_dir" in config:
        return (Path(config["graph_manifest_dir"]) / f"{split}.npz").exists()
    if split == "train":
        return any(
            key in config
            for key in ("mp20_lmdb_path", "organic_lmdb_path", "manifest_path")
        )
    return (
        f"mp20_{split}_lmdb_path" in config
        or f"organic_{split}_lmdb_path" in config
        or f"{split}_manifest_path" in config
    )
