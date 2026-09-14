# LVEB-CSP

Crystal representation learning with representative building blocks and their
unit-cell multiplicities. Generation is a separate downstream stage.

`Lvebm` shares one `UniversalEncoder` between representatives and complete crystals.
Context and target branches both propagate gradients into the shared encoder
by default (`stop_gradient: false`), matching LeWM. Training uses raw latent
MSE plus SIGReg on trainable context embeddings. Candidate energy is the same
mean squared latent distance. Context and target embedding standard deviations
are logged to help detect collapse.

## Dataset and preprocessing

`preprocessing_utils.py` copies `build_crystal_graph` from
[Meta's all-atom-diffusion-transformer](https://github.com/facebookresearch/all-atom-diffusion-transformer/tree/b9ce505f170597a7c8ca50d13ce8e15df21cf8c9/src/data/components).
It retains the upstream graph-array fields and CrystalNN/none branches, adds
radius neighbors for `UniversalEncoder`, and removes the canonical-cell assertion
so source cells and rotated evaluation cells remain valid.
`crystal_dataset.py` reuses the tensor-construction block from upstream
`CrystalDataset.__getitem__`, with `z`, `edge_index`, and `edge_shifts` for our
encoder and unused diffusion fields omitted.
The original attribution and CC BY-NC 4.0 license are in
`third_party/all-atom-diffusion-transformer/`.

LMDB loading, representative building-block batches, and source-family split
preparation are project adapters; the upstream code has no equivalent for them.
The existing LMDB serves as the cache, avoiding a whole-dataset in-memory cache
of 1.2 million structures. No CSV/CIF round trip is performed. Source cells and
atom ordering are preserved; primitive/conventional cell conversion would also
require updating the supplied multiplicities and block-instance groups.

The model and graph terminology are shared across organic and inorganic crystals.
The current source adapter in `process_one` reads the CCDC LMDB schema described
below; additional source formats will need their own field mapping and suitable
building-block definitions.

Each batch contains:

```python
{
    "context": Batch.from_data_list(representative_graphs),
    "block_batch": tensor([0, 0, 1]),  # which crystal each representative belongs to
    "multiplicity": tensor([2, 4, 3]), # copies in that crystal's source unit cell
    "target": Batch.from_data_list(complete_crystal_graphs),
}
```

In the current CCDC source, representative species/coordinate banks come from `mol_atom_type_2` and
`mol_atom_pos_2`, with boundaries in `mol2_group_slices`. Counts come from
`repeat_dict_2` in `mol_group_id_2` order. `mol_bb_num_vec_2` contains block atom
counts, not multiplicities. `random_block_geometry` samples representative
geometries from the stored bank during training; validation/test use geometry
zero. Target poses and lattice are not passed to
the representative encoder. A cutoff graph is built for each representative
independently. Complete crystal positions are fractional coordinates converted
to Cartesian coordinates using `lattice_matrix`; periodic edges include source
image shifts. `block_instance_id` identifies each target atom's block copy in
the source cell for packing evaluation. There is no fixed atom-count limit or
padding in the loader.

## Preparation and training

From the repository root, prepare disjoint split manifests without modifying
any input LMDB:

```bash
PYTHONPATH=src python scripts/prepare_crystal_jepa.py \
  --sources \
  /instrument/cif_xray/users/xuhengyi/pxrd_mol_final/data/ccdc_mol_all_valid_40w_filter/train.lmdb \
  /instrument/cif_xray/users/xuhengyi/pxrd_mol_final/data/ccdc_mol_all_valid_40w_filter/valid.lmdb \
  /instrument/cif_xray/users/xuhengyi/pxrd_mol_final/data/ccdc_mol_all_valid_40w_filter/test.lmdb \
  /instrument/cif_xray/users/xuhengyi/pxrd_mol_final/data/ccdc_mol_all_valid_40w_filter/test_500.lmdb \
  --output-dir outputs/crystal_jepa/splits --workers 8
```

The prepared manifests already exist in this workspace. Preparation removes
exact duplicate records, groups filenames before their `_std_` suffix and then
removes trailing refcode digits, and assigns whole families deterministically
using seed 42. Every cell/augmentation variant and numbered refcode in a family
stays in one split. Source-index caches let subsequent preparation reuse the
completed scan. `audit.json` reports independently checked family/hash overlap.
This is a source-family split, not a verified chemical-identity split across
unrelated refcodes.

The current manifests contain 1,073,436 training, 56,557 validation, and 69,995
test records. All 500 entries in the old `test_500.lmdb` duplicated records in
the source pool and were removed. Cross-split refcode-family and exact-record
overlap are both zero.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m lvebcsp.train.train_jepa \
  --config configs/crystal_jepa.yaml
```

For one server with four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/train_crystal_jepa.sh \
  --manifest-dir /path/to/splits \
  --output-dir /path/to/crystal_jepa_run
```

The launcher uses `configs/crystal_jepa_ddp.yaml` and PyTorch DistributedDataParallel
with one process per GPU. Omit `NPROC_PER_NODE` to use all visible GPUs, or set it
to 8 for eight GPUs. `--config` selects another YAML. Activate an environment with
CUDA-enabled PyTorch and this project's dependencies (`pip install -e .`) first.
On another server, copy the prepared manifests and ensure the LMDB paths in their
`sources.json` point to accessible files; `--manifest-dir` selects the manifests.
All launcher paths are resolved from the repository root.

`batch_size: 80` and `eval_batch_size: 80` are **per GPU**; four GPUs give a global
training batch of 320. Adjust them with `--batch-size` and `--eval-batch-size` if
needed for larger crystals. `num_workers: 2` is per process for each loader;
workers use spawn and remain alive across epochs. `precision: auto` selects BF16
on supported GPUs and FP32 otherwise; use `--precision fp32` to disable autocast.
AdamW starts at `1e-4`. PyTorch's
[ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
adapts the learning rate to the validation MSE + SIGReg loss: it halves the rate
after four consecutive epochs without at least a `1e-4` relative improvement,
with a minimum rate of `1e-6`. Every rank steps the scheduler after the validation
loss is reduced across ranks. When no validation split is provided, the training
loss is used. The scheduler's state and current rate are restored on resume.

Training shards are reshuffled each epoch, with incomplete batches dropped so
all ranks take the same number of steps. Training SIGReg uses differentiable
gathering of the global batch and common random projections, in FP32.
Validation visits each crystal once without duplicated padding, then reduces
sample-weighted losses across ranks. Validation SIGReg and embedding diagnostics
are computed within each local batch, so keep validation batch size and GPU count
fixed when comparing these diagnostics.

Only rank 0 logs and writes `training_config.yaml`, `best.ckpt`, and `last.ckpt`.
Checkpoints contain model, optimizer, scheduler, epoch, step count, early-stopping
state, and validation metrics. Resume into the same output directory:

```bash
NPROC_PER_NODE=4 bash scripts/train_crystal_jepa.sh \
  --config /path/to/crystal_jepa_run/training_config.yaml \
  --resume /path/to/crystal_jepa_run/last.ckpt
```

Resume continues at the next epoch; it does not replay worker augmentation RNG
states exactly. Keep the same GPU count, batch size, and scheduler configuration
to retain the intended optimization schedule. For a short installation check:

```bash
NPROC_PER_NODE=4 bash scripts/train_crystal_jepa.sh \
  --max-epochs 1 --max-train-batches 2 --max-eval-batches 2 \
  --max-train-samples 640 --max-valid-samples 160 \
  --output-dir outputs/crystal_jepa/ddp_check
```

The multi-GPU config disables external logging and end-of-training packing
evaluation. The original single-device command remains available.

## Packing evaluation

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m lvebcsp.eval.crystal_jepa \
  --checkpoint outputs/crystal_jepa/train/best.ckpt --split test \
  --device cuda --max-samples 32 --max-families 32 \
  --output outputs/crystal_jepa/train/packing_test.json
```

The evaluator reports rigid rotation/translation invariance, sensitivity to
independent translations of building-block instances, native-versus-perturbed energy,
and within-family candidate retrieval (top-1 and MRR). The latter compares
different refcodes with matching inferred heavy-atom connectivity hashes and
multiplicities, excluding equivalent packings with `StructureMatcher`. It also
reports distances between their target embeddings. If no eligible group exists,
retrieval metrics are null and the query count is zero.

Synthetic perturbations are not verified polymorphs. Inferred heavy-atom
connectivity does not establish bond orders, hydrogens, or experimental polymorph
labels. The diagnostics test the representation and ranking behavior; an
identical building-block context can legitimately have multiple compatible targets.
Full validation packing evaluation runs on the best checkpoint after training
when `packing_eval_samples` is nonzero. Use the test split for final evaluation.

Run regression checks with `OMP_NUM_THREADS=1 python -m pytest -q`.
