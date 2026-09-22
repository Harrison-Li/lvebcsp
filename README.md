# LVEB-CSP

Crystal representation learning from complete, perturbed crystals, conditioned on
representative building blocks and their unit-cell multiplicities. Generation is a
separate downstream stage. JEPA uses **no atom masking**.

For a ground-truth crystal `Ct`, construct `Cc` by independently translating and
rotating its molecule/building-block copies. Every atom, its species, and the unit
cell are retained. The model's data flow is:

```python
context_tokens = encoder(Cc)                         # complete perturbed crystal
condition = condition_encoder(encoder(blocks), counts, block_batch)
prediction = predictor(context_tokens, condition)
target_tokens = encoder(Ct)                          # untouched ground truth
loss = mse(prediction, target_tokens) + slot_regularization
```

`Lvebm` shares one `UniversalEncoder` between the representatives, context, and target.
Both crystal branches receive MSE and SIGReg gradients, retaining the existing LeWM
objective. There is no EMA or stop-gradient target. Historical `stop_gradient` and
`stop_gradient_target` keys have no effect.

Each crystal is represented by `[B, num_latents, d_jepa]` tokens, without global
pooling. Current configs use `[B, 32, 512]`: each internal 256-wide token is projected
to 512 features. Set `crystal_encoder.output_dim` and `d_jepa` to 256 for
`[B, 32, 256]` instead. The predictor preserves the learned slot ordering and
predicts all clean crystal tokens. No atom queries or atom matching are used.

`LocalAtomTransformer` owns atom and lattice embeddings, periodic distances,
Gaussian radial features, the polynomial cutoff, and attention message passing.
It returns aggregated per-atom features `[N, atom_dim]`. `UniversalEncoder`
delegates graph processing to `local_encoder`, then compresses those features with
learned queries and latent attention. Its `compress(atoms, graph_index)` method
also accepts prepared atom features directly, without coordinates or edges.
Representatives have no target
lattice; their absent-cell embedding distinguishes them from periodic crystals.
The original cell determines periodic distances, so jointly rotating coordinates
and cell leaves the representation invariant. The cutoff is applied once per
local attention layer, inside softmax normalization; radial features are not
multiplied by the cutoff. Lattice information remains available without neighbors.

Default training uses `loss = prediction_mse + 0.1 * sigreg_slots`.
SIGReg averages the context and target branches equally and sees
`[32, B_local, 512]`: each slot defines a distribution, with crystals on each
GPU as samples. This is per-slot regularization, without pooling the 32 tokens.
Difference SIGReg is disabled (`mu_diff: 0`) in the model defaults and training
presets, and its computation is skipped. Per-slot SIGReg does not explicitly
penalize similarity between slots. The optional difference implementation remains
available for historical experiments via an explicit positive `mu_diff`.

Cosine diagnostics flatten the crystal tokens. `cos_prediction` compares predicted
and clean tokens; `cos_context_target` compares the paired raw context and target;
`cos_offdiag` compares different crystals in the local batch. Standard deviations
also use crystal tokens. These diagnostics and regularizers do not establish
packing retention or structural reconstruction. The separate `train_crystal_latent`
entry point below implements latent flow matching and crystal reconstruction.

Implementation: `models/jepa.py` defines the model and objective;
`data/crystal_dataset.py` constructs the two crystal views; `train/train_jepa.py`
contains training, validation, logging, and resume logic. Model/data factories and
checkpoint migration live in `train/jepa_setup.py`, with reusable scheduling and
early stopping in `train/training_utils.py`.

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
    "crystal_context": Batch.from_data_list(perturbed_crystal_graphs),
    "target": Batch.from_data_list(complete_crystal_graphs),
}
```

In the current CCDC source, representative species/coordinate banks come from `mol_atom_type_2` and
`mol_atom_pos_2`, with boundaries in `mol2_group_slices`. Counts come from
`repeat_dict_2` in `mol_group_id_2` order. `mol_bb_num_vec_2` contains block atom
counts, not multiplicities. Some deduplicated tables omit one-atom components;
the adapter restores their representatives/counts from `mol_atom_type` and
`mol_group_slices`, using zero-position templates without target geometry.
`random_block_geometry` samples representative
geometries from the stored bank during training; validation/test use geometry
zero. Target poses and lattice are not passed to
the representative encoder. A cutoff graph is built for each representative
independently. Complete crystal positions are fractional coordinates converted
to Cartesian coordinates using `lattice_matrix`; periodic edges include source
image shifts. `block_instance_id` identifies each target atom's block copy in
the source cell for context perturbation and packing evaluation. There is no fixed atom-count limit or
padding in the loader.

### Context translation and rotation

All JEPA presets now use `context_translation_std: [0.05, 0.75]` and
`context_rotation_degrees: [5.0, 90.0]`. A standard deviation and rotation angle
are sampled uniformly per crystal. Each molecular copy gets an independent
Gaussian translation in Cartesian angstroms and an independent random rotation
axis. The standard-deviation range does not bound the realized displacement.

Finite molecules are first unwrapped along covalent bonds, rotated around their
own centroid, and translated. Coordinates are then wrapped into the unchanged
cell, and all periodic radius edges/image shifts are rebuilt. Bond inference is
independent of the encoder cutoff. Disconnected or periodically bonded blocks
are translated only, because they have no unique finite molecular pose. The
clean target and representative conditions are unchanged. These perturbations
preserve internal molecular geometry but are unrelaxed and may create overlaps.

Both controls accept a scalar or `[min, max]`. Training samples fresh contexts;
validation/test use fixed per-record contexts without changing the training RNG.
Setting both controls to zero produces a separate copy of the complete clean
crystal, with the same model path. Every context contains all target atoms.

```bash
NPROC_PER_NODE=4 scripts/train_crystal_jepa.sh \
  --context-translation-std 0.05 0.75 \
  --context-rotation-degrees 5 90
```

`configs/crystal_jepa.yaml` is the single-device preset, and
`configs/crystal_jepa_ddp.yaml` is the launcher default. The existing
`configs/crystal_jepa_packing_ddp.yaml` also uses complete perturbed contexts and
keeps its own output directory. Evaluate packing sensitivity separately from
prediction loss: the predictor should recover clean tokens while the encoder
still distinguishes different packings.

The experimental `configs/crystal_jepa_pxrd.yaml` supports `condition_source:
pxrd` or `combined`; the existing presets default to `conformer`. PXRD uses the
chosen clean target's exact cell and positions, simulated with CuKa = 1.5406 Å.
It is an observed input to the predictor. The crystal encoder still sees only
crystal graphs. The PXRD preset uses `transformer_peaks_v1`: up to 80 peaks,
128-wide peak embeddings, three Transformer layers and four attention heads.
Reciprocal spacing and normalized intensity pass through separate MLPs; learned
rank embeddings use canonical decreasing-d order. Padding is masked before the
MLPs and in attention. The CLS token is projected to a 256-dimensional condition,
broadcast over the 32 crystal slots, and concatenated with each 512-dimensional
crystal token for the **gated MLP predictor**. The Transformer operates inside the
PXRD encoder. CLS aggregation still compresses the pattern to one vector; packing
retrieval with correct versus swapped PXRD is needed to establish a training gain.

The earlier `joint_peaks_v1` encoder remains available for its existing checkpoints;
it applies joint nonlinear peak processing followed by masked mean pooling.
`combined` adds the PXRD vector to the conformer/count condition without changing
predictor width. All modes retain the shared encoder, target gradients, prediction
MSE and per-slot SIGReg, with difference SIGReg off.

```bash
NPROC_PER_NODE=1 scripts/train_crystal_jepa.sh \
  --config configs/crystal_jepa_pxrd.yaml \
  --init-from outputs/crystal_jepa/train_packing_ddp/best.ckpt
# To test both inputs, add --condition-source combined and a distinct --output-dir.
```

For PXRD-only conditioning, use the dedicated launcher. With no initialization
flag it starts from scratch; `--init-from` starts from existing model weights
with a fresh optimizer, and `--resume` continues a matching PXRD checkpoint.
The default output directory is `outputs/crystal_jepa/train_pxrd_transformer80`.

```bash
NPROC_PER_NODE=1 scripts/train_crystal_jepa_pxrd.sh
# Warm-start the existing crystal encoder and predictor:
NPROC_PER_NODE=1 scripts/train_crystal_jepa_pxrd.sh \
  --init-from outputs/crystal_jepa/train_packing_ddp/best.ckpt \
  --output-dir outputs/crystal_jepa/train_pxrd_warmstart
```

Changing condition source or PXRD preprocessing requires `--init-from` and a new
output directory; optimizer resume requires the same conditioning contract.
Warm starts retain shared crystal/predictor weights and initialize missing input
branches explicitly. Changing the PXRD encoder version with `--init-from`
initializes that encoder from scratch; optimizer resume also checks Transformer
depth, heads, dropout, widths and peak capacity. Graph datasets cache target patterns by species, positions,
cell, simulation settings and pymatgen version. Simulation never uses the
perturbed context. If a target is replaced in an experiment, replace its PXRD too.
The scan retains the first `pxrd.p_max` peaks in increasing angle after the
relative intensity threshold; caching avoids repeated simulation across epochs.
Inference with PXRD or combined conditioning requires `peak_d`, `peak_i` and a
boolean `peak_mask` for the desired target observation. The existing downstream
molecule-conditioned latent generator requires a conformer or combined checkpoint.

`scripts/benchmark_jepa_pxrd.py` compares all three conditions on identical
native/alternative target batches, using the earlier audited graph caches. It
measures encoder geometry sensitivity separately from prediction and includes
within-parent PXRD swaps, intensity reassignment and context swaps. Run with
`--output-dir outputs/crystal_jepa/pxrd_conditioning_benchmark_20260918_fullpeaks --workers 8`; settings,
initial/endpoint measurements, checkpoint hashes and a report are saved there.
These experiments use simulated PXRD and synthetic, unrelaxed arrangements;
experimental-pattern generalization and full CSP performance require separate tests.

## Preparation and training

From the repository root, prepare disjoint split manifests without modifying
any input LMDB:

```bash
PYTHONPATH=src python scripts/prepare_crystal_jepa.py \
  --sources \
  /nanolab/libo/lvebpxrd/lvebcsp/dataset/train.lmdb \
  /nanolab/libo/lvebpxrd/lvebcsp/dataset/valid.lmdb \
  /nanolab/libo/lvebpxrd/lvebcsp/dataset/test.lmdb \
  --output-dir dataset/crystal_jepa/splits --workers 8
```

The prepared manifests live in `dataset/crystal_jepa/splits`, with their LMDB
paths recorded in `sources.json`. Filtered copies live in
`dataset/crystal_latent/splits`. Relocation preserves every existing split record.
Preparation removes
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
The manifests do not reference the old `test_500.lmdb`, so that file is not
required to train or evaluate these splits.

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
needed for larger crystals. `num_workers: 8` is per process for each loader;
workers use spawn and remain alive across epochs. Both JEPA and latent training
default to `precision: fp32`, with autocast disabled. BF16 requires an explicit
`--precision bf16` or `--precision auto` override.
The first batches include worker startup and GPU warm-up. For steady-state
throughput, tune `--num-workers` for the server's available CPU cores and storage
speed. Radius graphs are built in those CPU workers on each fetch. Console metrics
refresh every 20 steps (`--log-every-steps`); full epoch metrics still include every
batch. Metric totals stay on the GPU between updates, and padding sizes are
computed during CPU collation to avoid synchronizing just to read graph sizes.
AdamW starts at `1e-4`. PyTorch's
[ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
adapts the learning rate to the combined validation loss above: it halves the rate
after four consecutive epochs without at least a `1e-4` relative improvement,
with a minimum rate of `1e-6`. Every rank steps the scheduler after the validation
loss is reduced across ranks. When no validation split is provided, the training
loss is used. The scheduler's state and current rate are restored on resume.

Training shards are reshuffled each epoch, with incomplete batches dropped so
all ranks take the same number of steps. Each GPU computes SIGReg on its local
batch with its own random projections; DDP averages model gradients across GPUs.
Validation uses one fixed permutation, generated from `seed` before splitting
indices across GPUs. This mixes related crystal families while keeping batches
identical across epochs. Each crystal is visited once without duplicated padding,
then sample-weighted losses are reduced across ranks. The evaluation code seeds
random projections and token pairings externally and restores training RNG state
afterward. SIGReg and embedding
diagnostics are computed within each local batch, so keep batch sizes and GPU
count fixed when comparing these diagnostics.

Checkpoints record the validation order, seed, sample count, batch size, GPU count,
and batch limit. Resuming an older checkpoint or changing these settings resets
best-loss, early-stopping, and plateau-scheduler tracking because the old losses
are not comparable. Model weights, optimizer state, and the current learning rate
are restored; step/epoch scheduler state is retained. New validation losses become
the baseline. Mixing families fixes the ordering bias; packing/reconstruction
quality still needs an independent geometry metric.

Only rank 0 logs and writes `training_config.yaml`, `best.ckpt`, and `last.ckpt`.
Checkpoints contain model, optimizer, scheduler, epoch, step count, early-stopping
state, and validation metrics. Resume into the same output directory:

```bash
NPROC_PER_NODE=4 bash scripts/train_crystal_jepa.sh \
  --config /path/to/crystal_jepa_run/training_config.yaml \
  --resume /path/to/crystal_jepa_run/last.ckpt
```

Resume continues at the next epoch; it does not replay worker augmentation RNG
states exactly. Keep the GPU count, batch size, and scheduler configuration fixed
to retain the intended optimization schedule. Current output directories are
`train_rigid_context` and `train_rigid_context_ddp`.

Checkpoints record the `mse_sigreg_context_target_slots_differences` objective and
`independent_rigid_blocks_v1` context version. Older checkpoints require
`--init-from` with the new config and a new output directory, because their context
construction or objective differs. Changing either perturbation range, either
regularizer weight, or the SIGReg projection count also requires a new run.

To continue the packing checkpoint with difference SIGReg removed, warm-start a
new output directory using the updated packing preset:

```bash
NPROC_PER_NODE=4 bash scripts/train_crystal_jepa.sh \
  --config configs/crystal_jepa_packing_ddp.yaml \
  --init-from outputs/crystal_jepa/train_packing_ddp/best.ckpt \
  --output-dir outputs/crystal_jepa/train_packing_slots_only_ddp
```

This starts fresh optimizer, scheduler, and loss-monitoring state. New checkpoints
save `mu_diff` explicitly; checkpoints that omitted it used the historical `0.01`
default. Resuming with changed weights is rejected to avoid mixing loss histories.

Warm starts preserve compatible encoder, conditioner, and predictor weights.
Obsolete atom prediction heads and global readout weights are discarded. Missing
lattice embeddings are initialized for training; evaluation and frozen-encoder
use require a checkpoint with the trained lattice path. Masked-atom training and
`--context-mask-ratio` have been removed. Remove `context_mask_ratio` from old YAML
files when creating a new run.

```bash
NPROC_PER_NODE=4 bash scripts/train_crystal_jepa.sh \
  --init-from /path/to/old_jepa/last.ckpt \
  --output-dir outputs/crystal_jepa/train_rigid_context_ddp
```

For a short installation check:

```bash
NPROC_PER_NODE=4 bash scripts/train_crystal_jepa.sh \
  --max-epochs 1 --max-train-batches 2 --max-eval-batches 2 \
  --max-train-samples 640 --max-valid-samples 160 \
  --output-dir outputs/crystal_jepa/ddp_check
```

The multi-GPU config disables external logging and end-of-training packing
evaluation. The original single-device command remains available.

## Crystal latent flow matching and reconstruction

`models/eaf.py` implements `EAF` (also exported as `CrystalLatentDiffusion`
for the existing entry points), following
[ELF's embedding-space flow and final decoding design](https://github.com/lillian039/ELF/tree/pytorch_elf).
The trajectory has shape `[B, 32, 512]`. Time and denoise/decode mode embeddings
condition a shared DiT backbone. The JEPA encoder, per-token projector, and
multiplicity conditioner remain frozen in evaluation mode. Trainable projections
adapt their outputs to the flow backbone.

The supplied building blocks fix the generated species and atom count.
`LocalAtomDecoder` expands each latent into `max_atoms: 32` learned, species-aware
atom queries attending to that latent alone. There is no atom self-attention.
A separate head predicts six Gram-Cholesky parameters, which decode to a
positive-definite cell. The default capacity is **32*32 = 1024 atoms**; increase
`latent_diffusion.max_atoms` for larger crystals. Padded output slots are masked
and excluded from the generated graph.

The model reuses `FlowMatchingInterpolant` with `center_noise=False`: Gaussian
latent noise must retain variation in its token mean. Existing coordinate users
keep centered noise by default. Training samples one time per crystal and uses:

```text
z0 = frozen_JEPA(true_crystal)
zt = (1 - t) * noise + t * z0
v0 = z0 - noise
z_hat = zt + (1 - t) * velocity(zt, t, condition)
generated_crystal = decode(z_hat, condition, supplied_species)
loss = loss.flow * MSE(velocity, v0)
     + loss.jepa * MSE(frozen_JEPA(generated_crystal), z0)
```

Both weights default to 1. The estimated clean endpoint is decoded once at
`t=1` during training, so the JEPA term trains the decoder and velocity predictor
without a rollout. Periodic neighbors are rebuilt from the generated geometry
with pymatgen's CPU radius search. Only neighbor selection is detached;
Cartesian positions and cell translations retain gradients through the frozen
encoder. No coordinate/lattice reconstruction loss or atom assignment is used.
The encoder's learned latent queries give the 32 embeddings consistent slots
regardless of atom order, allowing direct token MSE without Hungarian matching.

At sampling time, Euler integration uses only the 32 latent tokens from `t=0`
to `t=1`, followed by one decode. Source and generated graph encoding costs
remain part of training, but generated graphs are never built inside the
sampling loop. The collator checks composition and keeps target graphs in their
source order; atom maps and copy correspondence are unnecessary for the loss.
JEPA embedding agreement does not establish exact structural reconstruction;
use the held-out structure-matching evaluation below to assess that separately.

Filter incomplete representative tables once before training. This compares
building-block compositions and multiplicities with the actual crystal copies,
excluding mismatches from train/valid/test while retaining permuted atoms and
copies. It preserves the original split membership and order, leaves the source
LMDBs unchanged, and writes `excluded.jsonl` and `filter_report.json` alongside
the filtered manifests. All ranks then use the same dataset; no samples are
dropped independently inside DataLoader workers.

```bash
PYTHONPATH=src python scripts/filter_crystal_data.py \
  --manifest-dir dataset/crystal_jepa/splits \
  --output-dir dataset/crystal_latent/splits --workers 8
```

Train on four GPUs using those filtered manifests, with 80 crystals per GPU and
adaptive initial learning rate `1e-4`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/train_crystal_latent.sh \
  --init-from /path/to/jepa/best.ckpt \
  --num-workers 4
```

`--init-from` imports the saved JEPA representation architecture and weights,
discarding any removed global readout. The checkpoint must include the explicit
lattice embeddings; first fine-tune older weights with JEPA `--init-from`.
The JEPA predictor is not part of the downstream model. A new run
requires `--init-from` because the representation is frozen; `--resume` restores
an EAF checkpoint. Settings are in `configs/crystal_latent.yaml`.
The launcher defaults to FP32 and supports CPU/Gloo and CUDA/NCCL, optional BF16, sample-weighted validation
without duplicated records, adaptive scheduling, and full optimizer/scheduler
resume. Default outputs go to `outputs/crystal_latent/train_local`. The previous
atom-query decoder checkpoints cannot be resumed with this architecture. Start a
new run initialized from a pretrained JEPA checkpoint.
Only checkpoints from the new decoder can be resumed:

```bash
NPROC_PER_NODE=4 bash scripts/train_crystal_latent.sh \
  --config outputs/crystal_latent/train_local/training_config.yaml \
  --resume outputs/crystal_latent/train_local/last.ckpt
```

For a short installation check, add `--max-epochs 1 --max-train-batches 2
--max-eval-batches 2 --max-train-samples 640 --max-valid-samples 160` and a separate
`--output-dir`. This checks execution, not crystal-generation quality.

Sample using only the representatives and counts from a dataset item:

```bash
PYTHONPATH=src python -m lvebcsp.inference.sample_crystal_latent \
  --checkpoint outputs/crystal_latent/train_local/best.ckpt \
  --manifest-dir dataset/crystal_latent/splits --split test --index 0 \
  --num-samples 4 --steps 100 \
  --output-dir outputs/crystal_latent/samples
```

The dataset option loads a normal record but discards its target before calling
the model. For entirely new inputs, use `--condition condition.pt` instead of
`--manifest-dir`, where the file is saved with
`torch.save({"context": representative_graphs, "multiplicity": counts}, path)`.
Representatives are PyG `Data` objects with `z`, centered Cartesian `pos`, and
isolated `edge_index` covering the encoder cutoff. No target coordinates or cell
are required. The sampler writes `samples.pt` (including all 32 generated tokens)
and one CIF per candidate with the supplied composition. Sampling uses Euler
integration in FP32 by default.

The Python interface uses the same target-free batch:

```python
from lvebcsp.data.crystal_latent import collate_crystal_latents
from lvebcsp.train.train_jepa import batch_to_device

batch = collate_crystal_latents([
    {"context": representative_graphs, "multiplicity": counts},
])
batch = batch_to_device(batch, device)
samples = model.eval().sample(batch, steps=100)
# samples["tokens"]: [B, 32, 512]; samples["frac_coords"]: [B, 1024, 3]
# samples["atom_mask"] and samples["atom_types"] preserve the supplied composition.
```

## Crystal reconstruction match rate

This test encodes each complete crystal into 32 tokens and decodes one
reconstruction with the representative/count condition. Coordinates and lattice
are predicted; atom species and count come from the condition. The report includes
JEPA embedding MSE and permutation-aware structure matching. It uses clean tokens
and no flow sampling.

The structure matcher and metrics are reused from the copied dlmcsp
`mp20_realpxrd_solver.py`: `ltol=0.3`, `stol=0.5`, `angle_tol=10`, and a strict
`StructureMatcher.get_rms_dist(...)[0] < 0.5` success rule. Primitive-cell reduction
and lattice scaling retain the original defaults. This RMSD is normalized by
the average free length per atom; the threshold is **not 0.5 angstrom**.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_crystal_reconstruction.sh \
  --checkpoint outputs/crystal_latent/train_local/best.ckpt \
  --manifest-dir dataset/crystal_latent/splits \
  --split valid --max-cases 1000 --batch-size 32 --num-workers 4 \
  --output-dir outputs/crystal_latent/reconstruction
```

Cases are sampled uniformly without replacement using the saved seed. Use
`--max-cases 0` for the full split and `--split test` for final evaluation.
Outputs include `metrics.json`, per-case JSON/CSV, exact manifest indices, the
checkpoint epoch/hash, and reference/reconstruction CIFs for every matched case
plus the first 16 examples (`--save-cifs` changes the example count).
A target roundtrip control checks structure conversion using the same matching
protocol. Failed matches remain in the denominator. Candidate validity requires
atoms and a usable lattice; it does not establish physical validity.

## Packing evaluation

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m lvebcsp.eval.crystal_jepa \
  --checkpoint outputs/crystal_jepa/train_rigid_context_ddp/best.ckpt --split test \
  --device cuda --max-samples 32 --max-families 32 \
  --output outputs/crystal_jepa/train_rigid_context_ddp/packing_test.json
```

The evaluator reports rigid rotation/translation invariance, sensitivity to
independent translations of building-block instances, native-versus-perturbed energy,
and within-family candidate retrieval (top-1 and MRR). The latter compares
different refcodes with matching inferred heavy-atom connectivity hashes and
multiplicities, excluding equivalent packings with `StructureMatcher`. It also
reports distances between their target embeddings. If no eligible group exists,
retrieval metrics are null and the query count is zero.

Queries include the complete perturbed reference crystal and its lattice. These
scores measure recovery of clean packing from a perturbed reference. The evaluator
uses the checkpoint's translation and rotation ranges with deterministic contexts;
they do not measure molecule-only CSP.

Synthetic perturbations are not verified polymorphs. Inferred heavy-atom
connectivity does not establish bond orders, hydrogens, or experimental polymorph
labels. The diagnostics test the representation and ranking behavior; an
identical building-block context can legitimately have multiple compatible targets.
Full validation packing evaluation runs on the best checkpoint after training
when `packing_eval_samples` is nonzero. Use the test split for final evaluation.

The frozen-encoder MLP probe predicts periodic inter-block distance histograms
from complete-crystal token arrays. It compares the trained encoder, three random
encoders, and context predictions from complete perturbed crystals.
A separate MLP predicts descriptor
changes from embedding differences for translations and rotations at fixed
building blocks, multiplicities, and cell. Encoder weights stay frozen throughout.
The cache retains all tokens; the MLP flattens them without averaging. Evaluating
an old pooled checkpoint now bypasses its removed readout and probes its tokens,
so these results differ from the earlier pooled-vector experiment. Reports record
the feature shape and whether the source checkpoint had a global readout.

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python -m lvebcsp.eval.crystal_encoder_probe \
  --checkpoint outputs/crystal_jepa/train_ddp_ctx_tgt/last.ckpt \
  --output-dir outputs/crystal_jepa/encoder_probe
```

Settings are in `configs/crystal_encoder_probe.yaml`. The evaluator saves a
checkpoint snapshot, group-isolated samples, cached FP32 features, trained MLPs,
per-run predictions, and `report.md`. Use `--prepare-only` for feature extraction;
use `--fit-only --checkpoint outputs/crystal_jepa/encoder_probe/frozen_checkpoint.pt`
with the same output directory to refit from the cache. Evaluation groups are
filtered against sampled probe-training chemistry; this does not establish
chemical-identity isolation from the entire encoder-pretraining corpus.

Run regression checks with `OMP_NUM_THREADS=1 python -m pytest -q`.
