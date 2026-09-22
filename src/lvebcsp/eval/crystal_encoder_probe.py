"""Frozen-encoder MLP probes of periodic inter-block distance distributions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

from ase import Atoms
from ase.data import covalent_radii
import networkx as nx
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from scipy.spatial.transform import Rotation
import torch
from torch import nn
import yaml

from lvebcsp.data.crystal_dataset import CrystalDataset, collate_crystal_graphs
from lvebcsp.eval.crystal_jepa import atoms_to_graph
from lvebcsp.train.train_jepa import batch_to_device, build_jepa_from_config, load_jepa_weights


def packing_descriptor(graph, cutoff=6.0, bins=24):
    """Unwrap finite blocks and count neighbors belonging to other physical copies.

    An image of the same block is a different physical copy unless its edge
    translation equals the unwrapping offset difference of its two atoms.
    Connectivity is used only to assign those offsets, never as a probe input.
    """
    pos, cell, numbers = graph.pos.numpy().astype(float), graph.cell[0].numpy().astype(float), graph.z.numpy()
    _, groups = np.unique(graph.block_instance_id.numpy(), return_inverse=True)
    structure = Structure(cell, numbers, pos, coords_are_cartesian=True)
    dst, src, shifts, distance = structure.get_neighbor_list(cutoff + 1e-4)
    shifts = np.rint(shifts).astype(int)
    radii = covalent_radii[numbers]
    bonds = ((groups[src] == groups[dst]) & (distance > 0.2)
             & (distance < 1.2 * (radii[src] + radii[dst])))
    adjacency = [[] for _ in pos]
    for a, b, image in zip(dst[bonds], src[bonds], shifts[bonds]):
        adjacency[a].append((b, image))
    offsets = np.zeros((len(pos), 3), dtype=int)
    visited = np.zeros(len(pos), dtype=bool)
    for group in range(groups.max() + 1):
        members = np.flatnonzero(groups == group)
        root = int(members[0])
        visited[root] = True
        stack = [root]
        while stack:
            a = stack.pop()
            for b, image in adjacency[a]:
                proposed = offsets[a] + image
                if visited[b]:
                    if not np.array_equal(offsets[b], proposed):
                        raise ValueError("Block has inconsistent periodic connectivity")
                else:
                    offsets[b], visited[b] = proposed, True
                    stack.append(b)
        if not visited[members].all():
            raise ValueError("Block is disconnected under the covalent-radius criterion")
    intra = (groups[src] == groups[dst]) & (shifts == offsets[src] - offsets[dst]).all(axis=1)
    rounded_distance = np.round(distance, 4)
    inter = ~intra & (rounded_distance <= cutoff)
    histogram = np.histogram(rounded_distance[inter], bins=np.linspace(0, cutoff, bins + 1))[0]
    clash_ratio = float(np.min(distance[inter] / (radii[src[inter]] + radii[dst[inter]]))) if inter.any() else float('inf')
    return histogram.astype(np.float32) / len(pos), pos + offsets @ cell, groups, clash_ratio


def rebuild_graph(graph, positions, cutoff, cell=None):
    atoms = Atoms(numbers=graph.z.numpy(), positions=positions,
                  cell=graph.cell[0].numpy() if cell is None else cell, pbc=True)
    atoms.wrap()
    result = atoms_to_graph(atoms, cutoff)
    result.block_instance_id = graph.block_instance_id.clone()
    if 'atom_map' in graph:
        result.atom_map = graph.atom_map.clone()
    return result


def block_signature(item):
    """Use the same inferred heavy-atom connectivity rule as family retrieval."""
    signatures = []
    for graph, count in zip(item['context'], item['multiplicity']):
        numbers, pos = graph.z.numpy(), graph.pos.numpy()
        distance = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
        radii = covalent_radii[numbers]
        bonds = np.triu((distance > 0.2) & (distance < 1.2 * (radii[:, None] + radii[None, :])), 1)
        molecule = nx.Graph()
        molecule.add_nodes_from((i, {'element': str(z)}) for i, z in enumerate(numbers))
        molecule.add_edges_from(zip(*np.where(bonds)))
        signatures.append((nx.weisfeiler_lehman_graph_hash(molecule, node_attr='element'), int(count)))
    return tuple(sorted(signatures))


def select_structures(config, model_config):
    """Sample refcodes, then isolate connected family/chemistry groups and deduplicate."""
    rng = np.random.default_rng(config['seed'])
    records, audit, grouping = [], {}, nx.Graph()
    manifest_dir = Path(model_config['graph_manifest_dir'])
    cutoff = model_config['crystal_encoder']['cutoff']
    from lvebcsp.train.jepa_setup import graph_pxrd_config
    pxrd_config = graph_pxrd_config(model_config) if model_config.get('condition_source', 'conformer') != 'conformer' else None
    for split_id, split in enumerate(('train', 'valid', 'test')):
        dataset = CrystalDataset(manifest_dir / f'{split}.npz', cutoff=cutoff,
                                 context_translation_std=model_config.get('context_translation_std', 0.0),
                                 context_rotation_degrees=model_config.get('context_rotation_degrees', 0.0),
                                 pxrd_config=pxrd_config)
        _, first = np.unique(dataset.refcode, return_index=True)
        by_family = defaultdict(list)
        for index in rng.permutation(first):
            by_family[str(dataset.family[index])].append(int(index))
        families = list(by_family)
        rng.shuffle(families)
        # Round-robin over families avoids filling the sample with one large family.
        chosen = [by_family[f][depth] for depth in range(max(map(len, by_family.values())))
                  for f in families if depth < len(by_family[f])][:config[f'{split}_samples']]
        failures = []
        for n, index in enumerate(chosen):
            item = dataset[index]
            try:
                label, unwrapped, groups, clash = packing_descriptor(item['target'], cutoff, config['bins'])
                if len(np.unique(groups)) != int(item['multiplicity'].sum()):
                    raise ValueError('Copy IDs do not match multiplicities')
            except ValueError as exc:
                failures.append({'refcode': item['refcode'], 'reason': str(exc)})
                continue
            if clash < config['minimum_clash_ratio']:
                failures.append({'refcode': item['refcode'], 'reason': 'Severe inter-block overlap'})
                continue
            signature = block_signature(item)
            chemistry = tuple(sorted({h for h, _ in signature}))
            family_node, chemistry_node = ('family', item['family']), ('chemistry', chemistry)
            grouping.add_edge(family_node, chemistry_node)
            graph = item['target']
            records.append(dict(split=split, split_id=split_id, index=index,
                                source=dataset.sources[int(dataset.source[index])], row=int(dataset.row[index]),
                                item=item, label=label, unwrapped=unwrapped, groups=groups,
                                signature=signature, family_node=family_node,
                                structure=Structure(graph.cell[0].numpy(), graph.z.tolist(), graph.pos.numpy(), coords_are_cartesian=True)))
            if (n + 1) % 250 == 0:
                print(f'{split}: prepared {n + 1}/{len(chosen)} distinct refcodes', flush=True)
        audit[split] = dict(manifest=str(dataset.path), manifest_sha256=hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
                            manifest_rows=len(dataset), unique_refcodes=len(first), selected_refcodes=len(chosen), excluded_geometry=failures)
    component = {node: i for i, nodes in enumerate(nx.connected_components(grouping)) for node in nodes}
    owner = {}
    for record in records:
        group = component[record['family_node']]
        record['group'] = group
        owner[group] = min(owner.get(group, 2), record['split_id'])
    matcher = StructureMatcher(primitive_cell=True, scale=False, attempt_supercell=False)
    retained, representatives = [], defaultdict(list)
    for record in records:
        split_audit = audit[record['split']]
        if record['split_id'] != owner[record['group']]:
            split_audit['excluded_overlap'] = split_audit.get('excluded_overlap', 0) + 1
            continue
        key = tuple(h for h, _ in record['signature'])
        if any(matcher.fit(record['structure'], other) for other in representatives[key]):
            split_audit['excluded_equivalent'] = split_audit.get('excluded_equivalent', 0) + 1
            continue
        representatives[key].append(record['structure'])
        record['parent'] = len(retained)
        retained.append(record)
    for split in audit:
        selected = [r for r in retained if r['split'] == split]
        audit[split]['retained_structures'] = len(selected)
        audit[split]['retained_groups'] = len({r['group'] for r in selected})
        print(f'{split}: {len(selected)} structures in {audit[split]["retained_groups"]} independent groups', flush=True)
    return retained, audit


def controlled_variants(record, config, cutoff):
    rng = np.random.default_rng(config['seed'] + 1009 * record['parent'])
    graph, positions, groups = record['item']['target'], record['unwrapped'], record['groups']
    count = groups.max() + 1
    for kind, strengths in (('translation', config['translation_std']), ('rotation', config['rotation_degrees'])):
        if kind == 'translation' and count < 2:
            continue
        if kind == 'rotation' and all(np.sum(groups == g) == 1 for g in range(count)):
            continue
        for strength in strengths:
            for draw in range(config['variant_draws']):
                changed = positions.copy()
                if kind == 'translation':
                    shifts = rng.normal(scale=strength, size=(count, 3))
                    changed += (shifts - shifts.mean(0))[groups]
                else:
                    axes = rng.normal(size=(count, 3))
                    rotations = Rotation.from_rotvec(axes / np.linalg.norm(axes, axis=1, keepdims=True) * np.deg2rad(strength)).as_matrix()
                    for group, rotation in enumerate(rotations):
                        selected = groups == group
                        center = positions[selected].mean(0)
                        changed[selected] = (positions[selected] - center) @ rotation.T + center
                yield kind, strength, draw, rebuild_graph(graph, changed, cutoff)


@torch.inference_mode()
def export_features(records, models, config, output, device):
    cutoff = models['trained'].context_encoder.config.cutoff
    rng = np.random.default_rng(config['seed'])
    variant_parents = set()
    for split in ('train', 'valid', 'test'):
        parents = [r['parent'] for r in records if r['split'] == split]
        variant_parents.update(rng.choice(parents, min(len(parents), config[f'{split}_variant_parents']), replace=False).tolist())
    has_context = models['trained'].prediction_target == 'crystal_tokens'
    arrays = {name: [] for name in [*models, *(['context'] if has_context else []), 'labels']}
    metadata, pending, original_queries = [], [], {}
    audit = dict(skipped_variants=[], label_invariance=[], representation_invariance=[])

    def flush():
        if not pending:
            return
        batch = batch_to_device(collate_crystal_graphs([item for item, _, _ in pending]), device)
        for name, model in models.items():
            arrays[name].append(model.encode_tgt(batch['target']).cpu().numpy().copy())
        if has_context:
            query = models['trained'].encode_ctx(batch['context'], batch['multiplicity'], batch['block_batch'],
                                                crystal_context=batch.get('crystal_context'),
                                                **{key: batch[key] for key in ('peak_d', 'peak_i', 'peak_mask')
                                                   if key in batch}).cpu().numpy()
        for index, (_, label, meta) in enumerate(pending):
            parent = meta['parent']
            if has_context:
                if meta['kind'] == 'original':
                    original_queries[parent] = query[index].copy()
                arrays['context'].append(original_queries[parent])
            arrays['labels'].append(label)
            metadata.append(meta)
        pending.clear()

    for record_index, record in enumerate(records):
        item = record['item']
        meta = dict(split=record['split'], group=record['group'], parent=record['parent'],
                    refcode=item['refcode'], family=item['family'], source=record['source'], row=record['row'],
                    signature=record['signature'], kind='original', strength=0.0, draw=0)
        pending.append((item, record['label'], meta))
        if record_index < config['invariance_samples']:
            graph = item['target']
            rotation = Rotation.random(random_state=rng).as_matrix()
            changed = rebuild_graph(graph, graph.pos.numpy() @ rotation.T + rng.normal(size=3), cutoff,
                                    cell=graph.cell[0].numpy() @ rotation.T)
            label, _, _, _ = packing_descriptor(changed, cutoff, config['bins'])
            error = float(np.max(np.abs(label - record['label'])))
            audit['label_invariance'].append(error)
            if error > 1e-5:
                raise ValueError(f'Packing labels changed under a rigid transform: {item["refcode"]}, max error={error}')
            z = models['trained'].encode_tgt(graph.clone().to(device))
            z_rot = models['trained'].encode_tgt(changed.to(device))
            audit['representation_invariance'].append(float((z - z_rot).square().mean()))
        if record['parent'] in variant_parents:
            for kind, strength, draw, changed in controlled_variants(record, config, cutoff):
                variant_meta = {**meta, 'kind': kind, 'strength': float(strength), 'draw': draw}
                try:
                    label, _, _, clash = packing_descriptor(changed, cutoff, config['bins'])
                    if clash < config['minimum_clash_ratio']:
                        raise ValueError('Severe inter-block overlap')
                except ValueError as exc:
                    audit['skipped_variants'].append({**variant_meta, 'reason': str(exc)})
                    continue
                pending.append(({**item, 'target': changed}, label, variant_meta))
                if len(pending) >= config['batch_size']:
                    flush()
        if len(pending) >= config['batch_size']:
            flush()
        if (record_index + 1) % 100 == 0:
            print(f'Encoded {record_index + 1}/{len(records)} parents; {len(metadata)} structures and variants', flush=True)
    flush()
    cache = {name: np.concatenate(arrays[name]) for name in models}
    cache['labels'] = np.stack(arrays['labels'])
    if has_context:
        cache['context'] = np.stack(arrays['context'])
    np.savez_compressed(output / 'features.npz', **cache)
    (output / 'samples.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'geometry_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return cache, metadata


def group_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weight = 1 / counts[inverse]
    return weight / weight.sum()


def fit_mlp(features, targets, metadata, config, seed, device, shuffle_labels=False):
    # Preserve every token feature; flattening for an MLP does not pool tokens.
    features = features.reshape(len(features), -1)
    torch.manual_seed(seed)
    indices = {split: np.array([i for i, m in enumerate(metadata) if m['split'] == split]) for split in ('train', 'valid', 'test')}
    groups = np.array([m['group'] for m in metadata])
    train, valid = indices['train'], indices['valid']
    weight = group_weights(groups[train])
    x_mean = np.sum(features[train] * weight[:, None], 0)
    x_std = np.sqrt(np.sum((features[train] - x_mean) ** 2 * weight[:, None], 0)).clip(1e-6)
    y_mean = np.sum(targets[train] * weight[:, None], 0)
    y_std = np.sqrt(np.sum((targets[train] - y_mean) ** 2 * weight[:, None], 0))
    active = y_std > 1e-6
    x = torch.tensor((features - x_mean) / x_std, dtype=torch.float32, device=device)
    y = torch.tensor((targets[:, active] - y_mean[active]) / y_std[active], dtype=torch.float32, device=device)
    if shuffle_labels:
        y[train] = y[np.random.default_rng(seed).permutation(train)].clone()
    train_index, valid_index = torch.tensor(train, device=device), torch.tensor(valid, device=device)
    train_weight = torch.tensor(weight * len(train), dtype=torch.float32, device=device)
    valid_weight = torch.tensor(group_weights(groups[valid]), dtype=torch.float32, device=device)
    head = nn.Sequential(nn.Linear(features.shape[1], config['hidden_dim']), nn.GELU(), nn.Linear(config['hidden_dim'], int(active.sum()))).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, waiting, history, best_state, best_epoch = float('inf'), 0, [], None, -1
    for epoch in range(config['max_epochs']):
        head.train()
        order = torch.randperm(len(train), device=device)
        for chosen in order.split(config['probe_batch_size']):
            batch = train_index[chosen]
            loss = ((head(x[batch]) - y[batch]).square().mean(-1) * train_weight[chosen]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            valid_loss = float(((head(x[valid_index]) - y[valid_index]).square().mean(-1) * valid_weight).sum())
        history.append(valid_loss)
        if valid_loss < best:
            best, waiting, best_epoch = valid_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            waiting += 1
        if waiting >= config['patience']:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        predictions = np.broadcast_to(y_mean, targets.shape).copy()
        predictions[:, active] = head(x).cpu().numpy() * y_std[active] + y_mean[active]
    fitted = dict(state_dict=best_state, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, active=active,
                  seed=seed, best_epoch=best_epoch, validation_loss=best, validation_history=history,
                  hidden_dim=config['hidden_dim'], input_dim=features.shape[1])
    return predictions.astype(np.float32), fitted


def evaluate_predictions(prediction, target, metadata, split, contact_bins, direct_changes=False):
    original = {m['parent']: i for i, m in enumerate(metadata) if m['kind'] == 'original'}
    result = {}
    errors = {}
    for kind in (('variant',) if direct_changes else ('original', 'variant')):
        rows = np.array([i for i, m in enumerate(metadata) if m['split'] == split and (m['kind'] == 'original') == (kind == 'original')], dtype=int)
        if not len(rows):
            continue
        groups = np.array([metadata[i]['group'] for i in rows])
        weights = group_weights(groups)
        y, pred = target[rows].astype(float), prediction[rows].astype(float)
        if kind == 'variant' and not direct_changes:
            parents = np.array([original[metadata[i]['parent']] for i in rows])
            y -= target[parents]
            pred -= prediction[parents]
        mse_by_row = ((pred - y) ** 2).mean(-1)
        mse = float(weights @ mse_by_row)
        mae = float(weights @ np.abs(pred - y).mean(-1))
        contact_mae = float(weights @ np.abs((pred - y)[:, :contact_bins].sum(-1)))
        stats = dict(samples=len(rows), groups=len(np.unique(groups)), histogram_mse=mse, histogram_mae=mae, contact_count_mae=contact_mae)
        if kind == 'original':
            variance = np.sum((y - np.sum(y * weights[:, None], 0)) ** 2 * weights[:, None], 0)
            active = variance > 1e-10
            per_bin_r2 = 1 - np.sum((pred - y) ** 2 * weights[:, None], 0)[active] / variance[active]
            stats['r2'] = float(per_bin_r2.mean())
            stats['r2_variance_weighted'] = float(1 - np.sum((pred - y) ** 2 * weights[:, None]) / variance.sum())
            stats['r2_bins'] = np.flatnonzero(active).tolist()
            stats['r2_per_bin'] = per_bin_r2.tolist()
        else:
            zero_mse = float(weights @ (y ** 2).mean(-1))
            stats['zero_change_mse'] = zero_mse
            stats['delta_mse_reduction'] = 1 - mse / zero_mse if zero_mse else None
        result[kind] = stats
        errors[kind] = {int(group): float(mse_by_row[groups == group].mean()) for group in np.unique(groups)}
    return result, errors


def write_report(output, identity, runs, per_group, change_runs, change_errors):
    rng = np.random.default_rng(2026)
    summary = {}
    metadata = json.loads((output / 'samples.json').read_text())
    counts = {split: dict(originals=sum(m['split'] == split and m['kind'] == 'original' for m in metadata),
                          variants=sum(m['split'] == split and m['kind'] != 'original' for m in metadata),
                          groups=len({m['group'] for m in metadata if m['split'] == split}))
              for split in ('train', 'valid', 'test')}
    for title, results, errors, kind in (
        ('absolute_profile', runs, per_group, 'original'),
        ('change_from_profile_probe', runs, per_group, 'variant'),
        ('direct_change_probe', change_runs, change_errors, 'variant'),
    ):
        summary[title] = {}
        for name, prefix in (('trained', 'trained_seed'), ('context', 'context_seed'), ('random', 'random_'),
                             ('mean', 'mean'), ('zero', 'zero'), ('shuffled_labels', 'shuffled_labels')):
            labels = [label for label in results if label == prefix or label.startswith(prefix)]
            labels = [label for label in labels if kind in results[label]]
            if not labels:
                continue
            values = [results[label][kind] for label in labels]
            metrics = {key: dict(mean=float(np.mean([v[key] for v in values])),
                                std=float(np.std([v[key] for v in values])))
                       for key in ('histogram_mae', 'histogram_mse', 'contact_count_mae', 'r2',
                                   'r2_variance_weighted', 'delta_mse_reduction') if key in values[0]}
            metrics.update(runs=len(labels), samples=values[0]['samples'], groups=values[0]['groups'])
            if kind == 'variant':
                reference = 'zero' if 'zero' in results else 'mean'
                group_ids = sorted(errors[reference][kind], key=int)
                zero_error = np.array([errors[reference][kind][group] for group in group_ids])
                model_error = np.mean([[errors[label][kind][group] for group in group_ids] for label in labels], 0)
                draws = rng.integers(len(group_ids), size=(2000, len(group_ids)))
                improvement = 1 - model_error[draws].mean(1) / zero_error[draws].mean(1)
                metrics['delta_reduction_95ci'] = np.quantile(improvement, [.025, .975]).tolist()
            summary[title][name] = metrics
    summary['sample_counts'] = counts
    # Paired uncertainty in the error gap uses the same groups for both methods.
    summary['trained_vs_random'] = {}
    for title, errors, kind in (('absolute_profile', per_group, 'original'), ('direct_change_probe', change_errors, 'variant')):
        trained_names = [name for name in errors if name.startswith('trained_seed')]
        random_names = [name for name in errors if name.startswith('random_')]
        group_ids = sorted(errors[trained_names[0]][kind], key=int)
        trained_error = np.mean([[errors[name][kind][g] for g in group_ids] for name in trained_names], 0)
        random_error = np.mean([[errors[name][kind][g] for g in group_ids] for name in random_names], 0)
        difference = trained_error - random_error
        draws = rng.integers(len(group_ids), size=(2000, len(group_ids)))
        summary['trained_vs_random'][title] = dict(mse_gap=float(difference.mean()),
            mse_gap_95ci=np.quantile(difference[draws].mean(1), [.025, .975]).tolist(),
            interpretation='Positive means the trained representation gives larger probe errors than random encoders.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [f'Frozen encoder MLP probe: checkpoint epoch={identity["epoch"]} '
             f'({identity["epoch"] + 1} completed epochs), step={identity["global_step"]}.', '',
             f'Feature shape per crystal: {identity.get("feature_shape", "legacy pooled vector")}. '
             'Token arrays are flattened without pooling for the MLP.', '',
             f'Context translation std: {identity.get("context_translation_std", 0.0)} angstroms; '
             f'rotation: {identity.get("context_rotation_degrees", 0.0)} degrees. '
             'Contexts retain all atoms with independently perturbed molecular poses.', '',
             'All encoders were frozen and evaluated in FP32. Readouts have one 128-unit GELU hidden layer. '
             'Constant descriptor bins are filled with their training mean; the MLP predicts the remaining bins. '
             'Hyperparameters and early stopping use validation data. Test errors are balanced over connected family/chemistry groups.', '',
             f'Original structures: {counts["train"]["originals"]} training / {counts["valid"]["originals"]} validation / '
             f'{counts["test"]["originals"]} test. Controlled variants: {counts["train"]["variants"]} / '
             f'{counts["valid"]["variants"]} / {counts["test"]["variants"]}, with every parent kept in its original split.', '',
             '| Input | Original-profile MAE | Original-profile R² (variance weighted) | Contact-count MAE (<4 Å) |',
             '|---|---:|---:|---:|']
    for name, metrics in summary['absolute_profile'].items():
        lines.append(f'| {name} | {metrics["histogram_mae"]["mean"]:.4f} | '
                     f'{metrics["r2_variance_weighted"]["mean"]:.4f} | {metrics["contact_count_mae"]["mean"]:.4f} |')
    lines += ['', 'The direct-change task fits a separate MLP to embedding differences and measured descriptor differences. '
              'Both are standardized using training pairs only, so small geometric signals can be amplified without test-set statistics. '
              'The context is held fixed within each parent, producing a zero input difference; its constant-readout baseline is the training mean change.', '',
              '| Input to change probe | Change MSE reduction versus predicting zero | 95% group-bootstrap interval | Change contact-count MAE |',
              '|---|---:|---:|---:|']
    for name, metrics in summary['direct_change_probe'].items():
        low, high = metrics['delta_reduction_95ci']
        lines.append(f'| {name} | {metrics["delta_mse_reduction"]["mean"]:.2%} | '
                     f'[{low:.2%}, {high:.2%}] | {metrics["contact_count_mae"]["mean"]:.4f} |')
    lines += ['', 'A zero change score matches predicting no change; positive values improve on that baseline and negative values are worse. '
              'The confidence interval resamples held-out groups after averaging errors across seeds. Random results pool three random encoders and three MLP seeds per encoder.', '',
              'Original and paired results answer different questions. Predicting typical contact profiles can use molecular identity. '
              'Predicting differences between variants at fixed molecules, counts, internal geometry, and cell tests recoverable arrangement information.', '',
              'Grouping uses refcode families and inferred heavy-atom connectivity, with conflicting evaluation groups and equivalent structures removed. '
              'This audit covers the sampled probe data, not all chemical identities in the encoder-pretraining corpus. '
              'Synthetic changes are unrelaxed translations/rotations and are not verified polymorphs. '
              'Neither a successful descriptor probe nor a failed small readout establishes full structure-generation capability or proves all geometry absent.', '',
              f'[Per-run metrics]({output / "results.json"})', f'[Split audit]({output / "split_audit.json"})',
              f'[Geometry audit]({output / "geometry_audit.json"})', f'[Configuration]({output / "probe_config.yaml"})']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/crystal_encoder_probe.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(config.get('cpu_threads', 2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'probe_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    payload = Path(args.checkpoint).read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    started = time.monotonic()
    identity = {key: checkpoint.get(key) for key in ('epoch', 'global_step', 'source_checkpoint', 'source_sha256')}
    identity['snapshot_sha256'] = hashlib.sha256(payload).hexdigest()
    identity['context_translation_std'] = checkpoint['config'].get('context_translation_std', 0.0)
    identity['context_rotation_degrees'] = checkpoint['config'].get('context_rotation_degrees', 0.0)
    identity['prediction_target'] = checkpoint['config'].get('prediction_target', 'crystal_tokens')
    if not args.fit_only and Path(args.checkpoint).resolve() != output / 'frozen_checkpoint.pt':
        (output / 'frozen_checkpoint.pt').write_bytes(payload)
    del payload
    if args.fit_only:
        saved = json.loads((output / 'checkpoint_identity.json').read_text())
        if saved['snapshot_sha256'] != identity['snapshot_sha256']:
            raise ValueError('Feature cache belongs to a different checkpoint')
        identity = saved
        with np.load(output / 'features.npz') as arrays:
            cache = dict(arrays)
        metadata = json.loads((output / 'samples.json').read_text())
    else:
        models = {'trained': build_jepa_from_config(checkpoint['config']).to(device).eval()}
        load_jepa_weights(models['trained'], checkpoint['model'])
        identity['source_had_pooled_readout'] = any(k.startswith('context_encoder.readout') for k in checkpoint['model'])
        for seed in config['random_encoder_seeds']:
            torch.manual_seed(seed)
            models[f'random_{seed}'] = build_jepa_from_config(checkpoint['config']).to(device).eval()
        for model in models.values():
            model.requires_grad_(False)
        (output / 'checkpoint_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        print(f'Frozen checkpoint epoch={checkpoint["epoch"]} step={checkpoint["global_step"]}', flush=True)
        records, audit = select_structures(config, checkpoint['config'])
        (output / 'split_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        cache, metadata = export_features(records, models, config, output, device)
        identity['feature_shape'] = list(cache['trained'].shape[1:])
        (output / 'checkpoint_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
        assert all(p.grad is None for model in models.values() for p in model.parameters())
        del records, models
        torch.cuda.empty_cache()
    if args.prepare_only:
        print(f'Feature cache saved in {output}', flush=True)
        return
    target = cache['labels']
    groups = np.array([m['group'] for m in metadata])
    train = np.array([i for i, m in enumerate(metadata) if m['split'] == 'train'])
    mean = np.sum(target[train] * group_weights(groups[train])[:, None], 0)
    contact_bins = int(round(4.0 / checkpoint['config']['crystal_encoder']['cutoff'] * config['bins']))
    runs, per_group, predictions = {}, {}, {'mean': np.broadcast_to(mean, target.shape).copy()}
    runs['mean'], per_group['mean'] = evaluate_predictions(predictions['mean'], target, metadata, 'test', contact_bins)
    for feature in ['trained', 'context', *[f'random_{s}' for s in config['random_encoder_seeds']]]:
        if feature not in cache:
            continue
        for seed in config['probe_seeds']:
            label = f'{feature}_seed{seed}'
            prediction, fitted = fit_mlp(cache[feature], target, metadata, config, seed, device)
            predictions[label] = prediction
            scores, errors = evaluate_predictions(prediction, target, metadata, 'test', contact_bins)
            scores.update(best_epoch=fitted['best_epoch'], validation_loss=fitted['validation_loss'])
            runs[label], per_group[label] = scores, errors
            torch.save(fitted, output / f'{label}.pt')
            print(f'{label}: epoch={fitted["best_epoch"] + 1} original_R2={scores["original"]["r2"]:.4f} '
                  f'delta_reduction={scores.get("variant", {}).get("delta_mse_reduction")}', flush=True)
            (output / 'results.json').write_text(json.dumps(dict(checkpoint=identity, runs=runs, elapsed_seconds=time.monotonic()-started), indent=2) + '\n')
    prediction, fitted = fit_mlp(cache['trained'], target, metadata, config, config['probe_seeds'][0], device, shuffle_labels=True)
    predictions['shuffled_labels'] = prediction
    runs['shuffled_labels'], per_group['shuffled_labels'] = evaluate_predictions(prediction, target, metadata, 'test', contact_bins)
    torch.save(fitted, output / 'shuffled_labels.pt')
    np.savez_compressed(output / 'predictions.npz', **predictions)
    (output / 'group_errors.json').write_text(json.dumps(per_group, indent=2) + '\n')
    original = {m['parent']: i for i, m in enumerate(metadata) if m['kind'] == 'original'}
    rows = np.array([i for i, m in enumerate(metadata) if m['kind'] != 'original'])
    parents = np.array([original[metadata[i]['parent']] for i in rows])
    change_metadata = [metadata[i] for i in rows]
    changes = target[rows] - target[parents]
    change_train = np.array([i for i, m in enumerate(change_metadata) if m['split'] == 'train'])
    change_groups = np.array([m['group'] for m in change_metadata])
    mean_change = np.sum(changes[change_train] * group_weights(change_groups[change_train])[:, None], 0)
    change_predictions = dict(zero=np.zeros_like(changes), mean=np.broadcast_to(mean_change, changes.shape).copy())
    change_runs, change_errors = {}, {}
    for name, prediction in change_predictions.items():
        change_runs[name], change_errors[name] = evaluate_predictions(prediction, changes, change_metadata, 'test', contact_bins, direct_changes=True)
    for feature in ['trained', *[f'random_{s}' for s in config['random_encoder_seeds']]]:
        features = cache[feature][rows] - cache[feature][parents]
        for seed in config['probe_seeds']:
            label = f'{feature}_seed{seed}'
            prediction, fitted = fit_mlp(features, changes, change_metadata, config, seed, device)
            change_predictions[label] = prediction
            scores, errors = evaluate_predictions(prediction, changes, change_metadata, 'test', contact_bins, direct_changes=True)
            scores.update(best_epoch=fitted['best_epoch'], validation_loss=fitted['validation_loss'])
            change_runs[label], change_errors[label] = scores, errors
            torch.save(fitted, output / f'change_{label}.pt')
            print(f'change_{label}: epoch={fitted["best_epoch"] + 1} reduction={scores["variant"]["delta_mse_reduction"]:.4f}', flush=True)
            (output / 'results.json').write_text(json.dumps(dict(checkpoint=identity, runs=runs, change_runs=change_runs, elapsed_seconds=time.monotonic()-started), indent=2) + '\n')
    prediction, fitted = fit_mlp(cache['trained'][rows] - cache['trained'][parents], changes, change_metadata, config, config['probe_seeds'][0], device, shuffle_labels=True)
    change_predictions['shuffled_labels'] = prediction
    change_runs['shuffled_labels'], change_errors['shuffled_labels'] = evaluate_predictions(prediction, changes, change_metadata, 'test', contact_bins, direct_changes=True)
    torch.save(fitted, output / 'change_shuffled_labels.pt')
    np.savez_compressed(output / 'change_predictions.npz', **change_predictions)
    (output / 'change_group_errors.json').write_text(json.dumps(change_errors, indent=2) + '\n')
    (output / 'results.json').write_text(json.dumps(dict(checkpoint=identity, runs=runs, change_runs=change_runs, elapsed_seconds=time.monotonic()-started), indent=2) + '\n')
    write_report(output, identity, runs, per_group, change_runs, change_errors)
    print(f'Probe results saved in {output}', flush=True)


if __name__ == '__main__':
    main()
