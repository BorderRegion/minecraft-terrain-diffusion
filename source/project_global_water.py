"""Experimental whole-region hydrostatic decoding with unchanged ground voxels."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects, label


def spill_heights(ground, ocean_level):
    """Minimum escape elevation, with the sampled ocean level at the outer boundary."""
    h, w = ground.shape
    filled = np.empty_like(ground)
    seen = np.zeros(ground.shape, dtype=bool)
    queue = []
    for z, x in [(z, x) for z in range(h) for x in (0, w-1)] + [(z, x) for z in (0, h-1) for x in range(w)]:
        if not seen[z, x]:
            seen[z, x] = True
            level = max(int(ground[z, x]), ocean_level)
            filled[z, x] = level
            heapq.heappush(queue, (level, z, x))
    while queue:
        level, z, x = heapq.heappop(queue)
        for zz, xx in ((z-1, x), (z+1, x), (z, x-1), (z, x+1)):
            if 0 <= zz < h and 0 <= xx < w and not seen[zz, xx]:
                seen[zz, xx] = True
                elevation = max(level, int(ground[zz, xx]))
                filled[zz, xx] = elevation
                heapq.heappush(queue, (elevation, zz, xx))
    return filled


def open_water_faces(ground, levels):
    """Count internal wet faces exposed above lower neighboring ground or water."""
    wet = levels > ground
    violations = 0
    for axis in (0, 1):
        a = [slice(None)]*2; b = [slice(None)]*2
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        for left, right in ((a, b), (b, a)):
            top_right = np.where(wet[right], levels[right], ground[right])
            violations += int((wet[left] & (top_right < levels[left])).sum())
    return violations


def project(raw, ocean_level, minimum_seeds=4, minimum_seed_fraction=.01):
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[0] != 9 or min(raw.shape[1:]) < 3 or not np.isfinite(raw).all():
        raise ValueError('expected a finite nine-channel terrain region')
    if not -64 <= ocean_level <= 319 or minimum_seeds < 1 or not 0 <= minimum_seed_fraction <= 1:
        raise ValueError('invalid water projection configuration')
    ground = np.rint(raw[1]).astype(np.int16)
    if ground.min() < -64 or ground.max() > 319:
        raise ValueError('ground exceeds Minecraft bounds')
    source_wet = raw[0] > raw[1]+1
    filled = spill_heights(ground, int(ocean_level))
    basins, count = label(filled > ground)
    water_level = ground.copy()
    accepted = []
    for basin_id, bounds in enumerate(find_objects(basins), 1):
        if bounds is None:
            continue
        mask = basins[bounds] == basin_id
        seeds = source_wet[bounds] & mask
        if seeds.sum() < max(minimum_seeds, int(np.ceil(mask.sum()*minimum_seed_fraction))):
            continue
        capacity = np.unique(filled[bounds][mask])
        if len(capacity) != 1:
            raise AssertionError('connected depression has inconsistent spill elevation')
        touches_edge = ((bounds[0].start == 0 and mask[0].any()) or
                        (bounds[0].stop == ground.shape[0] and mask[-1].any()) or
                        (bounds[1].start == 0 and mask[:, 0].any()) or
                        (bounds[1].stop == ground.shape[1] and mask[:, -1].any()))
        proposed = int(ocean_level) if touches_edge else int(np.rint(np.median(raw[0][bounds][seeds])))
        level = min(proposed, int(capacity[0]))
        candidates, _ = label(mask & (ground[bounds] < level))
        seed_counts = np.bincount(candidates[seeds], minlength=int(candidates.max())+1)
        choose = seed_counts >= minimum_seeds
        choose[0] = False
        selected = choose[candidates]
        if not selected.any():
            continue
        water_level[bounds][selected] = level
        accepted.append({'basin': basin_id, 'spill_y': int(capacity[0]), 'water_top_y': level,
                         'ocean_boundary_connected': bool(touches_edge), 'water_columns': int(selected.sum())})
    wet = water_level > ground
    violations = open_water_faces(ground, water_level)
    if violations:
        raise AssertionError(f'{violations} internal water faces remain open')
    output = raw.copy()
    output[1] = ground
    output[0] = ground
    # The existing export format detects water using a >1 float-depth threshold.
    # This epsilon preserves one-block shallows while exporting integer water tops.
    output[0, wet] = np.minimum(water_level[wet]+.01, 319)
    shallow = wet & (output[0]-output[1] <= 1)
    output[1, shallow] -= .01
    if not np.array_equal(np.rint(output[1]), np.rint(raw[1])) or not np.array_equal(output[2:], raw[2:]):
        raise AssertionError('ground voxel geometry or materials changed')
    _, components = label(wet)
    report = {'ocean_top_water_block_y': int(ocean_level), 'depressions': count, 'selected_basins': accepted,
              'raw_wet_fraction': float(source_wet.mean()), 'projected_wet_fraction': float(wet.mean()),
              'wet_mask_changed_fraction': float((wet != source_wet).mean()),
              'water_columns_added': int((wet & ~source_wet).sum()), 'water_columns_removed': int((source_wet & ~wet).sum()),
              'water_components': int(components), 'internal_open_water_faces': violations,
              'ground_voxels_changed': int((np.rint(output[1]) != np.rint(raw[1])).sum()),
              'ground_float_quantization_rmse': float(np.sqrt(np.mean((output[1]-raw[1])**2))),
              'boundary_water_columns': int(wet[0].sum()+wet[-1].sum()+wet[1:-1, 0].sum()+wet[1:-1, -1].sum()),
              'minimum_seed_columns': minimum_seeds, 'minimum_seed_fraction': minimum_seed_fraction,
              'note': 'Experimental inference-time hydrology, not learned output. Ground/material voxels retained; wet extent can change. Internal static face closure does not prove Minecraft stability or behavior beyond the clipped region.'}
    return output, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sample', required=True)
    p.add_argument('--prior', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    sample, out = Path(a.sample), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads(Path(a.prior).read_text())
    raw = np.load(sample/'global_field.npy')
    output, report = project(raw, prior['modal_water_top_block_y'])
    report['input_field_sha256'] = hashlib.sha256((sample/'global_field.npy').read_bytes()).hexdigest()
    report['training_prior_sha256'] = hashlib.sha256(Path(a.prior).read_bytes()).hexdigest()
    np.save(out/'global_field.npy', output)
    from sample import export_schematic
    from render_3d import render
    manifest = json.loads((sample/'manifest.json').read_text())
    whole = export_schematic(output, out/'terrain_full.schem', decorate=True, seed=manifest['seed'])
    manifest.update({'full': whole, 'water_decoder': 'experimental_seeded_global_hydrostatic_v1', 'parent_sample': str(sample)})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out/'projection-report.json').write_text(json.dumps(report, indent=2))
    render(output, out/'terrain-3d.png')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
